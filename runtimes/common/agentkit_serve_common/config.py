"""Typed loader for the frozen ``/agent/agent.yaml`` ABI.

This mirrors ``docs/agent-abi.md`` EXACTLY — it is the Python (reader) half of the
contract whose Go (writer) half lives in ``pkg/agentkit/abi``. The shape here is
the *baked* file: ``instructions`` is already a fully-resolved scalar string and
``model.provider`` is always ``openai-compatible`` in v0.
"""

from __future__ import annotations

import hashlib
import json
import math
from decimal import Decimal
import os
import posixpath
import re
import sys
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

# The ABI schema version this reader understands (agent-abi.md: ``abiVersion: v0``).
ABI_VERSION = "v0"
_PROVIDER_OPENAI_COMPATIBLE = "openai-compatible"
_ENV_NAME_RE = re.compile(r"^[A-Z0-9_]+$")
_HEADER_NAME_RE = re.compile(r"^[A-Za-z0-9!#$%&'*+\-.^_`|~]+$")
_BROKERED_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_SECRET_VALUE_PREFIXES = ("sk-", "sk_", "ghp_", "github_pat_", "xoxb-", "AKIA")
_TOOL_TYPE_MCP = "mcp"
_TRANSPORT_STDIO = "stdio"
_TRANSPORT_STREAMABLE_HTTP = "streamable-http"
_CONTEXT_TYPE_SEARCH = "search"
_CONTEXT_TYPE_SKILLS = "skills"
_CONTEXT_TYPE_MEMORY = "memory"
_CONTEXT_SOURCE_FILESYSTEM = "filesystem"
_CONTEXT_SOURCE_MCP = "mcp"
_AUTH_BEARER = "bearer"
_AUTH_WORKLOAD_IDENTITY = "workload-identity-token"
_APPROVAL_VALUES = {"never", "auto", "always"}
_CREDENTIAL_HEADER_NAMES = {
    "authorization",
    "proxy-authorization",
    "cookie",
    "set-cookie",
    "x-api-key",
    "api-key",
    "ocp-apim-subscription-key",
    "subscription-key",
    "x-functions-key",
}
_BROKERED_CLASSES = {"read", "write", "coordination"}
_BROKERED_SCHEMA_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_BROKERED_SCHEMA_BYTES = 64 * 1024
_UNSAFE_BROKERED_FIELD_NAMES = {
    "auth",
    "authorization",
    "apikey",
    "bearer",
    "cookie",
    "credential",
    "credentials",
    "endpoint",
    "endpoints",
    "executionendpoint",
    "executionurl",
    "header",
    "headers",
    "ocpapimsubscriptionkey",
    "password",
    "proxyauthorization",
    "secret",
    "secretref",
    "setcookie",
    "subscriptionkey",
    "token",
    "tokens",
    "url",
    "urls",
    "xapikey",
    "xfunctionskey",
}



class _Strict(BaseModel):
    """Base model that forbids unknown keys so a writer/reader drift surfaces loudly."""

    model_config = ConfigDict(extra="forbid")


def _validate_env_name(value: str, *, field: str) -> str:
    """Validate an env var NAME from the ABI, never a secret value."""
    if not value or not _ENV_NAME_RE.fullmatch(value):
        raise ValueError(f"{field} must be an env var NAME matching [A-Z0-9_]+")
    if value.startswith(_SECRET_VALUE_PREFIXES):
        raise ValueError(f"{field} looks like a secret value; provide an env var NAME")
    return value


def _looks_like_secret_literal(value: str) -> bool:
    if not value:
        return False
    return value.startswith(_SECRET_VALUE_PREFIXES)


def _contains_secret_prefix(value: str) -> bool:
    return any(prefix in value for prefix in _SECRET_VALUE_PREFIXES)


def _unsafe_brokered_text(value: str) -> bool:
    lowered = value.lower()
    normalized = re.sub(r"[^a-z0-9]", "", lowered)
    return (
        (_looks_like_secret_literal(value) or _contains_secret_prefix(value))
        or "://" in value
        or re.search(r"\bbearer\b", lowered) is not None
        or re.search(r"\bbasic\b", lowered) is not None
        or "authorization" in lowered
        or "secret" in lowered
        or "token" in lowered
        or "password" in lowered
        or "passphrase" in lowered
        or "pwd" in lowered
        or "api key" in lowered
        or "apikey" in lowered
        or "apikey" in normalized
        or "xapikey" in normalized
        or "subscriptionkey" in normalized
        or "xfunctionskey" in normalized
        or "cookie" in lowered
        or "set-cookie" in lowered
        or "x-api-key" in lowered
        or "api-key" in lowered
        or "subscription-key" in lowered
        or "x-functions-key" in lowered
        or "ocp-apim-subscription-key" in lowered
        or "private key" in lowered
        or "privatekey" in lowered
        or "key material" in lowered
        or ".svc" in lowered
        or "cluster.local" in lowered
    )


def _canonical_number(value: int | float) -> str:
    if isinstance(value, bool):
        raise TypeError("boolean is not a JSON number")
    if isinstance(value, int):
        return str(value)
    if not math.isfinite(value):
        raise ValueError("JSON numbers must be finite")
    decimal = Decimal(str(value))
    if decimal == decimal.to_integral_value():
        return format(decimal.to_integral_value(), "f")
    out = format(decimal, "f")
    if "." in out:
        out = out.rstrip("0").rstrip(".")
    return out


def _canonical_json(value: Any) -> str:
    """Return a deterministic JSON representation for digests and drift checks."""

    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _canonical_number(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, list):
        return "[" + ",".join(_canonical_json(item) for item in value) + "]"
    if isinstance(value, Mapping):
        parts: list[str] = []
        for key in sorted(value):
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            parts.append(json.dumps(key, ensure_ascii=False, separators=(",", ":")) + ":" + _canonical_json(value[key]))
        return "{" + ",".join(parts) + "}"
    raise TypeError(f"unsupported JSON value {type(value).__name__}")


def brokered_tool_schema_digest(
    *,
    name: str,
    description: str,
    brokered_class: str,
    parameters: Mapping[str, Any],
) -> str:
    """Digest the exact safe schema surface AgentKit exposes to a model."""

    payload = {
        "name": name,
        "description": description,
        "brokeredClass": brokered_class,
        "parameters": parameters,
    }
    return "sha256:" + hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _unsafe_brokered_key(value: str) -> str | None:
    normalized = re.sub(r"[^A-Za-z0-9]", "", value).lower()
    if normalized in _UNSAFE_BROKERED_FIELD_NAMES:
        return value
    auth_like = (normalized.startswith("auth") and not normalized.startswith("author")) or normalized.endswith("auth")
    if (
        auth_like
        or "authorization" in normalized
        or "header" in normalized
        or "url" in normalized
        or "endpoint" in normalized
        or "cookie" in normalized
        or "secret" in normalized
        or "token" in normalized
        or "password" in normalized
        or "passphrase" in normalized
        or "pwd" in normalized
        or "apikey" in normalized
        or "accesskey" in normalized
        or "privatekey" in normalized
        or "keymaterial" in normalized
    ):
        return value
    if "credential" in normalized or "executionurl" in normalized or "executionendpoint" in normalized:
        return value
    return None


def _reject_unsafe_brokered_schema_keys(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} keys must be strings")
            unsafe = _unsafe_brokered_key(key)
            if unsafe is not None:
                raise ValueError(f"{path}.{unsafe} is not safe for brokered tool schemas")
            if key in {"required", "dependentRequired"}:
                _reject_unsafe_brokered_property_name_values(child, path=f"{path}.{key}")
            _reject_unsafe_brokered_schema_keys(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            _reject_unsafe_brokered_schema_keys(child, path=f"{path}[{idx}]")
    elif isinstance(value, str) and _unsafe_brokered_text(value):
        raise ValueError(f"{path} contains URL or secret-like material")



_JSON_SCHEMA_TYPES = {"object", "string", "integer", "number", "boolean", "array", "null"}
_NUMERIC_SCHEMA_KEYS = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"}
_INTEGER_SCHEMA_KEYS = {"minLength", "maxLength", "minItems", "maxItems", "minProperties", "maxProperties"}


def _validate_json_schema_subset(schema: Any, *, path: str) -> None:
    if not isinstance(schema, Mapping):
        raise ValueError(f"{path} must be a JSON Schema object")
    for unsupported_key in (
        "allOf",
        "anyOf",
        "oneOf",
        "not",
        "$ref",
        "if",
        "then",
        "else",
        "contains",
        "minContains",
        "maxContains",
        "propertyNames",
        "dependentSchemas",
        "patternProperties",
        "unevaluatedProperties",
        "unevaluatedItems",
        "prefixItems",
        "uniqueItems",
    ):
        if unsupported_key in schema:
            raise ValueError(f"{path}.{unsupported_key} is not supported for deterministic brokered tool schemas")
    schema_type = schema.get("type")
    if schema_type is not None:
        if isinstance(schema_type, str):
            if schema_type not in _JSON_SCHEMA_TYPES:
                raise ValueError(f"{path}.type {schema_type!r} is not supported")
        elif isinstance(schema_type, list) and schema_type and all(isinstance(item, str) for item in schema_type):
            unsupported = [item for item in schema_type if item not in _JSON_SCHEMA_TYPES]
            if unsupported:
                raise ValueError(f"{path}.type contains unsupported value {unsupported[0]!r}")
        else:
            raise ValueError(f"{path}.type must be a string or string array")
    if "properties" in schema:
        properties = schema["properties"]
        if not isinstance(properties, Mapping):
            raise ValueError(f"{path}.properties must be an object")
        for name, child in properties.items():
            if not isinstance(name, str):
                raise ValueError(f"{path}.properties keys must be strings")
            _validate_json_schema_subset(child, path=f"{path}.properties.{name}")
    if "items" in schema:
        items = schema["items"]
        if isinstance(items, Mapping):
            _validate_json_schema_subset(items, path=f"{path}.items")
        elif isinstance(items, list):
            raise ValueError(f"{path}.items array form is not supported for brokered tool schemas")
        else:
            raise ValueError(f"{path}.items must be an object")
    if "required" in schema:
        required = schema["required"]
        if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
            raise ValueError(f"{path}.required must be a string array")
    if "dependentRequired" in schema:
        dependent_required = schema["dependentRequired"]
        if not isinstance(dependent_required, Mapping):
            raise ValueError(f"{path}.dependentRequired must be an object")
        for key, value in dependent_required.items():
            if not isinstance(key, str) or not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise ValueError(f"{path}.dependentRequired values must be string arrays")
    if "enum" in schema:
        enum_values = schema["enum"]
        if not isinstance(enum_values, list):
            raise ValueError(f"{path}.enum must be an array")
        if not enum_values:
            raise ValueError(f"{path}.enum must contain at least one value")
    if "pattern" in schema:
        raise ValueError(f"{path}.pattern is not supported for deterministic brokered tool schemas")
    if "additionalProperties" in schema:
        additional_properties = schema["additionalProperties"]
        if not isinstance(additional_properties, (bool, dict)):
            raise ValueError(f"{path}.additionalProperties must be a boolean or object")
        if isinstance(additional_properties, dict):
            _validate_json_schema_subset(additional_properties, path=f"{path}.additionalProperties")
    constraint_keys = _NUMERIC_SCHEMA_KEYS | _INTEGER_SCHEMA_KEYS | {"pattern"}
    if any(key in schema for key in ("enum", "const", "default")) and any(key in schema for key in constraint_keys):
        raise ValueError(f"{path} combines enum/const/default with constraints unsupported by deterministic brokered synthesis")
    if "multipleOf" in schema:
        raise ValueError(f"{path}.multipleOf is not supported for deterministic brokered tool schemas")
    for key in _NUMERIC_SCHEMA_KEYS:
        if key in schema:
            value = schema[key]
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{path}.{key} must be a number")
    for key in _INTEGER_SCHEMA_KEYS:
        if key in schema:
            value = schema[key]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{path}.{key} must be a non-negative integer")

def _reject_unsafe_brokered_property_name_values(value: Any, *, path: str) -> None:
    if isinstance(value, str):
        unsafe = _unsafe_brokered_key(value)
        if unsafe is not None:
            raise ValueError(f"{path} value {unsafe!r} is not safe for brokered tool schemas")
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            _reject_unsafe_brokered_property_name_values(item, path=f"{path}[{idx}]")
    elif isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str):
                unsafe = _unsafe_brokered_key(key)
                if unsafe is not None:
                    raise ValueError(f"{path}.{unsafe} is not safe for brokered tool schemas")
            _reject_unsafe_brokered_property_name_values(child, path=f"{path}.{key}")


class Metadata(_Strict):
    name: str = Field(min_length=1)


class AuthSpec(_Strict):
    type: str = Field(min_length=1)
    token_env: str | None = Field(default=None, alias="tokenEnv")
    audience: str | None = None

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @field_validator("token_env")
    @classmethod
    def _token_env_name(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return _validate_env_name(value, field="auth.tokenEnv")

    @model_validator(mode="after")
    def _valid_shape(self) -> "AuthSpec":
        if self.type == _AUTH_BEARER:
            if not self.token_env:
                raise ValueError("bearer auth requires tokenEnv")
            if self.audience:
                raise ValueError("bearer auth must not set audience")
            return self
        if self.type == _AUTH_WORKLOAD_IDENTITY:
            if not self.audience:
                raise ValueError("workload identity auth requires audience")
            if self.token_env:
                raise ValueError("workload identity auth must not set tokenEnv")
            return self
        raise ValueError(f"unsupported auth type {self.type!r}")


class ModelSpec(_Strict):
    """The hosted, OpenAI-compatible model the agent talks to."""

    provider: str = Field(min_length=1)
    base_url: str = Field(alias="baseURL", min_length=1)
    name: str = Field(min_length=1)
    # NAME of the env var holding the API key (never the value). agent-abi.md §2.
    api_key_env: str | None = Field(default=None, alias="apiKeyEnv")
    auth: AuthSpec | None = None

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @field_validator("provider")
    @classmethod
    def _provider_supported(cls, value: str) -> str:
        if value != _PROVIDER_OPENAI_COMPATIBLE:
            raise ValueError(
                f"unsupported provider {value!r}; expected {_PROVIDER_OPENAI_COMPATIBLE!r}"
            )
        return value

    @field_validator("api_key_env")
    @classmethod
    def _api_key_env_name(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return _validate_env_name(value, field="model.apiKeyEnv")

    @model_validator(mode="after")
    def _model_auth_supported(self) -> "ModelSpec":
        if self.auth is not None and self.auth.type != _AUTH_WORKLOAD_IDENTITY:
            raise ValueError("model.auth supports only workload-identity-token")
        return self


class ToolHeaderSpec(_Strict):
    name: str = Field(min_length=1)
    value: str | None = None
    value_env: str | None = Field(default=None, alias="valueEnv")

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @field_validator("name")
    @classmethod
    def _header_name(cls, value: str) -> str:
        if not _HEADER_NAME_RE.fullmatch(value):
            raise ValueError("header name is invalid")
        return value

    @field_validator("value_env")
    @classmethod
    def _value_env_name(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return _validate_env_name(value, field="tools[].headers[].valueEnv")

    @model_validator(mode="after")
    def _exactly_one_value(self) -> "ToolHeaderSpec":
        if (self.value is None or self.value == "") == (self.value_env is None or self.value_env == ""):
            raise ValueError("headers must set exactly one of value or valueEnv")
        if self.name.lower() in _CREDENTIAL_HEADER_NAMES and self.value:
            raise ValueError("static credential headers are not allowed; use valueEnv or auth")
        if self.value and _looks_like_secret_literal(self.value):
            raise ValueError("header value looks like a secret; use valueEnv")
        return self


class ToolSpec(_Strict):
    """An MCP server: stdio command or Streamable HTTP remote transport."""

    name: str = Field(min_length=1)
    type: str | None = None
    transport: str | None = None
    command: list[str] = Field(default_factory=list)
    url_env: str | None = Field(default=None, alias="urlEnv")
    headers: list[ToolHeaderSpec] = Field(default_factory=list)
    auth: AuthSpec | None = None
    approval: str | None = None
    # NAMES only — serve passes ONLY these into stdio subprocess env (secret-bleed rule).
    env: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @field_validator("command")
    @classmethod
    def _command_non_empty_parts(cls, value: list[str]) -> list[str]:
        if any(part == "" for part in value):
            raise ValueError("command entries must be non-empty strings")
        return value

    @field_validator("url_env")
    @classmethod
    def _url_env_name(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return _validate_env_name(value, field="tools[].urlEnv")

    @field_validator("env")
    @classmethod
    def _env_names(cls, value: list[str]) -> list[str]:
        return [_validate_env_name(name, field="tools[].env") for name in value]

    @field_validator("approval")
    @classmethod
    def _approval_supported(cls, value: str | None) -> str | None:
        if value is not None and value not in _APPROVAL_VALUES:
            raise ValueError("approval must be one of never, auto, or always")
        if value in {"auto", "always"}:
            raise ValueError("tool approval policies are not supported by this runtime")
        return value

    @model_validator(mode="after")
    def _valid_tool_shape(self) -> "ToolSpec":
        if self.type is not None and self.type != _TOOL_TYPE_MCP:
            raise ValueError("tool type must be 'mcp'")
        variants = int(bool(self.command)) + int(bool(self.url_env))
        if variants != 1:
            raise ValueError("tool must set exactly one of command or urlEnv")
        if self.command:
            if self.transport not in (None, "", _TRANSPORT_STDIO):
                raise ValueError("command tools must omit transport or set stdio")
            if self.headers or self.auth or self.url_env:
                raise ValueError("command tools must not set urlEnv, headers, or auth")
        if self.url_env:
            if self.type != _TOOL_TYPE_MCP:
                raise ValueError("remote MCP tools must set type: mcp")
            if self.transport != _TRANSPORT_STREAMABLE_HTTP:
                raise ValueError("remote MCP tools must set transport: streamable-http")
            if self.env:
                raise ValueError("remote MCP tools must use headers/auth instead of env")
            if self.auth is not None and any(h.name.lower() == "authorization" for h in self.headers):
                raise ValueError("remote MCP tools must not set Authorization header when auth is configured")
        return self


class EnvVarSpec(_Strict):
    """A runtime environment variable requirement by NAME only."""

    name: str = Field(min_length=1)
    required: bool = False

    @field_validator("name")
    @classmethod
    def _name_is_env_var(cls, value: str) -> str:
        return _validate_env_name(value, field="env[].name")


class ContextProviderSpec(_Strict):
    name: str | None = None
    type: str = Field(min_length=1)
    source: str | None = None
    path: str | None = None
    tool_ref: str | None = Field(default=None, alias="toolRef")
    index: str | None = None
    endpoint_env: str | None = Field(default=None, alias="endpointEnv")
    index_env: str | None = Field(default=None, alias="indexEnv")
    store_name_env: str | None = Field(default=None, alias="storeNameEnv")
    auth: AuthSpec | None = None

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @field_validator("endpoint_env", "index_env", "store_name_env")
    @classmethod
    def _env_names(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return _validate_env_name(value, field="context.providers[].env")

    @model_validator(mode="after")
    def _valid_context_provider_shape(self) -> "ContextProviderSpec":
        if self.auth is not None and self.auth.type == _AUTH_BEARER:
            raise ValueError("context providers do not support bearer auth; use workload-identity-token")
        if self.type == _CONTEXT_TYPE_SEARCH:
            if not self.endpoint_env:
                raise ValueError("search context providers require endpointEnv")
            if not self.index_env:
                raise ValueError("search context providers require indexEnv")
            return self
        if self.type == _CONTEXT_TYPE_MEMORY:
            if not self.endpoint_env:
                raise ValueError("memory context providers require endpointEnv")
            if not self.store_name_env:
                raise ValueError("memory context providers require storeNameEnv")
            return self
        if self.type == _CONTEXT_TYPE_SKILLS:
            if self.auth is not None:
                raise ValueError("skills context providers must not set auth; configure auth on the referenced MCP tool")
            if self.source == _CONTEXT_SOURCE_FILESYSTEM:
                if not self.path:
                    raise ValueError("filesystem skills require path")
                normalized = posixpath.normpath(self.path)
                if normalized != "/agent/skills" and not normalized.startswith("/agent/skills/"):
                    raise ValueError("filesystem skills path must be an absolute path under /agent/skills")
            elif self.source == _CONTEXT_SOURCE_MCP:
                if not self.tool_ref:
                    raise ValueError("MCP skills require toolRef")
            else:
                raise ValueError("skills context source must be filesystem or mcp")
            return self
        raise ValueError("context provider type must be search, skills, or memory")


class ContextSpec(_Strict):
    providers: list[ContextProviderSpec] = Field(default_factory=list)


def _schema_types(schema: Mapping[str, Any]) -> list[str]:
    if "type" not in schema:
        return []
    raw = schema.get("type")
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list) and all(isinstance(item, str) for item in raw):
        return list(raw)
    raise ValueError("brokered tool JSON Schema type must be a string or string array")


def _value_matches_schema_type(value: Any, schema_type: str) -> bool:
    if schema_type == "null":
        return value is None
    if schema_type == "boolean":
        return isinstance(value, bool)
    if schema_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if schema_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "array":
        return isinstance(value, list)
    if schema_type == "object":
        return isinstance(value, Mapping)
    raise ValueError(f"unsupported brokered tool JSON Schema type {schema_type!r}")


def _validate_schema_value_constraints(value: Any, *, path: str) -> None:
    if not isinstance(value, Mapping):
        return
    types = _schema_types(value)
    if types:
        for keyword in ("const", "default"):
            if keyword in value and not any(_value_matches_schema_type(value[keyword], schema_type) for schema_type in types):
                raise ValueError(f"{path}.{keyword} must match the declared JSON Schema type")
        enum = value.get("enum")
        if enum is not None:
            if not isinstance(enum, list):
                raise ValueError(f"{path}.enum must be an array")
            for idx, item in enumerate(enum):
                if not any(_value_matches_schema_type(item, schema_type) for schema_type in types):
                    raise ValueError(f"{path}.enum[{idx}] must match the declared JSON Schema type")
    properties = value.get("properties")
    if isinstance(properties, Mapping):
        for name, child in properties.items():
            if isinstance(child, Mapping):
                _validate_schema_value_constraints(child, path=f"{path}.properties.{name}")
    items = value.get("items")
    if isinstance(items, Mapping):
        _validate_schema_value_constraints(items, path=f"{path}.items")
    elif isinstance(items, list):
        for idx, child in enumerate(items):
            if isinstance(child, Mapping):
                _validate_schema_value_constraints(child, path=f"{path}.items[{idx}]")
    additional = value.get("additionalProperties")
    if isinstance(additional, Mapping):
        _validate_schema_value_constraints(additional, path=f"{path}.additionalProperties")


class BrokeredToolSpec(_Strict):
    """Static, safe Orka-brokered tool schema exposed to hosted Foundry models."""

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    brokered_class: Literal["read", "write", "coordination"] = Field(alias="brokeredClass")
    parameters: dict[str, Any]
    schema_digest: str | None = Field(default=None, alias="schemaDigest")

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _reject_unsafe_top_level_fields(cls, data: Any) -> Any:
        if isinstance(data, Mapping):
            for key in data:
                if isinstance(key, str) and _unsafe_brokered_key(key) is not None and key not in {"schemaDigest"}:
                    raise ValueError(f"brokered tool field {key!r} is unsafe")
        return data

    @field_validator("description")
    @classmethod
    def _safe_description(cls, value: str) -> str:
        if _unsafe_brokered_text(value):
            raise ValueError("brokered tool description must not contain URLs or secret-like material")
        return value

    @field_validator("name")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        if not _BROKERED_TOOL_NAME_RE.fullmatch(value):
            raise ValueError("brokered tool name must match [A-Za-z0-9_-]{1,64}")
        return value

    @field_validator("parameters")
    @classmethod
    def _valid_json_schema(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError("brokered tool parameters must be a JSON Schema object")
        try:
            encoded = _canonical_json(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("brokered tool parameters must be JSON serializable") from exc
        if len(encoded.encode("utf-8")) > _MAX_BROKERED_SCHEMA_BYTES:
            raise ValueError("brokered tool parameters schema is too large")
        cloned = json.loads(encoded)
        if cloned.get("type") != "object":
            raise ValueError("brokered tool parameters schema must set type: object")
        _validate_json_schema_subset(cloned, path="brokeredTools[].parameters")
        _reject_unsafe_brokered_schema_keys(cloned, path="brokeredTools[].parameters")
        _validate_schema_value_constraints(cloned, path="brokeredTools[].parameters")
        return cloned

    @field_validator("schema_digest")
    @classmethod
    def _valid_schema_digest(cls, value: str | None) -> str | None:
        if value is None:
            return value
        normalized = value.lower()
        if value != normalized or not _BROKERED_SCHEMA_DIGEST_RE.fullmatch(value):
            raise ValueError("brokered tool schemaDigest must be sha256:<64 lowercase hex>")
        return value

    @model_validator(mode="after")
    def _schema_digest_matches(self) -> "BrokeredToolSpec":
        if self.schema_digest is not None:
            actual = brokered_tool_schema_digest(
                name=self.name,
                description=self.description,
                brokered_class=self.brokered_class,
                parameters=self.parameters,
            )
            if self.schema_digest != actual:
                raise ValueError("brokered tool schemaDigest does not match the safe schema")
        return self


class ObservabilityOTelSpec(_Strict):
    endpoint_env: str | None = Field(default=None, alias="endpointEnv")

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @field_validator("endpoint_env")
    @classmethod
    def _endpoint_env_name(cls, value: str | None) -> str | None:
        if value is None:
            return value
        _validate_env_name(value, field="observability.otel.endpointEnv")
        raise ValueError("observability.otel.endpointEnv is not supported by current runtimes")


class ObservabilityLogsSpec(_Strict):
    level_env: str | None = Field(default=None, alias="levelEnv")

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @field_validator("level_env")
    @classmethod
    def _level_env_name(cls, value: str | None) -> str | None:
        if value is None:
            return value
        _validate_env_name(value, field="observability.logs.levelEnv")
        raise ValueError("observability.logs.levelEnv is not supported by current runtimes")


class ObservabilitySpec(_Strict):
    otel: ObservabilityOTelSpec = Field(default_factory=ObservabilityOTelSpec)
    logs: ObservabilityLogsSpec = Field(default_factory=ObservabilityLogsSpec)


class ExposeSpec(_Strict):
    openai: bool
    port: int

    @field_validator("openai")
    @classmethod
    def _openai_required(cls, value: bool) -> bool:
        if not value:
            raise ValueError("expose.openai must be true in v0")
        return value

    @field_validator("port")
    @classmethod
    def _port_range(cls, value: int) -> int:
        if value < 1 or value > 65535:
            raise ValueError("expose.port must be between 1 and 65535")
        return value


class AgentSpec(_Strict):
    """The whole baked ``agent.yaml``."""

    abi_version: str = Field(alias="abiVersion")
    metadata: Metadata
    model: ModelSpec
    # Fully-resolved system prompt scalar (writer resolves inline|file -> string).
    instructions: str
    tools: list[ToolSpec] = Field(default_factory=list)
    brokered_tools: list[BrokeredToolSpec] = Field(default_factory=list, alias="brokeredTools")
    env: list[EnvVarSpec] = Field(default_factory=list)
    context: ContextSpec = Field(default_factory=ContextSpec)
    observability: ObservabilitySpec = Field(default_factory=ObservabilitySpec)
    expose: ExposeSpec

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @field_validator("env")
    @classmethod
    def _unique_env_names(cls, value: list[EnvVarSpec]) -> list[EnvVarSpec]:
        seen: set[str] = set()
        duplicates: list[str] = []
        for entry in value:
            if entry.name in seen:
                duplicates.append(entry.name)
            seen.add(entry.name)
        if duplicates:
            names = ", ".join(sorted(set(duplicates)))
            raise ValueError(f"duplicate env var declarations: {names}")
        return value

    @field_validator("brokered_tools")
    @classmethod
    def _unique_brokered_tool_names(cls, value: list[BrokeredToolSpec]) -> list[BrokeredToolSpec]:
        seen: set[str] = set()
        duplicates: list[str] = []
        for entry in value:
            if entry.name in seen:
                duplicates.append(entry.name)
            seen.add(entry.name)
        if duplicates:
            names = ", ".join(sorted(set(duplicates)))
            raise ValueError(f"duplicate brokered tool declarations: {names}")
        return value

    @model_validator(mode="after")
    def _valid_context_tool_refs(self) -> "AgentSpec":
        tools = {tool.name: tool for tool in self.tools}
        owned_tool_names = set(tools)
        brokered_names = {tool.name for tool in self.brokered_tools}
        if owned_tool_names and brokered_names:
            raise ValueError("tools and brokeredTools cannot be mixed in v0; direct AgentKit-owned tools are disabled in brokered Foundry mode")
        overlap = owned_tool_names & brokered_names
        if overlap:
            raise ValueError(f"tool names cannot be both owned and brokered: {', '.join(sorted(overlap))}")
        for provider in self.context.providers:
            if provider.type == _CONTEXT_TYPE_SKILLS and provider.source == _CONTEXT_SOURCE_MCP:
                if not provider.tool_ref:
                    raise ValueError("MCP skills require toolRef")
                tool = tools.get(provider.tool_ref)
                if tool is None:
                    raise ValueError(f"MCP skills toolRef {provider.tool_ref!r} references unknown tool")
                if not tool.url_env or tool.transport != _TRANSPORT_STREAMABLE_HTTP:
                    raise ValueError(f"MCP skills toolRef {provider.tool_ref!r} must reference a streamable-http MCP tool")
        return self


class ConfigError(Exception):
    """Raised when ``agent.yaml`` is missing, unparseable, or invalid."""


def load(path: str | Path) -> AgentSpec:
    """Read and validate ``agent.yaml`` at ``path``.

    Raises :class:`ConfigError` with a clear, single-line message on any problem
    (missing file, non-mapping YAML, schema violation, or unsupported abiVersion).
    """
    p = Path(path)
    try:
        raw = p.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigError(f"agent config not found: {p}") from exc
    except OSError as exc:
        raise ConfigError(f"cannot read agent config {p}: {exc}") from exc

    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ConfigError(f"agent config {p} is not valid YAML: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError(f"agent config {p} must be a YAML mapping, got {type(data).__name__}")

    try:
        spec = AgentSpec.model_validate(data)
    except ValidationError as exc:
        # Compact every pydantic error onto its own line: "<loc>: <msg>".
        lines = [
            f"  {'.'.join(str(x) for x in err['loc']) or '<root>'}: {err['msg']}"
            for err in exc.errors()
        ]
        raise ConfigError(
            f"agent config {p} is invalid:\n" + "\n".join(lines)
        ) from exc

    if spec.abi_version != ABI_VERSION:
        raise ConfigError(
            f"agent config {p}: unsupported abiVersion {spec.abi_version!r} "
            f"(this build of agentkit-serve understands {ABI_VERSION!r})"
        )

    return spec


def validate_required_env(spec: AgentSpec) -> None:
    """Fail if required runtime env vars declared in the ABI are missing.

    Only env var NAMES are reported; values are never inspected beyond empty vs
    present and are never logged.
    """
    missing = [
        entry.name
        for entry in spec.env
        if entry.required and not os.environ.get(entry.name)
    ]
    if not missing:
        return
    names = ", ".join(missing)
    first = missing[0]
    plural = "s" if len(missing) != 1 else ""
    raise ConfigError(
        f"required env var{plural} not set: {names}; inject at runtime, e.g. "
        f"`docker run -e {first}=...`"
    )


def load_or_exit(path: str | Path) -> AgentSpec:
    """Like :func:`load` but validates runtime env and exits non-zero on error."""
    try:
        spec = load(path)
        validate_required_env(spec)
        return spec
    except ConfigError as exc:
        print(f"agentkit-serve: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
