"""Typed loader for the frozen ``/agent/agent.yaml`` ABI.

This mirrors ``docs/agent-abi.md`` EXACTLY — it is the Python (reader) half of the
contract whose Go (writer) half lives in ``pkg/agentkit/abi``. The shape here is
the *baked* file: ``instructions`` is already a fully-resolved scalar string and
``model.provider`` is always ``openai-compatible`` in v0.
"""

from __future__ import annotations

import os
import posixpath
import re
import sys
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

# The ABI schema version this reader understands (agent-abi.md: ``abiVersion: v0``).
ABI_VERSION = "v0"
_PROVIDER_OPENAI_COMPATIBLE = "openai-compatible"
_ENV_NAME_RE = re.compile(r"^[A-Z0-9_]+$")
_HEADER_NAME_RE = re.compile(r"^[A-Za-z0-9!#$%&'*+\-.^_`|~]+$")
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
        if self.type == _CONTEXT_TYPE_SKILLS and self.source == _CONTEXT_SOURCE_FILESYSTEM:
            if not self.path:
                raise ValueError("filesystem skills require path")
            normalized = posixpath.normpath(self.path)
            if normalized != "/agent/skills" and not normalized.startswith("/agent/skills/"):
                raise ValueError("filesystem skills path must be an absolute path under /agent/skills")
        return self


class ContextSpec(_Strict):
    providers: list[ContextProviderSpec] = Field(default_factory=list)


class ObservabilityOTelSpec(_Strict):
    endpoint_env: str | None = Field(default=None, alias="endpointEnv")

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @field_validator("endpoint_env")
    @classmethod
    def _endpoint_env_name(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return _validate_env_name(value, field="observability.otel.endpointEnv")


class ObservabilityLogsSpec(_Strict):
    level_env: str | None = Field(default=None, alias="levelEnv")

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @field_validator("level_env")
    @classmethod
    def _level_env_name(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return _validate_env_name(value, field="observability.logs.levelEnv")


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

    @model_validator(mode="after")
    def _valid_context_tool_refs(self) -> "AgentSpec":
        tools = {tool.name: tool for tool in self.tools}
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
