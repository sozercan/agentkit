"""Typed loader for the frozen ``/agent/agent.yaml`` ABI.

This mirrors ``docs/agent-abi.md`` EXACTLY — it is the Python (reader) half of the
contract whose Go (writer) half lives in ``pkg/agentkit/abi``. The shape here is
the *baked* file: ``instructions`` is already a fully-resolved scalar string and
``model.provider`` is always ``openai-compatible`` in v0. Do not add fields that
the writer does not emit.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

# The ABI schema version this reader understands (agent-abi.md: ``abiVersion: v0``).
ABI_VERSION = "v0"
_PROVIDER_OPENAI_COMPATIBLE = "openai-compatible"
_ENV_NAME_RE = re.compile(r"^[A-Z0-9_]+$")
_SECRET_VALUE_PREFIXES = ("sk-", "sk_", "ghp_", "github_pat_", "xoxb-", "AKIA")


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


class Metadata(_Strict):
    name: str = Field(min_length=1)


class ModelSpec(_Strict):
    """The hosted, OpenAI-compatible model the agent talks to."""

    provider: str = Field(min_length=1)
    base_url: str = Field(alias="baseURL", min_length=1)
    name: str = Field(min_length=1)
    # NAME of the env var holding the API key (never the value). agent-abi.md §2.
    api_key_env: str | None = Field(default=None, alias="apiKeyEnv")

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


class ToolSpec(_Strict):
    """A stdio MCP server (v0's only tool transport)."""

    name: str = Field(min_length=1)
    command: list[str] = Field(min_length=1)
    # NAMES only — serve passes ONLY these into the subprocess env (secret-bleed rule).
    env: list[str] = Field(default_factory=list)

    @field_validator("command")
    @classmethod
    def _command_non_empty_parts(cls, value: list[str]) -> list[str]:
        if any(part == "" for part in value):
            raise ValueError("command entries must be non-empty strings")
        return value

    @field_validator("env")
    @classmethod
    def _env_names(cls, value: list[str]) -> list[str]:
        return [_validate_env_name(name, field="tools[].env") for name in value]


class EnvVarSpec(_Strict):
    """A runtime environment variable requirement by NAME only."""

    name: str = Field(min_length=1)
    required: bool = False

    @field_validator("name")
    @classmethod
    def _name_is_env_var(cls, value: str) -> str:
        return _validate_env_name(value, field="env[].name")


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
