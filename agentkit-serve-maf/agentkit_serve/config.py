"""Typed loader for the frozen ``/agent/agent.yaml`` ABI.

This mirrors ``docs/agent-abi.md`` EXACTLY — it is the Python (reader) half of the
contract whose Go (writer) half lives in ``pkg/agentkit2llb``. The shape here is
the *baked* file: ``instructions`` is already a fully-resolved scalar string and
``model.provider`` is always ``openai-compatible`` in v0. Do not add fields that
the writer does not emit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

# The ABI schema version this reader understands (agent-abi.md: ``abiVersion: v0``).
ABI_VERSION = "v0"


class _Strict(BaseModel):
    """Base model that forbids unknown keys so a writer/reader drift surfaces loudly."""

    model_config = ConfigDict(extra="forbid")


class Metadata(_Strict):
    name: str


class ModelSpec(_Strict):
    """The hosted, OpenAI-compatible model the agent talks to."""

    provider: str
    base_url: str = Field(alias="baseURL")
    name: str
    # NAME of the env var holding the API key (never the value). agent-abi.md §2.
    api_key_env: str | None = Field(default=None, alias="apiKeyEnv")

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class ToolSpec(_Strict):
    """A stdio MCP server (v0's only tool transport)."""

    name: str
    command: list[str] = Field(default_factory=list)
    # NAMES only — serve passes ONLY these into the subprocess env (secret-bleed rule).
    env: list[str] = Field(default_factory=list)


class ExposeSpec(_Strict):
    openai: bool = False
    port: int = 8080


class AgentSpec(_Strict):
    """The whole baked ``agent.yaml``."""

    abi_version: str = Field(alias="abiVersion")
    metadata: Metadata
    model: ModelSpec
    # Fully-resolved system prompt scalar (writer resolves inline|file -> string).
    instructions: str
    tools: list[ToolSpec] = Field(default_factory=list)
    expose: ExposeSpec = Field(default_factory=ExposeSpec)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


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


def load_or_exit(path: str | Path) -> AgentSpec:
    """Like :func:`load` but prints the error to stderr and exits non-zero."""
    try:
        return load(path)
    except ConfigError as exc:
        print(f"agentkit-serve: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
