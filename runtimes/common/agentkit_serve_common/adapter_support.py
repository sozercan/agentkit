"""Framework-neutral helpers shared by runtime adapter modules.

The adapter modules should spend their complexity budget on translating the
frozen ``agent.yaml`` ABI into their framework's concrete agent/client/tool
objects. Cross-runtime invariants live here instead: model API-key resolution,
secret-safe tool env projection, MCP timeout parsing, command validation, and
normalizing framework/model failures into the common run error.
"""

from __future__ import annotations

import os
import re

from .config import AgentSpec, ToolSpec
from .conversation import FORWARDED_ROLES
from .runtime import AgentRunError

# Placeholder API key for OpenAI-compatible endpoints that need no auth (many
# local servers reject an EMPTY string but accept any non-empty token). Used only
# when the spec declares no apiKeyEnv. Never a real secret.
NO_AUTH_API_KEY = "not-needed"


MCP_TIMEOUT_ENV = "AGENTKIT_MCP_TIMEOUT"
_BRACED_ENV_REF_RE = re.compile(r"\$\{([^}]+)\}")


class AgentBuildError(Exception):
    """Raised when an adapter cannot construct its concrete agent."""


def resolve_api_key(spec: AgentSpec) -> str:
    """Resolve the model API key from the env var NAMED in the spec.

    Per the ABI, ``model.apiKeyEnv`` is the NAME of an env var; the value is
    injected at runtime (``docker run -e``). If the name is declared but the var
    is absent, fail fast with a clear, SECRET-FREE message.
    """
    name = spec.model.api_key_env
    if not name:
        # No auth declared — e.g. a co-located local model over baseURL.
        return NO_AUTH_API_KEY
    value = os.environ.get(name)
    if value is None or value == "":
        raise AgentBuildError(
            f"model.apiKeyEnv {name!r} is declared in agent.yaml but env var "
            f"{name!r} is not set; inject it at runtime, e.g. "
            f"`docker run -e {name}=...`"
        )
    return value


def declared_tool_env(tool: ToolSpec) -> dict[str, str]:
    """Select ONLY the env vars NAMED in ``tool.env``.

    The MCP subprocess must never inherit the full container environment — that
    would bleed the model API key (and every other secret) into every tool. We
    pass through exactly the declared names that are actually present.

    Some MCP stdio transports expand braced references like ``${VAR}`` inside env
    values against the parent process environment. Reject references to undeclared
    vars so a declared tool env such as ``TOOL_CONFIG=${OPENAI_API_KEY}`` cannot
    smuggle the model key into a subprocess unless that key was explicitly listed
    in the tool's own ``env`` allowlist.
    """
    allowed = set(tool.env)
    out = {name: os.environ[name] for name in tool.env if name in os.environ}
    for name, value in out.items():
        undeclared = sorted(ref for ref in _BRACED_ENV_REF_RE.findall(value) if ref not in allowed)
        if undeclared:
            raise AgentBuildError(
                f"tool {tool.name!r} env var {name!r} references undeclared env var(s) "
                f"{', '.join(undeclared)}; list every referenced env var in that tool's "
                "env allowlist or remove the ${...} reference"
            )
    return out


def split_tool_command(tool: ToolSpec, *, example: str) -> tuple[str, list[str]]:
    """Return ``(command, args)`` for a stdio MCP tool, or fail clearly."""
    if not tool.command:
        raise AgentBuildError(
            f"tool {tool.name!r} has an empty command; a stdio MCP server needs "
            f"at least the executable, e.g. command: {example}"
        )
    return tool.command[0], list(tool.command[1:])


def positive_float_env(name: str = MCP_TIMEOUT_ENV, *, default: float) -> float:
    """Read a positive float env var, falling back to ``default``."""
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        val = float(raw)
    except ValueError:
        return default
    return val if val > 0 else default


def positive_int_env(name: str = MCP_TIMEOUT_ENV, *, default: int | None) -> int | None:
    """Read a positive integer env var, accepting float strings, else ``default``."""
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        val = int(float(raw))
    except ValueError:
        return default
    return val if val > 0 else default


def upstream_status_code(exc: BaseException, *, default: int = 502) -> int:
    """Best-effort upstream HTTP status from a framework/model exception.

    Duck-typed on ``status_code`` (the OpenAI SDK's ``APIStatusError`` exposes it)
    so adapters can map real upstream 4xx/5xx responses through without importing
    SDK-specific exception classes. Some frameworks wrap the SDK error in their
    own exception and store the original as ``inner_exception`` or as the normal
    exception cause/context, so this walks the bounded exception chain.
    """
    seen: set[int] = set()
    cur: BaseException | None = exc
    for _ in range(10):  # bounded walk; guards against pathological cycles
        if cur is None or id(cur) in seen:
            break
        seen.add(id(cur))
        code = getattr(cur, "status_code", None)
        if isinstance(code, int) and 400 <= code <= 599:
            return code
        cur = getattr(cur, "inner_exception", None) or cur.__cause__ or cur.__context__
    return default


def normalize_agent_run_error(exc: Exception) -> AgentRunError:
    """Convert an adapter/framework exception into the common façade error."""
    return AgentRunError(
        f"agent run failed: {exc}",
        status=upstream_status_code(exc),
        code=exc.__class__.__name__,
    )
