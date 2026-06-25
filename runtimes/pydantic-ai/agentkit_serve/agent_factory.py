"""Build a pydantic-ai :class:`Agent` from a validated :class:`AgentSpec`.

Verified against modern pydantic-ai (1.107.x through 2.x). Key facts baked in here:

* The OpenAI-compatible model class is ``OpenAIChatModel`` (``OpenAIModel`` is a
  deprecated alias); its base_url/api_key come from an ``OpenAIProvider``.
* stdio MCP servers are passed to the agent as ``toolsets`` (the old
  ``mcp_servers=`` kwarg is gone). pydantic-ai 1.x exposes ``MCPServerStdio``;
  pydantic-ai 2.x uses ``MCPToolset(StdioTransport(...))``.
* The agent is itself an async context manager: ``async with agent:`` starts the
  MCP subprocesses; that is the modern replacement for ``run_mcp_servers()``.

The agent OWNS its system prompt (spec.instructions) and its tools — request-side
tools are rejected by the server, never merged here.

This module is the ONLY framework-specific surface of the adapter. It exposes a
NEUTRAL run contract that ``agentkit_serve_common.server`` consumes —
``build_agent`` + ``run_agent`` (the :class:`RuntimeFactory` protocol) — so the
shared server imports nothing from ``pydantic_ai``. That neutral seam is identical
across runtimes (it matches the MAF adapter's), which is what lets the ABI loader,
the ``/v1`` facade, and the CLI live in ``agentkit_serve_common`` (packaging
option B).
"""

from __future__ import annotations

import os
from typing import Any

from pydantic_ai import Agent

try:  # pydantic-ai 1.x
    from pydantic_ai.mcp import MCPServerStdio
except ImportError:  # pydantic-ai 2.x
    MCPServerStdio = None  # type: ignore[assignment]
    from pydantic_ai.mcp import MCPToolset, StdioTransport
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from agentkit_serve_common.config import AgentSpec, ToolSpec
from agentkit_serve_common.runtime import AgentRunError, RunResult

# pydantic-ai surfaces upstream HTTP failures as ModelHTTPError (has .status_code).
try:  # pragma: no cover - import shape guard across pydantic-ai versions
    from pydantic_ai.exceptions import ModelHTTPError
except Exception:  # pragma: no cover
    ModelHTTPError = None  # type: ignore[assignment]

# Placeholder API key for OpenAI-compatible endpoints that need no auth (many
# local servers reject an EMPTY string but accept any non-empty token). Used only
# when the spec declares no apiKeyEnv. Never a real secret.
_NO_AUTH_PLACEHOLDER = "not-needed"

# Roles forwarded from the request history into the pydantic-ai conversation.
_FORWARDED_ROLES = frozenset({"system", "user", "assistant"})

# Seconds to wait for a stdio MCP server's initialize handshake. pydantic-ai's
# default is 5s, which is too tight for a COLD `uvx`/`npx` tool: the first launch
# resolves, downloads, and installs the server package before it speaks MCP, which
# routinely exceeds 5s. Default generously and let operators tune via env.
_DEFAULT_MCP_INIT_TIMEOUT = 120.0


def _mcp_init_timeout() -> float:
    """MCP stdio init timeout (seconds), overridable via AGENTKIT_MCP_TIMEOUT."""
    raw = os.environ.get("AGENTKIT_MCP_TIMEOUT")
    if not raw:
        return _DEFAULT_MCP_INIT_TIMEOUT
    try:
        val = float(raw)
    except ValueError:
        return _DEFAULT_MCP_INIT_TIMEOUT
    return val if val > 0 else _DEFAULT_MCP_INIT_TIMEOUT


class AgentBuildError(Exception):
    """Raised when the agent cannot be constructed (e.g. missing API key env)."""


def _resolve_api_key(spec: AgentSpec) -> str:
    """Resolve the model API key from the env var NAMED in the spec.

    Per the ABI, ``model.apiKeyEnv`` is the NAME of an env var; the value is
    injected at runtime (``docker run -e``). If the name is declared but the var
    is absent, fail fast with a clear, SECRET-FREE message.
    """
    name = spec.model.api_key_env
    if not name:
        # No auth declared — e.g. a co-located local model over baseURL.
        return _NO_AUTH_PLACEHOLDER
    value = os.environ.get(name)
    if value is None or value == "":
        raise AgentBuildError(
            f"model.apiKeyEnv {name!r} is declared in agent.yaml but env var "
            f"{name!r} is not set; inject it at runtime, e.g. "
            f"`docker run -e {name}=...`"
        )
    return value


def build_model(spec: AgentSpec) -> OpenAIChatModel:
    """Construct the OpenAI-compatible chat model pointed at ``model.baseURL``."""
    provider = OpenAIProvider(
        base_url=spec.model.base_url,
        api_key=_resolve_api_key(spec),
    )
    return OpenAIChatModel(spec.model.name, provider=provider)


def _tool_env(tool: ToolSpec) -> dict[str, str]:
    """Select ONLY the env vars NAMED in ``tool.env`` (plan §10 secret-bleed rule).

    The MCP subprocess must never inherit the full container environment — that
    would bleed the model API key (and every other secret) into every tool. We
    pass through exactly the declared names that are actually present.
    """
    return {name: os.environ[name] for name in tool.env if name in os.environ}


def build_tool_server(tool: ToolSpec) -> Any:
    """Create a stdio MCP toolset for one tool spec."""
    if not tool.command:
        # config.AgentSpec allows an empty command shape; the writer never emits
        # one, but guard anyway so a hand-rolled agent.yaml fails clearly.
        raise AgentBuildError(
            f"tool {tool.name!r} has an empty command; a stdio MCP server needs "
            f"at least the executable, e.g. command: [\"npx\", \"-y\", \"...\"]"
        )

    # Generous init timeout: a cold uvx/npx tool installs its package before
    # speaking MCP, which exceeds pydantic-ai's 5s default (see above).
    timeout = _mcp_init_timeout()

    if MCPServerStdio is not None:
        return MCPServerStdio(
            command=tool.command[0],
            args=list(tool.command[1:]),
            env=_tool_env(tool),
            timeout=timeout,
            # tool_prefix namespaces tool names so two servers can't collide.
            tool_prefix=tool.name,
        )

    # pydantic-ai 2.x replaced MCPServerStdio with a transport + toolset pair.
    # Prefixing moved to Toolset.prefixed(...), preserving the same namespacing
    # interface as the 1.x tool_prefix argument.
    transport = StdioTransport(
        command=tool.command[0],
        args=list(tool.command[1:]),
        env=_tool_env(tool),
        # Match the agent lifespan: when pydantic-ai exits the toolset context,
        # the stdio subprocess should be torn down instead of kept alive.
        keep_alive=False,
    )
    return MCPToolset(transport, init_timeout=timeout).prefixed(tool.name)


def build_agent(spec: AgentSpec) -> Agent:
    """Assemble the pydantic-ai agent: model + system prompt + stdio MCP toolsets."""
    model = build_model(spec)
    toolsets = [build_tool_server(t) for t in spec.tools]
    return Agent(
        model,
        instructions=spec.instructions,
        toolsets=toolsets,
    )


def _to_message_history(history: list[tuple[str, str]] | None) -> list:
    """Map neutral ``(role, text)`` tuples to a pydantic-ai message_history list.

    The agent's own ``instructions`` are applied by pydantic-ai; this mirrors the
    server's previous inline ``_split_conversation`` mapping exactly.
    """
    # Imported lazily so config-only consumers don't pull the messages module.
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        SystemPromptPart,
        TextPart,
        UserPromptPart,
    )

    out: list = []
    for role, text in history or []:
        if not text or role not in _FORWARDED_ROLES:
            continue
        if role == "user":
            out.append(ModelRequest(parts=[UserPromptPart(content=text)]))
        elif role == "system":
            out.append(ModelRequest(parts=[SystemPromptPart(content=text)]))
        elif role == "assistant":
            out.append(ModelResponse(parts=[TextPart(content=text)]))
    return out


def _status_of(exc: Exception) -> int:
    """Best-effort upstream HTTP status from a model/SDK error (else 502).

    pydantic-ai surfaces upstream HTTP failures as ``ModelHTTPError`` with a
    ``.status_code``; map it through so a real upstream 4xx/5xx is preserved.
    """
    if ModelHTTPError is not None and isinstance(exc, ModelHTTPError):
        code = getattr(exc, "status_code", 502)
        return int(code) if code else 502
    return 502


def _result_text(result: object) -> str:
    """Extract the final assistant text from a pydantic-ai run result."""
    output = getattr(result, "output", None)
    return output if isinstance(output, str) else str(output)


def _result_usage(result: object) -> dict[str, int]:
    """Best-effort OpenAI usage block from the pydantic-ai run result (zeros if unknown)."""
    try:
        usage = result.usage
        # In current pydantic-ai ``usage`` is a property returning a RunUsage; in
        # older builds it was a method. Prefer the property value; only call it if
        # we got a bare callable WITHOUT the token attributes (the real method).
        if not hasattr(usage, "input_tokens") and callable(usage):
            usage = usage()
        prompt_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    except Exception:
        prompt_tokens = completion_tokens = 0
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


async def run_agent(agent: Agent, prompt: str, history: list[tuple[str, str]] | None = None) -> RunResult:
    """Framework-agnostic run helper so ``server.py`` stays identical across adapters.

    Accepts the neutral ``(history, prompt)`` the server produces, runs the
    pydantic-ai agent once (non-streaming), and returns a neutral :class:`RunResult`.
    Any framework/model error is normalized to :class:`AgentRunError` carrying an
    HTTP status, so the server never imports a framework or SDK exception type.
    """
    message_history = _to_message_history(history)
    try:
        result = await agent.run(prompt, message_history=message_history)
    except Exception as exc:  # noqa: BLE001 — normalized for the façade
        # Preserve the original framework exception's class name in error.code
        # (the pre-shared-core behavior); _status_of maps ModelHTTPError → status.
        raise AgentRunError(
            f"agent run failed: {exc}",
            status=_status_of(exc),
            code=exc.__class__.__name__,
        ) from exc
    return RunResult(text=_result_text(result), usage=_result_usage(result))

