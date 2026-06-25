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
shared server imports nothing from ``pydantic_ai``. Cross-runtime invariants such
as API-key resolution, secret-safe tool env projection, MCP timeout parsing, and
error normalization live in ``agentkit_serve_common.adapter_support``.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai import Agent

try:  # pydantic-ai 1.x
    from pydantic_ai.mcp import MCPServerStdio
except ImportError:  # pydantic-ai 2.x
    MCPServerStdio = None  # type: ignore[assignment]
    from pydantic_ai.mcp import MCPToolset, StdioTransport
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from agentkit_serve_common.adapter_support import (
    FORWARDED_ROLES,
    AgentBuildError,
    declared_tool_env,
    normalize_agent_run_error,
    positive_float_env,
    resolve_api_key,
    split_tool_command,
    upstream_status_code,
)
from agentkit_serve_common.config import AgentSpec, ToolSpec
from agentkit_serve_common.runtime import RunResult

# Seconds to wait for a stdio MCP server's initialize handshake. pydantic-ai's
# default is 5s, which is too tight for a COLD `uvx`/`npx` tool: the first launch
# resolves, downloads, and installs the server package before it speaks MCP, which
# routinely exceeds 5s. Default generously and let operators tune via env.
_DEFAULT_MCP_INIT_TIMEOUT = 120.0


def _mcp_init_timeout() -> float:
    """MCP stdio init timeout (seconds), overridable via AGENTKIT_MCP_TIMEOUT."""
    return positive_float_env(default=_DEFAULT_MCP_INIT_TIMEOUT)


def _resolve_api_key(spec: AgentSpec) -> str:
    """Compatibility wrapper around the shared adapter support Module."""
    return resolve_api_key(spec)


def build_model(spec: AgentSpec) -> OpenAIChatModel:
    """Construct the OpenAI-compatible chat model pointed at ``model.baseURL``."""
    provider = OpenAIProvider(
        base_url=spec.model.base_url,
        api_key=resolve_api_key(spec),
    )
    return OpenAIChatModel(spec.model.name, provider=provider)


def _tool_env(tool: ToolSpec) -> dict[str, str]:
    """Compatibility wrapper around the shared secret-safe tool env projection."""
    return declared_tool_env(tool)


def build_tool_server(tool: ToolSpec) -> Any:
    """Create a stdio MCP toolset for one tool spec."""
    command, args = split_tool_command(tool, example='["npx", "-y", "..."]')

    # Generous init timeout: a cold uvx/npx tool installs its package before
    # speaking MCP, which exceeds pydantic-ai's 5s default (see above).
    timeout = _mcp_init_timeout()
    env = declared_tool_env(tool)

    if MCPServerStdio is not None:
        return MCPServerStdio(
            command=command,
            args=args,
            env=env,
            timeout=timeout,
            # tool_prefix namespaces tool names so two servers can't collide.
            tool_prefix=tool.name,
        )

    # pydantic-ai 2.x replaced MCPServerStdio with a transport + toolset pair.
    # Prefixing moved to Toolset.prefixed(...), preserving the same namespacing
    # interface as the 1.x tool_prefix argument.
    transport = StdioTransport(
        command=command,
        args=args,
        env=env,
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
        if not text or role not in FORWARDED_ROLES:
            continue
        if role == "user":
            out.append(ModelRequest(parts=[UserPromptPart(content=text)]))
        elif role == "system":
            out.append(ModelRequest(parts=[SystemPromptPart(content=text)]))
        elif role == "assistant":
            out.append(ModelResponse(parts=[TextPart(content=text)]))
    return out


def _status_of(exc: Exception) -> int:
    """Compatibility wrapper around shared upstream status extraction."""
    return upstream_status_code(exc)


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


async def run_agent(
    agent: Agent,
    prompt: str,
    history: list[tuple[str, str]] | None = None,
) -> RunResult:
    """Run the pydantic-ai agent and return the neutral result shape."""
    message_history = _to_message_history(history)
    try:
        result = await agent.run(prompt, message_history=message_history)
    except Exception as exc:  # noqa: BLE001 — normalized for the façade
        raise normalize_agent_run_error(exc) from exc
    return RunResult(text=_result_text(result), usage=_result_usage(result))
