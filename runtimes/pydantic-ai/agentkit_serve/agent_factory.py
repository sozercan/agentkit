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
``build_runtime`` (the :class:`RuntimeFactory` protocol) — so the
shared server imports nothing from ``pydantic_ai``. Cross-runtime invariants such
as API-key resolution, secret-safe tool env projection, MCP timeout parsing, and
error normalization live in ``agentkit_serve_common.adapter_support``.
"""

from __future__ import annotations

from types import TracebackType
from typing import Any

from pydantic_ai import Agent

try:  # pydantic-ai 1.x
    from pydantic_ai.mcp import MCPServerStdio
except ImportError:  # pydantic-ai 2.x
    MCPServerStdio = None  # type: ignore[assignment]

try:
    from pydantic_ai.mcp import MCPToolset
except ImportError:  # pragma: no cover - older pydantic-ai without MCPToolset
    MCPToolset = None  # type: ignore[assignment]

try:
    from fastmcp.client.transports.stdio import StdioTransport
    from fastmcp.client.transports.http import StreamableHttpTransport
except ImportError:  # pragma: no cover - older dependency set without FastMCP transports
    StdioTransport = None  # type: ignore[assignment]
    StreamableHttpTransport = None  # type: ignore[assignment]
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from agentkit_serve_common.adapter_support import (
    FORWARDED_ROLES,
    AgentBuildError,
    declared_tool_env,
    normalize_agent_run_error,
    positive_float_env,
    resolve_api_key,
    resolve_tool_url,
    same_origin_mcp_httpx_client_factory,
    split_tool_command,
)
from agentkit_serve_common.config import AgentSpec, ToolSpec
from agentkit_serve_common.conversation import RunRequest
from agentkit_serve_common.runtime import (
    OfflineEchoRuntimeFactory,
    RunResult,
    RuntimeSession,
    offline_orka_echo_enabled,
)

# Seconds to wait for a stdio MCP server's initialize handshake. pydantic-ai's
# default is 5s, which is too tight for a COLD `uvx`/`npx` tool: the first launch
# resolves, downloads, and installs the server package before it speaks MCP, which
# routinely exceeds 5s. Default generously and let operators tune via env.
_DEFAULT_MCP_INIT_TIMEOUT = 120.0


def _mcp_init_timeout() -> float:
    """MCP stdio init timeout (seconds), overridable via AGENTKIT_MCP_TIMEOUT."""
    return positive_float_env(default=_DEFAULT_MCP_INIT_TIMEOUT)


def validate_supported_spec(spec: AgentSpec) -> None:
    if spec.model.auth is not None:
        raise AgentBuildError("pydantic-ai runtime does not support model.auth; use apiKeyEnv")
    if spec.context.providers:
        raise AgentBuildError("pydantic-ai runtime does not support context providers")


def build_model(spec: AgentSpec) -> OpenAIChatModel:
    """Construct the OpenAI-compatible chat model pointed at ``model.baseURL``."""
    provider = OpenAIProvider(
        base_url=spec.model.base_url,
        api_key=resolve_api_key(spec),
    )
    return OpenAIChatModel(spec.model.name, provider=provider)


def build_tool_server(tool: ToolSpec) -> Any:
    """Create an MCP toolset for one stdio or Streamable HTTP tool spec."""
    timeout = _mcp_init_timeout()

    if tool.url_env:
        url = resolve_tool_url(tool)
        if MCPToolset is None or StreamableHttpTransport is None:
            raise AgentBuildError(
                f"tool {tool.name!r} requires streamable-http MCP, but this pydantic-ai "
                "build does not expose a Streamable HTTP MCP transport"
            )
        transport = StreamableHttpTransport(
            url,
            httpx_client_factory=same_origin_mcp_httpx_client_factory(tool, url, timeout=timeout),
        )
        return MCPToolset(transport, init_timeout=timeout, read_timeout=timeout).prefixed(tool.name)

    command, args = split_tool_command(tool, example='["npx", "-y", "..."]')

    # Generous init timeout: a cold uvx/npx tool installs its package before
    # speaking MCP, which exceeds pydantic-ai's 5s default (see above).
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

    if MCPToolset is None or StdioTransport is None:
        raise AgentBuildError("this pydantic-ai build does not expose an MCP stdio transport")

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


class PydanticRuntime:
    """RuntimeSession Adapter around a pydantic-ai Agent."""

    def __init__(self, agent: Agent) -> None:
        self.agent = agent

    async def __aenter__(self) -> RuntimeSession:
        await self.agent.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        return await self.agent.__aexit__(exc_type, exc, tb)

    async def run(self, request: RunRequest) -> RunResult:
        return await run_agent(self.agent, request)


def supports_brokered_read() -> bool:
    return offline_orka_echo_enabled()


def supports_brokered_write() -> bool:
    return offline_orka_echo_enabled()


def supports_brokered_coordination() -> bool:
    return offline_orka_echo_enabled()


def build_runtime(spec: AgentSpec) -> RuntimeSession:
    """Build the runtime session consumed by the shared server."""
    if offline_orka_echo_enabled():
        return OfflineEchoRuntimeFactory().build_runtime(spec)
    validate_supported_spec(spec)
    return PydanticRuntime(build_agent(spec))


def _to_message_history(request: RunRequest) -> list:
    """Map a neutral RunRequest to a pydantic-ai message_history list.

    The agent's own ``instructions`` are applied by pydantic-ai; this function
    handles only prior conversation turns.
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
    for turn in request.history:
        if not turn.text or turn.role not in FORWARDED_ROLES:
            continue
        if turn.role == "user":
            out.append(ModelRequest(parts=[UserPromptPart(content=turn.text)]))
        elif turn.role == "system":
            out.append(ModelRequest(parts=[SystemPromptPart(content=turn.text)]))
        elif turn.role == "assistant":
            out.append(ModelResponse(parts=[TextPart(content=turn.text)]))
    return out


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


async def run_agent(agent: Agent, request: RunRequest) -> RunResult:
    """Run the pydantic-ai agent and return the neutral result shape."""
    message_history = _to_message_history(request)
    try:
        result = await agent.run(request.prompt, message_history=message_history)
    except Exception as exc:  # noqa: BLE001 — normalized for the façade
        raise normalize_agent_run_error(exc) from exc
    return RunResult(text=_result_text(result), usage=_result_usage(result))
