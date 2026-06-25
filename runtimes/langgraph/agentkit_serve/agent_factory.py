"""Build a LangChain/LangGraph agent from a validated :class:`AgentSpec`.

This adapter keeps AgentKit's generic runtime boundary intact: it consumes the
same frozen ``/agent/agent.yaml`` ABI as the pydantic-ai and MAF adapters and
serves the same non-streaming OpenAI ``/v1/chat/completions`` façade through
``agentkit_serve_common``. LangGraph is used internally via LangChain's
``create_agent`` helper; arbitrary user-authored graphs and Foundry
``/responses``/``/invocations`` hosting are intentionally out of scope here.

Verified during implementation against the installed package set:

* ``langchain.agents.create_agent(model=..., tools=..., system_prompt=...)``
  returns a compiled LangGraph with ``ainvoke``.
* ``ChatOpenAI(model=..., base_url=..., api_key=...)`` is the generic
  OpenAI-compatible chat model client.
* ``MultiServerMCPClient.session(server_name, auto_initialize=False)`` plus
  ``load_mcp_tools(..., server_name=..., tool_name_prefix=True)`` keeps stdio MCP
  sessions open for the server lifespan and namespaces tool names.

THE LOCK-IN BOUNDARY: imports here are confined to LangChain/LangGraph/OpenAI/MCP
packages and ``agentkit_serve_common``. NEVER import Azure / Foundry hosting
packages from this generic runtime; a future Foundry mode should be a separate
adapter/target.
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from datetime import timedelta
from types import TracebackType
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI

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
from agentkit_serve_common.conversation import RunRequest
from agentkit_serve_common.runtime import AgentRunError, RunResult, RuntimeSession

# Seconds to wait for a stdio MCP server's initialize handshake. A cold `uvx` or
# `npx` tool may download/install before speaking MCP, so match pydantic-ai's
# generous default and let operators tune via env.
_DEFAULT_MCP_INIT_TIMEOUT = 120.0


def _mcp_init_timeout() -> float:
    """MCP stdio init timeout (seconds), overridable via AGENTKIT_MCP_TIMEOUT."""
    return positive_float_env(default=_DEFAULT_MCP_INIT_TIMEOUT)


def _resolve_api_key(spec: AgentSpec) -> str:
    """Compatibility wrapper over the shared API-key resolver."""
    return resolve_api_key(spec)


def build_model(spec: AgentSpec) -> ChatOpenAI:
    """Construct the OpenAI-compatible chat model pointed at ``model.baseURL``."""
    return ChatOpenAI(
        model=spec.model.name,
        base_url=spec.model.base_url,
        api_key=_resolve_api_key(spec),
    )


def _tool_env(tool: ToolSpec) -> dict[str, str]:
    """Compatibility wrapper over the shared declared-only tool env helper."""
    return declared_tool_env(tool)


def build_mcp_connection(tool: ToolSpec) -> dict[str, Any]:
    """Convert an AgentKit stdio tool declaration into a LangChain MCP connection."""
    command, args = split_tool_command(tool, example='["uvx", "mcp-server-fetch"]')
    timeout = _mcp_init_timeout()
    return {
        "transport": "stdio",
        "command": command,
        "args": args,
        # Declared-only env, even when empty; never let the MCP SDK inherit the
        # model key or process env by omission.
        "env": _tool_env(tool),
        # MCP ClientSession exposes a read timeout as a timedelta. Use the same
        # operator knob for request/read waits that we use for initialize.
        "session_kwargs": {"read_timeout_seconds": timedelta(seconds=timeout)},
    }


class LangGraphRuntime:
    """Async lifespan wrapper around a compiled LangGraph agent.

    The shared FastAPI server enters this object once for the process lifespan,
    so stdio MCP sessions stay warm across requests and close on shutdown.
    """

    def __init__(self, spec: AgentSpec) -> None:
        self.spec = spec
        self.stack = AsyncExitStack()
        self.graph: Any | None = None
        self.client: MultiServerMCPClient | None = None

    async def __aenter__(self) -> RuntimeSession:
        try:
            model = build_model(self.spec)
            tools = await self._load_tools()
            self.graph = create_agent(
                model=model,
                tools=tools,
                system_prompt=self.spec.instructions,
            )
            return self
        except Exception:
            await self.stack.aclose()
            raise

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        self.graph = None
        await self.stack.aclose()
        return None

    async def run(self, request: RunRequest) -> RunResult:
        return await run_agent(self, request)

    async def _load_tools(self) -> list[Any]:
        if not self.spec.tools:
            return []

        connections = {tool.name: build_mcp_connection(tool) for tool in self.spec.tools}
        self.client = MultiServerMCPClient(connections, tool_name_prefix=True)

        tools: list[Any] = []
        for tool in self.spec.tools:
            session_cm = self.client.session(tool.name, auto_initialize=False)
            session = await self.stack.enter_async_context(session_cm)
            await asyncio.wait_for(session.initialize(), timeout=_mcp_init_timeout())
            tools.extend(
                await load_mcp_tools(
                    session,
                    server_name=tool.name,
                    tool_name_prefix=True,
                )
            )
        return tools


def build_runtime(spec: AgentSpec) -> LangGraphRuntime:
    """Build the runtime session consumed by the shared server."""
    return LangGraphRuntime(spec)


def build_agent(spec: AgentSpec) -> LangGraphRuntime:
    """Compatibility alias for wrappers that build an adapter runtime directly."""
    return build_runtime(spec)


def _to_messages(request: RunRequest) -> list[BaseMessage]:
    """Map a neutral RunRequest to LangChain messages.

    The agent's own ``spec.instructions`` is passed as ``system_prompt`` when the
    graph is created; do not duplicate it here.
    """
    messages: list[BaseMessage] = []
    for turn in request.history:
        if turn.role not in FORWARDED_ROLES or not turn.text:
            continue
        if turn.role == "system":
            messages.append(SystemMessage(content=turn.text))
        elif turn.role == "user":
            messages.append(HumanMessage(content=turn.text))
        elif turn.role == "assistant":
            messages.append(AIMessage(content=turn.text))
    messages.append(HumanMessage(content=request.prompt))
    return messages


def _status_of(exc: Exception) -> int:
    """Compatibility wrapper over shared upstream status unwrapping."""
    return upstream_status_code(exc)


def _state_messages(state: Any) -> list[Any]:
    """Extract the LangGraph ``messages`` list or raise a normalized run error."""
    messages = state.get("messages") if isinstance(state, dict) else getattr(state, "messages", None)
    if not isinstance(messages, list):
        raise AgentRunError(
            "agent run failed: LangGraph result did not contain a messages list",
            status=502,
            code="LangGraphResultError",
        )
    return messages


def _last_ai_message(state: Any) -> AIMessage:
    """Extract the final AIMessage from a LangGraph ``ainvoke`` state."""
    for msg in reversed(_state_messages(state)):
        if isinstance(msg, AIMessage):
            return msg
    raise AgentRunError(
        "agent run failed: LangGraph result did not contain an assistant message",
        status=502,
        code="LangGraphResultError",
    )


def _message_text(message: AIMessage) -> str:
    """Extract assistant text from LangChain message content.

    LangChain may surface content as a string or as a list of content blocks. Text
    blocks are joined; non-text blocks are ignored unless there is no text at all,
    in which case we fall back to ``str(content)`` for debuggability.
    """
    content = message.content
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
                continue
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(str(part.get("text", "")))
                continue
            part_type = getattr(part, "type", None)
            part_text = getattr(part, "text", None)
            if part_type == "text" and part_text is not None:
                text_parts.append(str(part_text))
        if text_parts:
            return "".join(text_parts)
    return str(content)


def _message_usage(message: AIMessage) -> dict[str, int]:
    """Map one LangChain ``usage_metadata`` block to the OpenAI usage shape."""
    usage = getattr(message, "usage_metadata", None)
    get = getattr(usage, "get", None)
    if not callable(get):
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    prompt_tokens = int(get("input_tokens", 0) or 0)
    completion_tokens = int(get("output_tokens", 0) or 0)
    total = get("total_tokens", None)
    total_tokens = int(total) if total is not None else prompt_tokens + completion_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _state_usage(state: Any) -> dict[str, int]:
    """Aggregate token usage across every model call in a LangGraph run.

    A tool-using LangGraph agent can make multiple model calls: one AIMessage may
    request a tool, and a later AIMessage contains the final answer. Each message
    carries per-call ``usage_metadata`` when available, so the OpenAI facade should
    report the sum for the whole agent turn rather than only the final answer.
    """
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for msg in _state_messages(state):
        if not isinstance(msg, AIMessage):
            continue
        usage = _message_usage(msg)
        totals["prompt_tokens"] += usage["prompt_tokens"]
        totals["completion_tokens"] += usage["completion_tokens"]
        totals["total_tokens"] += usage["total_tokens"]
    return totals


async def run_agent(agent: LangGraphRuntime, request: RunRequest) -> RunResult:
    """Run the compiled LangGraph once and return a neutral ``RunResult``."""
    if agent.graph is None:
        raise AgentRunError("agent graph is not initialized", status=500, code="AgentNotInitialized")

    try:
        state = await agent.graph.ainvoke({"messages": _to_messages(request)})
    except AgentRunError:
        raise
    except Exception as exc:  # noqa: BLE001 — normalized for the façade
        raise normalize_agent_run_error(exc) from exc

    msg = _last_ai_message(state)
    return RunResult(text=_message_text(msg), usage=_state_usage(state))
