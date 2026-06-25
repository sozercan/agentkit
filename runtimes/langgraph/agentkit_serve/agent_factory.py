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
import os
from contextlib import AsyncExitStack
from datetime import timedelta
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI

from agentkit_serve_common.config import AgentSpec, ToolSpec
from agentkit_serve_common.runtime import AgentRunError, RunResult

# Placeholder API key for OpenAI-compatible endpoints that need no auth (many
# local servers reject an EMPTY string but accept any non-empty token). Used only
# when the spec declares no apiKeyEnv. Never a real secret.
_NO_AUTH_PLACEHOLDER = "not-needed"

# Roles forwarded from the request history into the LangGraph conversation. The
# agent owns its tools, so client-supplied tool turns are meaningless and are
# dropped by the shared server before they reach here; this is a defensive gate.
_FORWARDED_ROLES = frozenset({"system", "user", "assistant"})

# Seconds to wait for a stdio MCP server's initialize handshake. A cold `uvx` or
# `npx` tool may download/install before speaking MCP, so match pydantic-ai's
# generous default and let operators tune via env.
_DEFAULT_MCP_INIT_TIMEOUT = 120.0


class AgentBuildError(Exception):
    """Raised when the agent cannot be constructed (e.g. missing API key env)."""


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


def build_model(spec: AgentSpec) -> ChatOpenAI:
    """Construct the OpenAI-compatible chat model pointed at ``model.baseURL``."""
    return ChatOpenAI(
        model=spec.model.name,
        base_url=spec.model.base_url,
        api_key=_resolve_api_key(spec),
    )


def _tool_env(tool: ToolSpec) -> dict[str, str]:
    """Select ONLY the env vars NAMED in ``tool.env`` (secret-bleed rule).

    The MCP subprocess must never inherit the full container environment — that
    would bleed the model API key (and every other secret) into every tool. We
    pass through exactly the declared names that are actually present. Passing an
    explicit empty dict is important: langchain-mcp-adapters intentionally
    inherits a default env subset only when env is omitted/None.
    """
    return {name: os.environ[name] for name in tool.env if name in os.environ}


def build_mcp_connection(tool: ToolSpec) -> dict[str, Any]:
    """Convert an AgentKit stdio tool declaration into a LangChain MCP connection."""
    if not tool.command:
        # config.AgentSpec allows an empty command shape; the writer never emits
        # one, but guard anyway so a hand-rolled agent.yaml fails clearly.
        raise AgentBuildError(
            f"tool {tool.name!r} has an empty command; a stdio MCP server needs "
            f'at least the executable, e.g. command: ["uvx", "mcp-server-fetch"]'
        )

    timeout = _mcp_init_timeout()
    return {
        "transport": "stdio",
        "command": tool.command[0],
        "args": list(tool.command[1:]),
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

    async def __aenter__(self) -> "LangGraphRuntime":
        try:
            tools = await self._load_tools()
            self.graph = create_agent(
                model=build_model(self.spec),
                tools=tools,
                system_prompt=self.spec.instructions,
            )
            return self
        except Exception:
            await self.stack.aclose()
            raise

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.graph = None
        await self.stack.aclose()

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


def build_agent(spec: AgentSpec) -> LangGraphRuntime:
    """Assemble the LangGraph runtime wrapper for one AgentKit agent spec."""
    return LangGraphRuntime(spec)


def _to_messages(history: list[tuple[str, str]] | None, prompt: str) -> list[BaseMessage]:
    """Map neutral ``(role, text)`` history + final prompt to LangChain messages.

    The agent's own ``spec.instructions`` is passed as ``system_prompt`` when the
    graph is created; do not duplicate it here.
    """
    messages: list[BaseMessage] = []
    for role, text in history or []:
        if role not in _FORWARDED_ROLES or not text:
            continue
        if role == "system":
            messages.append(SystemMessage(content=text))
        elif role == "user":
            messages.append(HumanMessage(content=text))
        elif role == "assistant":
            messages.append(AIMessage(content=text))
    messages.append(HumanMessage(content=prompt))
    return messages


def _status_of(exc: Exception) -> int:
    """Best-effort upstream HTTP status from a model/SDK error (else 502).

    Duck-typed on ``status_code`` and walks common wrapper links, matching the MAF
    adapter's behavior for OpenAI SDK errors wrapped by framework exceptions.
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
        nxt = getattr(cur, "inner_exception", None) or cur.__cause__ or cur.__context__
        cur = nxt
    return 502


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


async def run_agent(
    agent: LangGraphRuntime,
    prompt: str,
    history: list[tuple[str, str]] | None = None,
) -> RunResult:
    """Run the compiled LangGraph once and return a neutral ``RunResult``."""
    if agent.graph is None:
        raise AgentRunError("agent graph is not initialized", status=500, code="AgentNotInitialized")

    try:
        state = await agent.graph.ainvoke({"messages": _to_messages(history, prompt)})
    except AgentRunError:
        raise
    except Exception as exc:  # noqa: BLE001 — normalized for the façade
        # Preserve the original framework/model exception class name in error.code.
        raise AgentRunError(
            f"agent run failed: {exc}",
            status=_status_of(exc),
            code=exc.__class__.__name__,
        ) from exc

    msg = _last_ai_message(state)
    return RunResult(text=_message_text(msg), usage=_state_usage(state))
