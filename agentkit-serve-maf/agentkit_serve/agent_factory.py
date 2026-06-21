"""Build a Microsoft Agent Framework (MAF) agent from a validated AgentSpec.

This module is the ONLY net-new surface of the MAF runtime adapter: it is the
framework-specific translation layer behind the frozen ``/agent/agent.yaml`` ABI.
``config.py``, ``server.py`` and ``__main__.py`` are framework-agnostic.

Verified firsthand against the INSTALLED packages (agent-framework-core 1.9.0,
agent-framework-openai 1.8.2) — NOT from secondhand docs:

* ``OpenAIChatCompletionClient(model=, api_key=, base_url=)`` — the classic
  ``/v1/chat/completions`` client; proven against a plain OpenAI-compatible mock.
  (The plan's ``OpenAIChatClient`` also exists but targets the Responses API; the
  completion client is the one generic ``/v1`` endpoints — AIKit, vLLM, proxies —
  implement, so it is the AgentKit choice. Plan §4.1 / Open Q1: resolved.)
* ``MCPStdioTool(name, command, *, args=, env=, request_timeout=int|None)`` — a
  stdio MCP server, in the CORE package. There is NO ``timeout=`` kwarg; the knob
  is ``request_timeout`` (seconds), which maps to the MCP session read timeout and
  defaults to ``None`` (no cap). Plan §4.2 / Open Q2: resolved.
* ``Agent(client=, instructions=, *, name=, tools=)`` — note ``client=``, NOT the
  plan's ``chat_client=`` (which does not exist and would ``TypeError``).
  ``await agent.run(messages)`` returns an ``AgentResponse`` whose ``.text`` is the
  final answer and whose ``.usage_details`` carries token counts. Plan §4.3 / Q3.

THE LOCK-IN BOUNDARY (plan §12): imports here are confined to ``agent_framework``
(core) + ``agent_framework.openai``. NEVER an Azure / Foundry / CopilotStudio
package — including the first-party submodules ``agent_framework.azure`` /
``.foundry`` / ``.microsoft`` (which re-export the cloud surface). ``import
agent_framework.openai`` was verified to pull in zero ``azure*`` modules. The
boundary is enforced by ``tests/test_guardrails.py`` (AST-based, which is why a
naive source grep is not relied upon — it would false-positive on this docstring).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from agent_framework import Agent, Message, MCPStdioTool
from agent_framework.openai import OpenAIChatCompletionClient

from .config import AgentSpec, ToolSpec

# Placeholder API key for OpenAI-compatible endpoints that need no auth (many
# local servers reject an EMPTY string but accept any non-empty token). Used only
# when the spec declares no apiKeyEnv. Never a real secret.
_NO_AUTH_PLACEHOLDER = "not-needed"

# Roles we forward from the request history into the MAF conversation. The agent
# owns its tools, so client-supplied ``tool`` turns are meaningless and dropped by
# the server before they reach here; this set is a defensive second gate.
_FORWARDED_ROLES = frozenset({"system", "user", "assistant"})


class AgentBuildError(Exception):
    """Raised when the agent cannot be constructed (e.g. missing API key env)."""


class AgentRunError(Exception):
    """A runtime/model failure during ``agent.run``, carrying an HTTP status.

    The server maps this to the OpenAI error envelope WITHOUT importing any
    framework or model-SDK type — keeping ``server.py`` framework-agnostic.
    """

    def __init__(self, message: str, status: int = 502) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class RunResult:
    """Framework-neutral result of one agent run (the seam ``server.py`` reads)."""

    text: str
    usage: dict[str, int] = field(default_factory=dict)


def _mcp_request_timeout() -> int | None:
    """MCP stdio request timeout (seconds), overridable via ``AGENTKIT_MCP_TIMEOUT``.

    DELIBERATE DIVERGENCE from the pydantic-ai adapter (which defaults to 120s):
    MAF's ``request_timeout`` defaults to ``None`` (no read-timeout cap), which is
    ALREADY tolerant of a cold ``uvx``/``npx`` tool that downloads its package
    before speaking MCP. Capping it would risk killing a slow cold start, so we
    leave it unset by default and only honor an EXPLICIT operator override. This
    preserves the ``AGENTKIT_MCP_TIMEOUT`` knob (plan Open Q2) without regressing
    cold-start behavior.
    """
    raw = os.environ.get("AGENTKIT_MCP_TIMEOUT")
    if not raw:
        return None
    try:
        val = int(float(raw))
    except ValueError:
        return None
    return val if val > 0 else None


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


def build_client(spec: AgentSpec) -> OpenAIChatCompletionClient:
    """Construct the OpenAI-compatible chat client pointed at ``model.baseURL``."""
    return OpenAIChatCompletionClient(
        model=spec.model.name,
        base_url=spec.model.base_url,
        api_key=_resolve_api_key(spec),
    )


def _tool_env(tool: ToolSpec) -> dict[str, str]:
    """Select ONLY the env vars NAMED in ``tool.env`` (plan §10 secret-bleed rule).

    The MCP subprocess must never inherit the full container environment — that
    would bleed the model API key (and every other secret) into every tool. We
    pass through exactly the declared names that are actually present.
    """
    return {name: os.environ[name] for name in tool.env if name in os.environ}


def build_tool(tool: ToolSpec) -> MCPStdioTool:
    """Create a stdio MCP server for one tool spec.

    ``MCPStdioTool`` is an async context manager (``connect()``/``close()``); the
    server starts its subprocess once inside the agent's lifespan, mirroring the
    pydantic-ai adapter's ``async with agent:``.
    """
    if not tool.command:
        # config.AgentSpec allows an empty command shape; the writer never emits
        # one, but guard anyway so a hand-rolled agent.yaml fails clearly.
        raise AgentBuildError(
            f"tool {tool.name!r} has an empty command; a stdio MCP server needs "
            f'at least the executable, e.g. command: ["uvx", "mcp-server-fetch"]'
        )
    kwargs: dict[str, object] = {
        "name": tool.name,  # the server's identity (used in error messages/spans)
        "command": tool.command[0],
        "args": list(tool.command[1:]),
        "env": _tool_env(tool),
        # tool_name_prefix namespaces the tool names THIS server exposes, so two
        # servers can't collide (MAF raises "Duplicate tool name" otherwise). This
        # mirrors the pydantic-ai adapter's tool_prefix=tool.name. NOTE: the server
        # `name` above does NOT namespace exposed tools — tool_name_prefix is the
        # knob (verified against agent-framework-core: only tool_name_prefix feeds
        # _build_prefixed_mcp_name).
        "tool_name_prefix": tool.name,
    }
    timeout = _mcp_request_timeout()
    if timeout is not None:
        kwargs["request_timeout"] = timeout
    return MCPStdioTool(**kwargs)


def build_agent(spec: AgentSpec) -> Agent:
    """Assemble the MAF agent: client + system prompt + stdio MCP tools.

    The agent OWNS its system prompt (``spec.instructions``) and tools —
    request-side tools are rejected by the server, never merged here.
    """
    return Agent(
        client=build_client(spec),
        instructions=spec.instructions,
        name=spec.metadata.name,
        tools=[build_tool(t) for t in spec.tools],
    )


def _status_of(exc: Exception) -> int:
    """Best-effort upstream HTTP status from a model/SDK error (else 502).

    Duck-typed on ``status_code`` (the OpenAI SDK's ``APIStatusError`` exposes it)
    so we map a real upstream 4xx/5xx through WITHOUT importing the SDK error type
    — keeping the dependency surface minimal and version-resilient.
    """
    code = getattr(exc, "status_code", None)
    if isinstance(code, int) and 400 <= code <= 599:
        return code
    return 502


def _result_text(result: object) -> str:
    """Extract the final assistant text from a MAF ``AgentResponse``."""
    text = getattr(result, "text", None)
    return text if isinstance(text, str) else str(result)


def _result_usage(result: object) -> dict[str, int]:
    """Map MAF ``usage_details`` to the OpenAI usage block (zeros if unknown).

    MAF exposes ``usage_details`` as a dict-like with ``input_token_count`` /
    ``output_token_count`` / ``total_token_count``. The echo client used in tests
    reports none, so every access is guarded.
    """
    details = getattr(result, "usage_details", None)
    get = getattr(details, "get", None)
    if not callable(get):
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    prompt_tokens = int(get("input_token_count", 0) or 0)
    completion_tokens = int(get("output_token_count", 0) or 0)
    total = get("total_token_count", None)
    total_tokens = int(total) if total is not None else prompt_tokens + completion_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _to_messages(history: list[tuple[str, str]] | None, prompt: str) -> list[Message]:
    """Map neutral ``(role, text)`` history + the final user prompt to MAF messages.

    The agent's own ``instructions`` are prepended by MAF as a system message; this
    mirrors the pydantic-ai adapter's ``_split_conversation`` mapping shape.
    """
    messages: list[Message] = []
    for role, text in history or []:
        if role in _FORWARDED_ROLES and text:
            messages.append(Message(role=role, contents=[text]))
    messages.append(Message(role="user", contents=[prompt]))
    return messages


async def run_agent(agent: Agent, prompt: str, history: list[tuple[str, str]] | None = None) -> RunResult:
    """Framework-agnostic run helper so ``server.py`` stays identical across adapters.

    Accepts the neutral ``(history, prompt)`` the server produces, runs the MAF
    agent once (non-streaming), and returns a neutral :class:`RunResult`. Any
    framework/model error is normalized to :class:`AgentRunError` carrying an HTTP
    status, so the server never imports a framework or SDK exception type.
    """
    messages = _to_messages(history, prompt)
    try:
        result = await agent.run(messages)
    except Exception as exc:  # noqa: BLE001 — normalized for the façade
        raise AgentRunError(f"agent run failed: {exc}", status=_status_of(exc)) from exc
    return RunResult(text=_result_text(result), usage=_result_usage(result))
