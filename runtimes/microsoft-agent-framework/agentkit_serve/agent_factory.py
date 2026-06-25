"""Build a Microsoft Agent Framework (MAF) agent from a validated AgentSpec.

This module is the ONLY net-new surface of the MAF runtime adapter: it is the
framework-specific translation layer behind the frozen ``/agent/agent.yaml`` ABI.
The ABI loader, the ``/v1`` facade, and the CLI live in ``agentkit_serve_common``
(framework-neutral); this module satisfies its ``RuntimeFactory`` protocol.

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

Cross-runtime invariants such as API-key resolution, secret-safe tool env
projection, MCP timeout parsing, and error normalization live in
``agentkit_serve_common.adapter_support`` so this Module stays focused on MAF's
concrete Adapter shape.
"""

from __future__ import annotations

from agent_framework import Agent, MCPStdioTool, Message
from agent_framework.openai import OpenAIChatCompletionClient
from agentkit_serve_common.adapter_support import (
    FORWARDED_ROLES,
    AgentBuildError,
    declared_tool_env,
    normalize_agent_run_error,
    positive_int_env,
    resolve_api_key,
    split_tool_command,
    upstream_status_code,
)
from agentkit_serve_common.config import AgentSpec, ToolSpec
from agentkit_serve_common.conversation import RunRequest
from agentkit_serve_common.runtime import RunResult


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
    return positive_int_env(default=None)


def _resolve_api_key(spec: AgentSpec) -> str:
    """Compatibility wrapper around the shared adapter support Module."""
    return resolve_api_key(spec)


def build_client(spec: AgentSpec) -> OpenAIChatCompletionClient:
    """Construct the OpenAI-compatible chat client pointed at ``model.baseURL``."""
    return OpenAIChatCompletionClient(
        model=spec.model.name,
        base_url=spec.model.base_url,
        api_key=resolve_api_key(spec),
    )


def _tool_env(tool: ToolSpec) -> dict[str, str]:
    """Compatibility wrapper around the shared secret-safe tool env projection."""
    return declared_tool_env(tool)


def build_tool(tool: ToolSpec) -> MCPStdioTool:
    """Create a stdio MCP server for one tool spec.

    ``MCPStdioTool`` is an async context manager (``connect()``/``close()``); the
    server starts its subprocess once inside the agent's lifespan, mirroring the
    pydantic-ai adapter's ``async with agent:``.
    """
    command, args = split_tool_command(tool, example='["uvx", "mcp-server-fetch"]')
    kwargs: dict[str, object] = {
        "name": tool.name,  # the server's identity (used in error messages/spans)
        "command": command,
        "args": args,
        "env": declared_tool_env(tool),
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
    """Compatibility wrapper around shared upstream status extraction."""
    return upstream_status_code(exc)


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


def _to_messages(request: RunRequest) -> list[Message]:
    """Map a neutral RunRequest to MAF messages.

    The agent's own ``instructions`` are prepended by MAF as a system message; this
    function handles prior conversation turns plus the final user prompt.
    """
    messages: list[Message] = []
    for turn in request.history:
        if turn.role in FORWARDED_ROLES and turn.text:
            messages.append(Message(role=turn.role, contents=[turn.text]))
    messages.append(Message(role="user", contents=[request.prompt]))
    return messages


async def run_agent(agent: Agent, request: RunRequest) -> RunResult:
    """Run the MAF agent and return the neutral result shape."""
    messages = _to_messages(request)
    try:
        result = await agent.run(messages)
    except Exception as exc:  # noqa: BLE001 — normalized for the façade
        raise normalize_agent_run_error(exc) from exc
    return RunResult(text=_result_text(result), usage=_result_usage(result))
