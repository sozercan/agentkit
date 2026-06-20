"""Build a pydantic-ai :class:`Agent` from a validated :class:`AgentSpec`.

Verified against the INSTALLED pydantic-ai (1.107.x). Key facts baked in here:

* The OpenAI-compatible model class is ``OpenAIChatModel`` (``OpenAIModel`` is a
  deprecated alias); its base_url/api_key come from an ``OpenAIProvider``.
* stdio MCP servers are ``MCPServerStdio(command, args, *, env=...)`` and are
  passed to the agent as ``toolsets`` (the old ``mcp_servers=`` kwarg is gone).
* The agent is itself an async context manager: ``async with agent:`` starts the
  MCP subprocesses; that is the modern replacement for ``run_mcp_servers()``.

The agent OWNS its system prompt (spec.instructions) and its tools — request-side
tools are rejected by the server, never merged here.
"""

from __future__ import annotations

import os

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from .config import AgentSpec, ToolSpec

# Placeholder API key for OpenAI-compatible endpoints that need no auth (many
# local servers reject an EMPTY string but accept any non-empty token). Used only
# when the spec declares no apiKeyEnv. Never a real secret.
_NO_AUTH_PLACEHOLDER = "not-needed"

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


def build_tool_server(tool: ToolSpec) -> MCPServerStdio:
    """Create a stdio MCP server for one tool spec."""
    if not tool.command:
        # config.AgentSpec allows an empty command shape; the writer never emits
        # one, but guard anyway so a hand-rolled agent.yaml fails clearly.
        raise AgentBuildError(
            f"tool {tool.name!r} has an empty command; a stdio MCP server needs "
            f"at least the executable, e.g. command: [\"npx\", \"-y\", \"...\"]"
        )
    return MCPServerStdio(
        command=tool.command[0],
        args=list(tool.command[1:]),
        env=_tool_env(tool),
        # Generous init timeout: a cold uvx/npx tool installs its package before
        # speaking MCP, which exceeds pydantic-ai's 5s default (see above).
        timeout=_mcp_init_timeout(),
        # tool_prefix namespaces tool names so two servers can't collide.
        tool_prefix=tool.name,
    )


def build_agent(spec: AgentSpec) -> Agent:
    """Assemble the pydantic-ai agent: model + system prompt + stdio MCP toolsets."""
    model = build_model(spec)
    toolsets = [build_tool_server(t) for t in spec.tools]
    return Agent(
        model,
        instructions=spec.instructions,
        toolsets=toolsets,
    )
