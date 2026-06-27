"""LangGraph adapter-specific guardrails and translation tests."""

from __future__ import annotations

import ast
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agentkit_serve import agent_factory
from agentkit_serve_common.config import AgentSpec, ToolSpec
from agentkit_serve_common.conversation import ConversationTurn, RunRequest
from agentkit_serve_common.runtime import AgentRunError

_MODEL_NAME = "gpt-4o-mini"


def _spec_data(api_key_env: str | None = "OPENAI_API_KEY", tools: list[dict] | None = None) -> dict:
    model = {
        "provider": "openai-compatible",
        "baseURL": "https://api.openai.com/v1",
        "name": _MODEL_NAME,
    }
    if api_key_env is not None:
        model["apiKeyEnv"] = api_key_env
    return {
        "abiVersion": "v0",
        "metadata": {"name": "test-agent"},
        "model": model,
        "instructions": "Be helpful.",
        "tools": tools or [],
        "expose": {"openai": True, "port": 8080},
    }


def _spec(api_key_env: str | None = "OPENAI_API_KEY", tools: list[dict] | None = None) -> AgentSpec:
    return AgentSpec.model_validate(_spec_data(api_key_env=api_key_env, tools=tools))


def test_missing_api_key_env_fails_secret_free(monkeypatch):
    monkeypatch.delenv("MISSING_MODEL_KEY", raising=False)
    with pytest.raises(agent_factory.AgentBuildError) as ei:
        agent_factory._resolve_api_key(_spec(api_key_env="MISSING_MODEL_KEY"))
    msg = str(ei.value)
    assert "MISSING_MODEL_KEY" in msg
    assert "sk-" not in msg


def test_no_api_key_env_uses_placeholder(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert agent_factory._resolve_api_key(_spec(api_key_env=None)) == "not-needed"


def test_message_mapping_drops_unsupported_roles_and_does_not_duplicate_instructions():
    messages = agent_factory._to_messages(
        RunRequest(
            prompt="final question",
            history=(
                ConversationTurn("system", "request system"),
                ConversationTurn("user", "first question"),
                ConversationTurn("assistant", "first answer"),
                ConversationTurn("tool", "client tool result"),
                ConversationTurn("user", ""),
                ConversationTurn("unknown", "ignored"),
            ),
        )
    )

    assert [type(m) for m in messages] == [SystemMessage, HumanMessage, AIMessage, HumanMessage]
    assert [m.content for m in messages] == [
        "request system",
        "first question",
        "first answer",
        "final question",
    ]


def test_message_text_extraction_string_blocks_and_fallback():
    assert agent_factory._message_text(AIMessage(content="plain")) == "plain"
    assert (
        agent_factory._message_text(
            AIMessage(
                content=[
                    {"type": "text", "text": "hello"},
                    {"type": "image", "url": "ignored"},
                    " world",
                ]
            )
        )
        == "hello world"
    )
    non_text = [{"type": "image", "url": "x"}]
    assert agent_factory._message_text(AIMessage(content=non_text)) == str(non_text)


def test_message_usage_extraction():
    msg = AIMessage(
        content="ok",
        usage_metadata={"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
    )
    assert agent_factory._message_usage(msg) == {
        "prompt_tokens": 3,
        "completion_tokens": 4,
        "total_tokens": 7,
    }
    # Some providers omit total_tokens; LangChain's AIMessage model currently
    # requires it when constructing usage_metadata, so use a duck-typed object to
    # exercise the adapter's defensive mapper.
    msg_no_total = SimpleNamespace(usage_metadata={"input_tokens": 3, "output_tokens": 4})
    assert agent_factory._message_usage(msg_no_total)["total_tokens"] == 7
    assert agent_factory._message_usage(AIMessage(content="ok")) == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def test_state_usage_aggregates_all_ai_messages_in_tool_loop():
    state = {
        "messages": [
            HumanMessage(content="use a tool"),
            AIMessage(
                content="",
                usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
            ),
            HumanMessage(content="tool result"),
            AIMessage(
                content="final",
                usage_metadata={"input_tokens": 20, "output_tokens": 5, "total_tokens": 25},
            ),
        ]
    }

    assert agent_factory._state_usage(state) == {
        "prompt_tokens": 30,
        "completion_tokens": 7,
        "total_tokens": 37,
    }


def test_status_unwraps_framework_exception_chain():
    class _BadRequest(Exception):
        status_code = 400

    class _Wrapped(Exception):
        def __init__(self, msg, inner=None):
            super().__init__(msg)
            self.inner_exception = inner

    assert agent_factory._status_of(_BadRequest()) == 400
    try:
        try:
            raise _BadRequest("upstream 400")
        except _BadRequest as ex:
            raise _Wrapped("wrapped", inner=ex) from ex
    except _Wrapped as exc:
        assert agent_factory._status_of(exc) == 400
    assert agent_factory._status_of(ValueError("x")) == 502


def test_last_ai_message_errors_on_bad_state():
    with pytest.raises(AgentRunError) as ei:
        agent_factory._last_ai_message({"messages": [HumanMessage(content="hi")]})
    assert ei.value.status == 502
    assert ei.value.code == "LangGraphResultError"


def test_tool_env_declared_only_and_model_key_not_inherited(monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "model-secret")
    monkeypatch.setenv("TOOL_SECRET", "tool-secret")
    monkeypatch.setenv("UNDECLARED", "must-not-pass")
    tool = ToolSpec(name="fetch", command=["uvx", "mcp-server-fetch"], env=["TOOL_SECRET"])

    conn = agent_factory.build_mcp_connection(tool)

    assert conn["env"] == {"TOOL_SECRET": "tool-secret"}
    assert "MODEL_API_KEY" not in conn["env"]
    assert "UNDECLARED" not in conn["env"]


def test_tool_env_rejects_interpolation_of_undeclared_secret(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "model-secret")
    monkeypatch.setenv("TOOL_CONFIG", "token=${OPENAI_API_KEY}")
    tool = ToolSpec(name="fetch", command=["uvx", "mcp-server-fetch"], env=["TOOL_CONFIG"])

    with pytest.raises(agent_factory.AgentBuildError) as ei:
        agent_factory.build_mcp_connection(tool)

    msg = str(ei.value)
    assert "OPENAI_API_KEY" in msg
    assert "model-secret" not in msg


def test_tool_env_allows_interpolation_of_declared_env(monkeypatch):
    monkeypatch.setenv("TOOL_SECRET", "secret")
    monkeypatch.setenv("TOOL_CONFIG", "token=${TOOL_SECRET}")
    tool = ToolSpec(
        name="fetch",
        command=["uvx", "mcp-server-fetch"],
        env=["TOOL_CONFIG", "TOOL_SECRET"],
    )

    conn = agent_factory.build_mcp_connection(tool)

    assert conn["env"] == {"TOOL_CONFIG": "token=${TOOL_SECRET}", "TOOL_SECRET": "secret"}


def test_tool_env_empty_is_explicit_empty_dict(monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "model-secret")
    tool = ToolSpec(name="fetch", command=["uvx", "mcp-server-fetch"], env=[])
    assert agent_factory.build_mcp_connection(tool)["env"] == {}


def test_empty_tool_command_fails():
    # The strict ABI reader now rejects empty commands, but keep a defensive
    # adapter-level guard for hand-constructed specs.
    tool = ToolSpec.model_construct(name="fetch", command=[], env=[])
    with pytest.raises(agent_factory.AgentBuildError):
        agent_factory.build_mcp_connection(tool)


def test_mcp_timeout_env_default_invalid_negative_and_positive(monkeypatch):
    monkeypatch.delenv("AGENTKIT_MCP_TIMEOUT", raising=False)
    assert agent_factory._mcp_init_timeout() == 120.0

    monkeypatch.setenv("AGENTKIT_MCP_TIMEOUT", "garbage")
    assert agent_factory._mcp_init_timeout() == 120.0

    monkeypatch.setenv("AGENTKIT_MCP_TIMEOUT", "-1")
    assert agent_factory._mcp_init_timeout() == 120.0

    monkeypatch.setenv("AGENTKIT_MCP_TIMEOUT", "2.5")
    assert agent_factory._mcp_init_timeout() == 2.5
    conn = agent_factory.build_mcp_connection(ToolSpec(name="fetch", command=["cmd"], env=[]))
    assert conn["session_kwargs"]["read_timeout_seconds"].total_seconds() == 2.5


def test_missing_model_key_fails_before_loading_tools(monkeypatch):
    monkeypatch.delenv("MISSING_MODEL_KEY", raising=False)
    spec = _spec(
        api_key_env="MISSING_MODEL_KEY",
        tools=[{"name": "fetch", "command": ["cmd"], "env": []}],
    )
    runtime = agent_factory.LangGraphRuntime(spec)

    async def _should_not_load():
        raise AssertionError("tools loaded before model API key was resolved")

    with mock.patch.object(runtime, "_load_tools", _should_not_load):
        with pytest.raises(agent_factory.AgentBuildError):
            asyncio.run(runtime.__aenter__())


def test_load_tools_uses_persistent_sessions_and_prefixed_names():
    spec = _spec(tools=[{"name": "fetch", "command": ["cmd", "arg"], "env": []}])
    runtime = agent_factory.LangGraphRuntime(spec)
    initialized = []
    entered = []
    exited = []
    seen_connections = []

    class _Session:
        async def initialize(self):
            initialized.append(True)

    class _FakeClient:
        def __init__(self, connections, tool_name_prefix=False):
            seen_connections.append((connections, tool_name_prefix))

        @asynccontextmanager
        async def session(self, server_name, auto_initialize=True):
            entered.append((server_name, auto_initialize))
            try:
                yield _Session()
            finally:
                exited.append(server_name)

    async def _fake_load(session, *, server_name, tool_name_prefix):
        assert server_name == "fetch"
        assert tool_name_prefix is True
        return [SimpleNamespace(name=f"{server_name}_fetch")]

    with (
        mock.patch("agentkit_serve.agent_factory.MultiServerMCPClient", _FakeClient),
        mock.patch("agentkit_serve.agent_factory.load_mcp_tools", _fake_load),
    ):
        tools = asyncio.run(runtime._load_tools())
        asyncio.run(runtime.stack.aclose())

    assert tools[0].name == "fetch_fetch"
    assert initialized == [True]
    assert entered == [("fetch", False)]
    assert exited == ["fetch"]
    assert seen_connections[0][1] is True
    assert seen_connections[0][0]["fetch"]["env"] == {}


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def test_no_azure_or_foundry_imports_in_generic_langgraph_adapter():
    forbidden = {
        "azure",
        "azure_ai_agentserver",
        "azure_ai_projects",
        "langchain_azure_ai",
    }
    pkg_dir = Path(agent_factory.__file__).parent
    for path in pkg_dir.glob("*.py"):
        leaked = _imported_roots(path) & forbidden
        assert not leaked, f"{path.name} imports Azure/Foundry symbols {sorted(leaked)}"



def test_langgraph_rejects_model_workload_identity_auth():
    data = _spec_data()
    data["model"]["auth"] = {"type": "workload-identity-token", "audience": "https://ai.azure.com/.default"}
    spec = AgentSpec.model_validate(data)

    with pytest.raises(agent_factory.AgentBuildError, match="model.auth"):
        agent_factory.build_runtime(spec)


def test_langgraph_rejects_context_providers():
    data = _spec_data()
    data["context"] = {"providers": [{"type": "skills", "source": "filesystem", "path": "/agent/skills"}]}
    spec = AgentSpec.model_validate(data)

    with pytest.raises(agent_factory.AgentBuildError, match="context providers"):
        agent_factory.build_runtime(spec)
