from __future__ import annotations

import os
import sys
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest

from agentkit_serve_common import adapter_support as support
from agentkit_serve_common.config import AgentSpec, ToolSpec
from agentkit_serve_common.runtime import AgentRunError


def _spec(api_key_env: str | None = "OPENAI_API_KEY") -> AgentSpec:
    model = {
        "provider": "openai-compatible",
        "baseURL": "https://api.openai.com/v1",
        "name": "gpt-4o-mini",
    }
    if api_key_env is not None:
        model["apiKeyEnv"] = api_key_env
    return AgentSpec.model_validate(
        {
            "abiVersion": "v0",
            "metadata": {"name": "test-agent"},
            "model": model,
            "instructions": "Be helpful.",
            "tools": [],
            "expose": {"openai": True, "port": 8080},
        }
    )


def test_resolve_api_key_uses_declared_env_name_only():
    with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "secret-value"}, clear=True):
        assert support.resolve_api_key(_spec()) == "secret-value"


def test_resolve_api_key_without_declared_env_uses_placeholder():
    with mock.patch.dict(os.environ, {}, clear=True):
        assert support.resolve_api_key(_spec(api_key_env=None)) == support.NO_AUTH_API_KEY


def test_resolve_api_key_missing_declared_env_fails_secret_free():
    with mock.patch.dict(os.environ, {"OTHER_SECRET": "do-not-mention"}, clear=True):
        with pytest.raises(support.AgentBuildError) as exc:
            support.resolve_api_key(_spec(api_key_env="MISSING_KEY"))
    msg = str(exc.value)
    assert "MISSING_KEY" in msg
    assert "do-not-mention" not in msg


def test_declared_tool_env_passes_only_declared_present_names():
    tool = ToolSpec(
        name="fetch",
        command=["uvx", "mcp-server-fetch"],
        env=["FETCH_TOKEN", "ABSENT_VAR"],
    )
    with mock.patch.dict(
        os.environ,
        {"FETCH_TOKEN": "tok", "OPENAI_API_KEY": "sk-should-not-leak", "OTHER": "x"},
        clear=True,
    ):
        env = support.declared_tool_env(tool)
    assert env == {"FETCH_TOKEN": "tok"}


def test_declared_tool_env_rejects_interpolation_of_undeclared_env():
    tool = ToolSpec(name="fetch", command=["uvx", "mcp-server-fetch"], env=["FETCH_CONFIG"])
    with mock.patch.dict(
        os.environ,
        {"FETCH_CONFIG": "token=${OPENAI_API_KEY}", "OPENAI_API_KEY": "sk-should-not-leak"},
        clear=True,
    ):
        with pytest.raises(support.AgentBuildError) as exc:
            support.declared_tool_env(tool)
    msg = str(exc.value)
    assert "OPENAI_API_KEY" in msg
    assert "sk-should-not-leak" not in msg


def test_declared_tool_env_allows_interpolation_of_declared_env():
    tool = ToolSpec(
        name="fetch",
        command=["uvx", "mcp-server-fetch"],
        env=["FETCH_CONFIG", "FETCH_TOKEN"],
    )
    with mock.patch.dict(
        os.environ,
        {"FETCH_CONFIG": "token=${FETCH_TOKEN}", "FETCH_TOKEN": "tok"},
        clear=True,
    ):
        env = support.declared_tool_env(tool)
    assert env == {"FETCH_CONFIG": "token=${FETCH_TOKEN}", "FETCH_TOKEN": "tok"}


def test_split_tool_command_returns_executable_and_args():
    tool = ToolSpec(name="fetch", command=["uvx", "mcp-server-fetch", "--flag"], env=[])
    assert support.split_tool_command(tool, example='["uvx", "mcp-server-fetch"]') == (
        "uvx",
        ["mcp-server-fetch", "--flag"],
    )


def test_split_tool_command_rejects_empty_command():
    tool = ToolSpec.model_construct(name="broken", command=[], env=[])
    with pytest.raises(support.AgentBuildError, match="empty command"):
        support.split_tool_command(tool, example='["uvx", "mcp-server-fetch"]')


@pytest.mark.parametrize(
    ("raw", "want"),
    [(None, 120.0), ("", 120.0), ("30", 30.0), ("2.5", 2.5), ("0", 120.0), ("bad", 120.0)],
)
def test_positive_float_env(raw: str | None, want: float):
    env = {} if raw is None else {"AGENTKIT_MCP_TIMEOUT": raw}
    with mock.patch.dict(os.environ, env, clear=True):
        assert support.positive_float_env(default=120.0) == want


@pytest.mark.parametrize(
    ("raw", "want"),
    [(None, None), ("", None), ("30", 30), ("2.9", 2), ("0", None), ("bad", None)],
)
def test_positive_int_env(raw: str | None, want: int | None):
    env = {} if raw is None else {"AGENTKIT_MCP_TIMEOUT": raw}
    with mock.patch.dict(os.environ, env, clear=True):
        assert support.positive_int_env(default=None) == want


def test_upstream_status_code_walks_wrapped_exception_chain():
    class BadRequest(Exception):
        status_code = 400

    class Wrapped(Exception):
        def __init__(self, msg: str, inner: Exception | None = None):
            super().__init__(msg)
            self.inner_exception = inner

    assert support.upstream_status_code(BadRequest()) == 400

    try:
        try:
            raise BadRequest("upstream 400")
        except BadRequest as ex:
            raise Wrapped("wrapped", inner=ex) from ex
    except Wrapped as exc:
        assert support.upstream_status_code(exc) == 400

    assert support.upstream_status_code(ValueError("x")) == 502

    class Teapot(Exception):
        status_code = 200

    assert support.upstream_status_code(Teapot()) == 502


def test_upstream_status_code_handles_cycles():
    exc = RuntimeError("cycle")
    exc.__context__ = exc
    assert support.upstream_status_code(exc) == 502


def test_normalize_agent_run_error_preserves_code_and_status():
    class Unavailable(Exception):
        status_code = 503

    err = support.normalize_agent_run_error(Unavailable("upstream down"))
    assert isinstance(err, AgentRunError)
    assert err.status == 503
    assert err.code == "Unavailable"
    assert "upstream down" in str(err)


def _remote_tool(**overrides) -> ToolSpec:
    data = {
        "name": "toolbox",
        "type": "mcp",
        "transport": "streamable-http",
        "urlEnv": "TOOLBOX_ENDPOINT",
    }
    data.update(overrides)
    return ToolSpec.model_validate(data)


def test_resolve_tool_url_uses_declared_env_name_only():
    tool = _remote_tool()
    with mock.patch.dict(os.environ, {"TOOLBOX_ENDPOINT": "http://127.0.0.1:8000/mcp"}, clear=True):
        assert support.resolve_tool_url(tool) == "http://127.0.0.1:8000/mcp"


def test_resolve_tool_url_missing_env_fails_secret_free():
    tool = _remote_tool()
    with mock.patch.dict(os.environ, {"OTHER_SECRET": "do-not-mention"}, clear=True):
        with pytest.raises(support.AgentBuildError) as exc:
            support.resolve_tool_url(tool)
    msg = str(exc.value)
    assert "TOOLBOX_ENDPOINT" in msg
    assert "do-not-mention" not in msg


def test_resolve_tool_headers_supports_static_env_and_bearer_auth():
    tool = _remote_tool(
        headers=[
            {"name": "Foundry-Features", "value": "Toolboxes=V1Preview"},
            {"name": "X-Trace", "valueEnv": "TRACE_HEADER"},
        ],
        auth={"type": "bearer", "tokenEnv": "TOOLBOX_TOKEN"},
    )
    with mock.patch.dict(os.environ, {"TRACE_HEADER": "trace", "TOOLBOX_TOKEN": "tok"}, clear=True):
        headers = support.resolve_tool_headers(tool)
    assert headers == {
        "Foundry-Features": "Toolboxes=V1Preview",
        "X-Trace": "trace",
        "Authorization": "Bearer tok",
    }


def test_resolve_tool_headers_missing_value_env_fails_secret_free():
    tool = _remote_tool(headers=[{"name": "X-Trace", "valueEnv": "TRACE_HEADER"}])
    with mock.patch.dict(os.environ, {"OTHER_SECRET": "do-not-mention"}, clear=True):
        with pytest.raises(support.AgentBuildError) as exc:
            support.resolve_tool_headers(tool)
    msg = str(exc.value)
    assert "TRACE_HEADER" in msg
    assert "do-not-mention" not in msg


def test_resolve_workload_identity_token_uses_explicit_runtime_hook():
    tool = _remote_tool(auth={"type": "workload-identity-token", "audience": "https://ai.azure.com/.default"})
    with mock.patch.dict(os.environ, {"AGENTKIT_WORKLOAD_IDENTITY_TOKEN": "workload-token"}, clear=True):
        assert support.resolve_tool_headers(tool)["Authorization"] == "Bearer workload-token"


def test_resolve_tool_headers_can_defer_workload_identity_token():
    tool = _remote_tool(auth={"type": "workload-identity-token", "audience": "https://ai.azure.com/.default"})
    with mock.patch.dict(os.environ, {"AGENTKIT_WORKLOAD_IDENTITY_TOKEN": "workload-token"}, clear=True):
        assert support.resolve_tool_headers(tool, include_workload_identity=False) == {}
        assert support.resolve_tool_headers(tool)["Authorization"] == "Bearer workload-token"


def _api_key_spec() -> AgentSpec:
    return AgentSpec.model_validate(
        {
            "abiVersion": "v0",
            "metadata": {"name": "env-test"},
            "model": {
                "provider": "openai-compatible",
                "baseURL": "https://api.openai.com/v1",
                "name": "gpt-4o-mini",
                "apiKeyEnv": "MODEL_TOKEN",
            },
            "instructions": "Be helpful.",
            "tools": [],
            "expose": {"openai": True, "port": 8080},
        }
    )


def test_resolve_api_key_prefers_per_run_env_without_mutating_process_env():
    with mock.patch.dict(os.environ, {"MODEL_TOKEN": "process-token"}, clear=True):
        assert support.resolve_api_key(_api_key_spec(), env={"MODEL_TOKEN": "turn-token"}) == "turn-token"
        assert os.environ["MODEL_TOKEN"] == "process-token"


def test_resolve_tool_headers_prefers_per_run_env_and_keeps_errors_secret_free():
    tool = _remote_tool(
        headers=[{"name": "X-Trace", "valueEnv": "TRACE_HEADER"}],
        auth={"type": "bearer", "tokenEnv": "TOOLBOX_TOKEN"},
    )
    with mock.patch.dict(os.environ, {"TRACE_HEADER": "process", "TOOLBOX_TOKEN": "process-secret"}, clear=True):
        headers = support.resolve_tool_headers(
            tool,
            env={"TRACE_HEADER": "turn-trace", "TOOLBOX_TOKEN": "turn-secret"},
        )
        assert headers == {"X-Trace": "turn-trace", "Authorization": "Bearer turn-secret"}
    with mock.patch.dict(os.environ, {"OTHER_SECRET": "do-not-mention"}, clear=True):
        with pytest.raises(support.AgentBuildError) as exc:
            support.resolve_tool_headers(tool, env={"TRACE_HEADER": "visible"})
    msg = str(exc.value)
    assert "TOOLBOX_TOKEN" in msg
    assert "do-not-mention" not in msg


def _fake_azure_identity_module(factory_type: type) -> dict[str, ModuleType]:
    azure = ModuleType("azure")
    azure.__path__ = []  # type: ignore[attr-defined]
    identity = ModuleType("azure.identity")
    setattr(identity, "DefaultAzureCredential", factory_type)
    azure.identity = identity  # type: ignore[attr-defined]
    return {"azure": azure, "azure.identity": identity}


def test_default_azure_credential_fallback_closes_after_success():
    instances = []

    class FakeCredential:
        def __init__(self) -> None:
            self.closed = False
            instances.append(self)

        def get_token(self, audience: str):
            assert audience == "https://ai.azure.com/.default"
            result = SimpleNamespace()
            setattr(result, "token", "azure-token")
            return result

        def close(self) -> None:
            self.closed = True

    with (
        mock.patch.dict(os.environ, {}, clear=True),
        mock.patch.dict(sys.modules, _fake_azure_identity_module(FakeCredential)),
    ):
        resolved = support.resolve_workload_identity_token("https://ai.azure.com/.default")

    assert resolved == "azure-token"
    assert len(instances) == 1
    assert instances[0].closed is True


def test_default_azure_credential_fallback_closes_after_failure_without_masking_token_error():
    instances = []
    acquisition_error = RuntimeError("token unavailable")

    class FakeCredential:
        def __init__(self) -> None:
            self.closed = False
            instances.append(self)

        def get_token(self, audience: str):
            raise acquisition_error

        def close(self) -> None:
            self.closed = True
            raise RuntimeError("close failed")

    with (
        mock.patch.dict(os.environ, {}, clear=True),
        mock.patch.dict(sys.modules, _fake_azure_identity_module(FakeCredential)),
    ):
        with pytest.raises(support.AgentBuildError, match="token unavailable") as exc_info:
            support.resolve_workload_identity_token("https://ai.azure.com/.default")

    assert exc_info.value.__cause__ is acquisition_error
    assert len(instances) == 1
    assert instances[0].closed is True
