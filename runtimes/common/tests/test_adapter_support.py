from __future__ import annotations

import os
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
