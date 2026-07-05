from __future__ import annotations

import os
from types import TracebackType

import pytest

from agentkit_serve_common import cli
from agentkit_serve_common.config import AgentSpec
from agentkit_serve_common.runtime import RunResult, RuntimeSession


def _spec(port: int = 8080) -> AgentSpec:
    return AgentSpec.model_validate(
        {
            "abiVersion": "v0",
            "metadata": {"name": "cli-test"},
            "model": {
                "provider": "openai-compatible",
                "baseURL": "https://api.openai.com/v1",
                "name": "gpt-4o-mini",
            },
            "instructions": "Be helpful.",
            "tools": [],
            "expose": {"openai": True, "port": port},
        }
    )


class Runtime:
    async def __aenter__(self) -> RuntimeSession:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        return None

    async def run(self, request):  # noqa: ANN001
        return RunResult(text="ok")


class Factory:
    def build_runtime(self, spec: AgentSpec) -> RuntimeSession:
        return Runtime()


def test_cli_protocol_flag_selects_foundry_and_default_foundry_port(monkeypatch):
    captured = {}
    monkeypatch.setattr(cli, "load_or_exit", lambda path: _spec())
    monkeypatch.setattr(cli.uvicorn, "run", lambda app, **kwargs: captured.update({"app": app, **kwargs}))
    monkeypatch.delenv("AGENTKIT_PROTOCOL", raising=False)
    monkeypatch.delenv("AGENTKIT_PORT", raising=False)
    monkeypatch.delenv("AGENTKIT_AUTH_TOKEN", raising=False)

    cli.run(Factory(), ["--config", "agent.yaml", "--protocol", "foundry"])

    assert captured["port"] == 8088
    assert captured["host"] == "127.0.0.1"
    assert any(getattr(route, "path", None) == "/readiness" for route in captured["app"].routes)


def test_cli_protocol_env_selects_orka_and_requires_auth_token(monkeypatch):
    monkeypatch.setattr(cli, "load", lambda path: _spec())
    monkeypatch.setenv("AGENTKIT_PROTOCOL", "orka")
    monkeypatch.delenv("AGENTKIT_AUTH_TOKEN", raising=False)

    with pytest.raises(SystemExit) as exc:
        cli.run(Factory(), ["--config", "agent.yaml"])

    assert exc.value.code == 2


def test_cli_orka_with_auth_uses_orka_routes_and_port_override(monkeypatch):
    captured = {}
    monkeypatch.setattr(cli, "load", lambda path: _spec())
    monkeypatch.setattr(cli.uvicorn, "run", lambda app, **kwargs: captured.update({"app": app, **kwargs}))
    monkeypatch.setenv("AGENTKIT_AUTH_TOKEN", "token")
    monkeypatch.setenv("AGENTKIT_PORT", "9999")
    monkeypatch.delenv("AGENTKIT_PROTOCOL", raising=False)

    cli.run(Factory(), ["--config", "agent.yaml", "--protocol", "orka"])

    assert captured["port"] == 9999
    assert any(getattr(route, "path", None) == "/v1/capabilities" for route in captured["app"].routes)


def _spec_with_required_env() -> AgentSpec:
    return AgentSpec.model_validate(
        {
            "abiVersion": "v0",
            "metadata": {"name": "cli-test"},
            "model": {
                "provider": "openai-compatible",
                "baseURL": "https://api.openai.com/v1",
                "name": "gpt-4o-mini",
            },
            "instructions": "Be helpful.",
            "tools": [],
            "env": [{"name": "MODEL_TOKEN", "required": True}],
            "expose": {"openai": True, "port": 8080},
        }
    )


def test_cli_orka_skips_startup_required_env_validation_for_turn_env(monkeypatch):
    captured = {}
    monkeypatch.setattr(cli, "load", lambda path: _spec_with_required_env())
    monkeypatch.setattr(cli, "load_or_exit", lambda path: (_ for _ in ()).throw(AssertionError("load_or_exit should not run for orka")))
    monkeypatch.setattr(cli.uvicorn, "run", lambda app, **kwargs: captured.update({"app": app, **kwargs}))
    monkeypatch.setenv("AGENTKIT_AUTH_TOKEN", "token")
    monkeypatch.delenv("MODEL_TOKEN", raising=False)

    cli.run(Factory(), ["--config", "agent.yaml", "--protocol", "orka"])

    assert captured["port"] == 8080
    assert any(getattr(route, "path", None) == "/v1/turns" for route in captured["app"].routes)


def test_cli_protocol_flag_sets_agentkit_protocol_for_adapter_runtime(monkeypatch):
    captured = {}
    monkeypatch.setattr(cli, "load", lambda path: _spec())
    monkeypatch.setattr(cli.uvicorn, "run", lambda app, **kwargs: captured.update({"app": app, **kwargs}))
    monkeypatch.setenv("AGENTKIT_AUTH_TOKEN", "token")
    monkeypatch.delenv("AGENTKIT_PROTOCOL", raising=False)

    cli.run(Factory(), ["--config", "agent.yaml", "--protocol", "orka"])

    assert captured["port"] == 8080
    assert os.environ["AGENTKIT_PROTOCOL"] == "orka"
