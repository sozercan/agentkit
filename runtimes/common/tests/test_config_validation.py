from __future__ import annotations

from copy import deepcopy

import pytest
import yaml

from agentkit_serve_common.config import ConfigError, load


_BASE_SPEC = {
    "abiVersion": "v0",
    "metadata": {"name": "test-agent"},
    "model": {
        "provider": "openai-compatible",
        "baseURL": "https://api.openai.com/v1",
        "name": "gpt-4o-mini",
        "apiKeyEnv": "OPENAI_API_KEY",
    },
    "instructions": "Be helpful.",
    "tools": [
        {
            "name": "fetch",
            "command": ["uvx", "mcp-server-fetch"],
            "env": ["FETCH_TIMEOUT"],
        }
    ],
    "expose": {"openai": True, "port": 8080},
}


def _write_spec(tmp_path, spec: dict) -> str:
    path = tmp_path / "agent.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return str(path)


def _invalid_message(tmp_path, mutate) -> str:
    spec = deepcopy(_BASE_SPEC)
    mutate(spec)
    with pytest.raises(ConfigError) as exc:
        load(_write_spec(tmp_path, spec))
    return str(exc.value)


def test_load_rejects_unsupported_provider(tmp_path):
    msg = _invalid_message(tmp_path, lambda spec: spec["model"].update(provider="anthropic"))
    assert "model.provider" in msg
    assert "openai-compatible" in msg


def test_load_rejects_invalid_api_key_env_name(tmp_path):
    msg = _invalid_message(tmp_path, lambda spec: spec["model"].update(apiKeyEnv="sk-secret"))
    assert "model.apiKeyEnv" in msg
    assert "[A-Z0-9_]+" in msg


def test_load_rejects_tool_without_command(tmp_path):
    msg = _invalid_message(tmp_path, lambda spec: spec["tools"][0].update(command=[]))
    assert "tools.0.command" in msg
    assert "at least 1 item" in msg


def test_load_rejects_empty_tool_command_entry(tmp_path):
    msg = _invalid_message(tmp_path, lambda spec: spec["tools"][0].update(command=[""]))
    assert "tools.0.command" in msg
    assert "non-empty" in msg


def test_load_rejects_invalid_tool_env_name(tmp_path):
    msg = _invalid_message(tmp_path, lambda spec: spec["tools"][0].update(env=["FETCH-TOKEN"]))
    assert "tools.0.env" in msg
    assert "[A-Z0-9_]+" in msg


def test_load_rejects_expose_openai_false(tmp_path):
    msg = _invalid_message(tmp_path, lambda spec: spec["expose"].update(openai=False))
    assert "expose.openai" in msg
    assert "must be true" in msg


@pytest.mark.parametrize("port", [0, 65536])
def test_load_rejects_bad_port(tmp_path, port: int):
    msg = _invalid_message(tmp_path, lambda spec: spec["expose"].update(port=port))
    assert "expose.port" in msg
    assert "between 1 and 65535" in msg


def test_load_rejects_missing_expose(tmp_path):
    msg = _invalid_message(tmp_path, lambda spec: spec.pop("expose"))
    assert "expose" in msg
    assert "Field required" in msg
