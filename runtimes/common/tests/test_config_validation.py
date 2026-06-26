from __future__ import annotations

from copy import deepcopy

import pytest
import yaml

from agentkit_serve_common.config import ConfigError, load, load_or_exit, validate_required_env


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
    "env": [],
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


def test_load_rejects_secret_like_api_key_env_name(tmp_path):
    msg = _invalid_message(tmp_path, lambda spec: spec["model"].update(apiKeyEnv="AKIAEXAMPLE"))
    assert "model.apiKeyEnv" in msg
    assert "secret value" in msg


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


def test_load_accepts_env_declarations(tmp_path):
    spec_dict = deepcopy(_BASE_SPEC)
    spec_dict["env"] = [
        {"name": "REQUIRED_FOO", "required": True},
        {"name": "OPTIONAL_BAR"},
    ]

    spec = load(_write_spec(tmp_path, spec_dict))

    assert [entry.name for entry in spec.env] == ["REQUIRED_FOO", "OPTIONAL_BAR"]
    assert spec.env[0].required is True
    assert spec.env[1].required is False


def test_load_rejects_invalid_env_declaration_name(tmp_path):
    msg = _invalid_message(tmp_path, lambda spec: spec.update(env=[{"name": "required-foo"}]))
    assert "env.0.name" in msg
    assert "[A-Z0-9_]+" in msg


def test_load_rejects_duplicate_env_declarations(tmp_path):
    msg = _invalid_message(
        tmp_path,
        lambda spec: spec.update(env=[{"name": "REQUIRED_FOO"}, {"name": "REQUIRED_FOO"}]),
    )
    assert "env" in msg
    assert "duplicate env var declarations" in msg


def test_validate_required_env_is_secret_free(tmp_path, monkeypatch):
    spec_dict = deepcopy(_BASE_SPEC)
    spec_dict["env"] = [
        {"name": "REQUIRED_FOO", "required": True},
        {"name": "OPTIONAL_BAR"},
    ]
    spec = load(_write_spec(tmp_path, spec_dict))

    monkeypatch.delenv("REQUIRED_FOO", raising=False)
    with pytest.raises(ConfigError) as exc:
        validate_required_env(spec)
    msg = str(exc.value)
    assert "REQUIRED_FOO" in msg
    assert "OPTIONAL_BAR" not in msg
    assert "secret" not in msg

    monkeypatch.setenv("REQUIRED_FOO", "super-secret-value")
    validate_required_env(spec)


def test_load_or_exit_validates_required_env(tmp_path, monkeypatch, capsys):
    spec_dict = deepcopy(_BASE_SPEC)
    spec_dict["env"] = [{"name": "REQUIRED_FOO", "required": True}]
    path = _write_spec(tmp_path, spec_dict)
    monkeypatch.delenv("REQUIRED_FOO", raising=False)

    with pytest.raises(SystemExit) as exc:
        load_or_exit(path)

    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "REQUIRED_FOO" in captured.err
