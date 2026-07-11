from __future__ import annotations

import math
from copy import deepcopy

import pytest
import yaml

from _brokered_description_cases import HARMLESS_BROKERED_DESCRIPTIONS, UNSAFE_BROKERED_DESCRIPTIONS
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


def test_load_rejects_tool_without_command_or_url(tmp_path):
    msg = _invalid_message(tmp_path, lambda spec: spec["tools"][0].update(command=[]))
    assert "tools.0" in msg
    assert "command or urlEnv" in msg


def test_load_rejects_empty_tool_command_entry(tmp_path):
    msg = _invalid_message(tmp_path, lambda spec: spec["tools"][0].update(command=[""]))
    assert "tools.0.command" in msg
    assert "non-empty" in msg


def test_load_rejects_invalid_tool_env_name(tmp_path):
    msg = _invalid_message(tmp_path, lambda spec: spec["tools"][0].update(env=["FETCH-TOKEN"]))
    assert "tools.0.env" in msg
    assert "[A-Z0-9_]+" in msg


def test_load_rejects_duplicate_direct_tool_names(tmp_path):
    msg = _invalid_message(
        tmp_path,
        lambda spec: spec["tools"].append(
            {"name": "fetch", "command": ["uvx", "another-mcp-server"]}
        ),
    )
    assert "tools" in msg
    assert "duplicate tool name" in msg
    assert "fetch" in msg


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


def test_load_accepts_remote_mcp_tool(tmp_path):
    spec_dict = deepcopy(_BASE_SPEC)
    spec_dict["tools"] = [
        {
            "name": "toolbox",
            "type": "mcp",
            "transport": "streamable-http",
            "urlEnv": "TOOLBOX_ENDPOINT",
            "headers": [
                {"name": "Foundry-Features", "value": "Toolboxes=V1Preview"},
                {"name": "X-Trace", "valueEnv": "TOOLBOX_TRACE"},
            ],
            "auth": {"type": "bearer", "tokenEnv": "TOOLBOX_TOKEN"},
        }
    ]

    spec = load(_write_spec(tmp_path, spec_dict))

    tool = spec.tools[0]
    assert tool.url_env == "TOOLBOX_ENDPOINT"
    assert tool.headers[0].value == "Toolboxes=V1Preview"
    assert tool.headers[1].value_env == "TOOLBOX_TRACE"
    assert tool.auth is not None
    assert tool.auth.token_env == "TOOLBOX_TOKEN"


def test_load_rejects_invalid_remote_mcp_tool(tmp_path):
    spec_dict = deepcopy(_BASE_SPEC)
    spec_dict["tools"] = [
        {
            "name": "toolbox",
            "type": "mcp",
            "transport": "streamable-http",
            "urlEnv": "TOOLBOX_ENDPOINT",
            "headers": [{"name": "Authorization", "value": "Bearer nope"}],
        }
    ]

    with pytest.raises(ConfigError) as exc:
        load(_write_spec(tmp_path, spec_dict))

    msg = str(exc.value)
    assert "static credential" in msg


def test_load_accepts_context_shape(tmp_path):
    spec_dict = deepcopy(_BASE_SPEC)
    spec_dict["context"] = {
        "providers": [
            {
                "name": "knowledge",
                "type": "search",
                "endpointEnv": "SEARCH_ENDPOINT",
                "indexEnv": "SEARCH_INDEX",
            }
        ]
    }
    spec = load(_write_spec(tmp_path, spec_dict))

    assert spec.context.providers[0].endpoint_env == "SEARCH_ENDPOINT"


@pytest.mark.parametrize("header", ["X-API-Key", "Cookie"])
def test_load_rejects_static_credential_header_names(tmp_path, header: str):
    spec_dict = deepcopy(_BASE_SPEC)
    spec_dict["tools"] = [
        {
            "name": "toolbox",
            "type": "mcp",
            "transport": "streamable-http",
            "urlEnv": "TOOLBOX_ENDPOINT",
            "headers": [{"name": header, "value": "not-secret-looking"}],
        }
    ]

    with pytest.raises(ConfigError) as exc:
        load(_write_spec(tmp_path, spec_dict))

    assert "static credential" in str(exc.value)


def test_load_rejects_authorization_value_env_plus_auth(tmp_path):
    spec_dict = deepcopy(_BASE_SPEC)
    spec_dict["tools"] = [
        {
            "name": "toolbox",
            "type": "mcp",
            "transport": "streamable-http",
            "urlEnv": "TOOLBOX_ENDPOINT",
            "headers": [{"name": "Authorization", "valueEnv": "AUTH_HEADER"}],
            "auth": {"type": "bearer", "tokenEnv": "TOOLBOX_TOKEN"},
        }
    ]

    with pytest.raises(ConfigError) as exc:
        load(_write_spec(tmp_path, spec_dict))

    msg = str(exc.value)
    assert "Authorization" in msg
    assert "auth" in msg


def test_load_accepts_model_workload_identity_auth(tmp_path):
    spec_dict = deepcopy(_BASE_SPEC)
    spec_dict["model"].pop("apiKeyEnv", None)
    spec_dict["model"]["auth"] = {
        "type": "workload-identity-token",
        "audience": "https://ai.azure.com/.default",
    }

    spec = load(_write_spec(tmp_path, spec_dict))

    assert spec.model.auth is not None
    assert spec.model.auth.type == "workload-identity-token"
    assert spec.model.auth.audience == "https://ai.azure.com/.default"


def test_load_rejects_bearer_auth_for_context_provider(tmp_path):
    spec_dict = deepcopy(_BASE_SPEC)
    spec_dict["context"] = {
        "providers": [
            {
                "name": "knowledge",
                "type": "search",
                "endpointEnv": "SEARCH_ENDPOINT",
                "indexEnv": "SEARCH_INDEX",
                "auth": {"type": "bearer", "tokenEnv": "SEARCH_TOKEN"},
            }
        ]
    }

    with pytest.raises(ConfigError) as exc:
        load(_write_spec(tmp_path, spec_dict))

    assert "context providers do not support bearer auth" in str(exc.value)


def test_load_rejects_relative_filesystem_skills_path(tmp_path):
    spec_dict = deepcopy(_BASE_SPEC)
    spec_dict["context"] = {
        "providers": [{"type": "skills", "source": "filesystem", "path": "./skills"}]
    }

    with pytest.raises(ConfigError) as exc:
        load(_write_spec(tmp_path, spec_dict))

    assert "/agent/skills" in str(exc.value)


def test_load_rejects_mcp_skills_unknown_tool_ref(tmp_path):
    spec_dict = deepcopy(_BASE_SPEC)
    spec_dict["tools"] = [
        {
            "name": "toolbox",
            "type": "mcp",
            "transport": "streamable-http",
            "urlEnv": "TOOLBOX_ENDPOINT",
        }
    ]
    spec_dict["context"] = {
        "providers": [{"type": "skills", "source": "mcp", "toolRef": "missing"}]
    }

    with pytest.raises(ConfigError) as exc:
        load(_write_spec(tmp_path, spec_dict))

    assert "references unknown tool" in str(exc.value)


def test_load_rejects_escaped_filesystem_skills_path(tmp_path):
    spec_dict = deepcopy(_BASE_SPEC)
    spec_dict["context"] = {
        "providers": [{"type": "skills", "source": "filesystem", "path": "/agent/skills/../secrets"}]
    }

    with pytest.raises(ConfigError) as exc:
        load(_write_spec(tmp_path, spec_dict))

    assert "/agent/skills" in str(exc.value)


def test_load_rejects_model_bearer_auth(tmp_path):
    spec_dict = deepcopy(_BASE_SPEC)
    spec_dict["model"].pop("apiKeyEnv", None)
    spec_dict["model"]["auth"] = {"type": "bearer", "tokenEnv": "MODEL_TOKEN"}

    with pytest.raises(ConfigError) as exc:
        load(_write_spec(tmp_path, spec_dict))

    assert "model.auth supports only workload-identity-token" in str(exc.value)


def test_load_rejects_unknown_skills_source(tmp_path):
    spec_dict = deepcopy(_BASE_SPEC)
    spec_dict["context"] = {
        "providers": [{"type": "skills", "source": "filesytem", "path": "/agent/skills"}]
    }

    with pytest.raises(ConfigError) as exc:
        load(_write_spec(tmp_path, spec_dict))

    assert "skills context source must be filesystem or mcp" in str(exc.value)


def test_load_rejects_unknown_context_provider_type(tmp_path):
    spec_dict = deepcopy(_BASE_SPEC)
    spec_dict["context"] = {"providers": [{"type": "serach"}]}

    with pytest.raises(ConfigError) as exc:
        load(_write_spec(tmp_path, spec_dict))

    assert "context provider type must be search, skills, or memory" in str(exc.value)


def test_load_rejects_search_context_missing_env_fields(tmp_path):
    spec_dict = deepcopy(_BASE_SPEC)
    spec_dict["context"] = {"providers": [{"type": "search", "endpointEnv": "SEARCH_ENDPOINT"}]}

    with pytest.raises(ConfigError) as exc:
        load(_write_spec(tmp_path, spec_dict))

    assert "search context providers require indexEnv" in str(exc.value)


def test_load_rejects_auth_on_skills_context_provider(tmp_path):
    spec_dict = deepcopy(_BASE_SPEC)
    spec_dict["context"] = {
        "providers": [
            {
                "type": "skills",
                "source": "filesystem",
                "path": "/agent/skills",
                "auth": {"type": "workload-identity-token", "audience": "https://ai.azure.com/.default"},
            }
        ]
    }

    with pytest.raises(ConfigError) as exc:
        load(_write_spec(tmp_path, spec_dict))

    assert "skills context providers must not set auth" in str(exc.value)


def test_load_rejects_unsupported_tool_approval_policy(tmp_path):
    spec_dict = deepcopy(_BASE_SPEC)
    spec_dict["tools"] = [{"name": "fetch", "command": ["uvx", "mcp-server-fetch"], "approval": "always"}]

    with pytest.raises(ConfigError) as exc:
        load(_write_spec(tmp_path, spec_dict))

    assert "tool approval policies are not supported" in str(exc.value)



def test_load_rejects_unsupported_log_observability(tmp_path):
    spec_dict = deepcopy(_BASE_SPEC)
    spec_dict["observability"] = {"logs": {"levelEnv": "LOG_LEVEL"}}

    with pytest.raises(ConfigError) as exc:
        load(_write_spec(tmp_path, spec_dict))

    assert "observability.logs.levelEnv is not supported" in str(exc.value)



def test_load_rejects_unsupported_otel_observability(tmp_path):
    spec_dict = deepcopy(_BASE_SPEC)
    spec_dict["observability"] = {"otel": {"endpointEnv": "OTEL_EXPORTER_OTLP_ENDPOINT"}}

    with pytest.raises(ConfigError) as exc:
        load(_write_spec(tmp_path, spec_dict))

    assert "observability.otel.endpointEnv is not supported" in str(exc.value)


def test_load_accepts_static_brokered_tools_with_matching_digest(tmp_path):
    from agentkit_serve_common.config import brokered_tool_schema_digest

    parameters = {"type": "object", "properties": {"site": {"type": "string"}}, "required": ["site"]}
    digest = brokered_tool_schema_digest(
        name="check-network-telemetry",
        description="Read sanitized optical telemetry.",
        brokered_class="read",
        parameters=parameters,
    )
    spec_dict = deepcopy(_BASE_SPEC)
    spec_dict["tools"] = []
    spec_dict["brokeredTools"] = [
        {
            "name": "check-network-telemetry",
            "description": "Read sanitized optical telemetry.",
            "brokeredClass": "read",
            "parameters": parameters,
            "schemaDigest": digest,
        }
    ]

    spec = load(_write_spec(tmp_path, spec_dict))

    assert spec.brokered_tools[0].name == "check-network-telemetry"
    assert spec.brokered_tools[0].brokered_class == "read"
    assert spec.brokered_tools[0].schema_digest == digest


@pytest.mark.parametrize(
    "description",
    HARMLESS_BROKERED_DESCRIPTIONS,
)
def test_load_accepts_harmless_brokered_tool_descriptions(tmp_path, description: str):
    spec_dict = deepcopy(_BASE_SPEC)
    spec_dict.update(
        tools=[],
        brokeredTools=[
            {
                "name": "safe_lookup",
                "description": description,
                "brokeredClass": "read",
                "parameters": {"type": "object"},
            }
        ],
    )

    spec = load(_write_spec(tmp_path, spec_dict))

    assert spec.brokered_tools[0].description == description


@pytest.mark.parametrize(
    "description",
    UNSAFE_BROKERED_DESCRIPTIONS,
)
def test_load_rejects_unsafe_brokered_tool_descriptions(tmp_path, description: str):
    msg = _invalid_message(
        tmp_path,
        lambda spec: spec.update(
            tools=[],
            brokeredTools=[
                {
                    "name": "safe_lookup",
                    "description": description,
                    "brokeredClass": "read",
                    "parameters": {"type": "object"},
                }
            ],
        ),
    )
    assert "brokeredTools.0.description" in msg
    assert "secret-like" in msg or "URLs" in msg


@pytest.mark.parametrize("field", ["url", "headers", "secretRef", "auth", "token"])
def test_load_rejects_unsafe_brokered_tool_fields(tmp_path, field: str):
    def mutate(spec):
        spec["tools"] = []
        spec["brokeredTools"] = [
            {
                "name": "safe_lookup",
                "description": "safe schema",
                "brokeredClass": "read",
                "parameters": {"type": "object"},
                field: "should-not-cross",
            }
        ]

    msg = _invalid_message(tmp_path, mutate)
    assert "brokeredTools.0" in msg
    assert "unsafe" in msg or "Extra inputs" in msg


@pytest.mark.parametrize("unsafe_name", ["token", "authHeader", "authorizationHeader", "httpHeaders", "accessKey", "clientSecretValue", "tokenValue", "apiSecretKey", "cookie", "subscriptionKey", "xFunctionsKey"])
def test_load_rejects_unsafe_brokered_parameter_names(tmp_path, unsafe_name: str):
    msg = _invalid_message(
        tmp_path,
        lambda spec: spec.update(
            tools=[],
            brokeredTools=[
                {
                    "name": "safe_lookup",
                    "description": "safe schema",
                    "brokeredClass": "read",
                    "parameters": {"type": "object", "properties": {unsafe_name: {"type": "string"}}},
                }
            ],
        ),
    )
    assert "brokeredTools.0.parameters" in msg
    assert "not safe" in msg


@pytest.mark.parametrize(
    "value",
    [
        "see https://internal-tool",
        "Bearer abc",
        "authorization header",
        "token=abc123",
        "Cookie: session=abc",
        "x-api-key: abc",
        "api_key=abc",
        "x_api_key: abc",
        "format password",
        "enter passphrase",
    ],
)
def test_load_rejects_unsafe_brokered_schema_string_values(tmp_path, value: str):
    msg = _invalid_message(
        tmp_path,
        lambda spec: spec.update(
            tools=[],
            brokeredTools=[
                {
                    "name": "safe_lookup",
                    "description": "safe schema",
                    "brokeredClass": "read",
                    "parameters": {"type": "object", "properties": {"site": {"type": "string", "description": value}}},
                }
            ],
        ),
    )
    assert "brokeredTools.0.parameters" in msg
    assert "secret-like" in msg or "URL" in msg


def test_load_rejects_private_key_brokered_parameter_names_and_strings(tmp_path):
    name_msg = _invalid_message(
        tmp_path,
        lambda spec: spec.update(
            tools=[],
            brokeredTools=[
                {
                    "name": "safe_lookup",
                    "description": "safe schema",
                    "brokeredClass": "read",
                    "parameters": {"type": "object", "properties": {"privateKey": {"type": "string"}}},
                }
            ],
        ),
    )
    text_msg = _invalid_message(
        tmp_path,
        lambda spec: spec.update(
            tools=[],
            brokeredTools=[
                {
                    "name": "safe_lookup",
                    "description": "safe schema",
                    "brokeredClass": "read",
                    "parameters": {"type": "object", "properties": {"site": {"type": "string", "default": "BEGIN PRIVATE KEY"}}},
                }
            ],
        ),
    )
    assert "not safe" in name_msg
    assert "secret-like" in text_msg or "URL" in text_msg


@pytest.mark.parametrize("field", ["authentication", "authConfig", "clientSecret", "dbPassword", "passphrase", "pwd", "apiKey", "api-key", "baseUrl", "callbackURL", "apiEndpoint", "sessionCookie", "cookies"])
def test_load_rejects_common_credential_brokered_parameter_names(tmp_path, field: str):
    msg = _invalid_message(
        tmp_path,
        lambda spec: spec.update(
            tools=[],
            brokeredTools=[
                {
                    "name": "safe_lookup",
                    "description": "safe schema",
                    "brokeredClass": "read",
                    "parameters": {"type": "object", "properties": {field: {"type": "string"}}},
                }
            ],
        ),
    )
    assert "brokeredTools.0.parameters" in msg
    assert "not safe" in msg


@pytest.mark.parametrize("field", ["clientSecret", "dbPassword", "apiKey", "api-key"])
def test_load_rejects_common_credential_brokered_required_names(tmp_path, field: str):
    msg = _invalid_message(
        tmp_path,
        lambda spec: spec.update(
            tools=[],
            brokeredTools=[
                {
                    "name": "safe_lookup",
                    "description": "safe schema",
                    "brokeredClass": "read",
                    "parameters": {"type": "object", "required": [field]},
                }
            ],
        ),
    )
    assert "brokeredTools.0.parameters" in msg
    assert "not safe" in msg


def test_load_rejects_secret_literals_inside_brokered_schema_strings(tmp_path):
    msg = _invalid_message(
        tmp_path,
        lambda spec: spec.update(
            tools=[],
            brokeredTools=[
                {
                    "name": "safe_lookup",
                    "description": "safe schema",
                    "brokeredClass": "read",
                    "parameters": {"type": "object", "properties": {"site": {"type": "string", "default": "sk-not-a-real-secret"}}},
                }
            ],
        ),
    )
    assert "secret-like material" in msg


def test_load_rejects_unsafe_text_literals_inside_brokered_schema_strings(tmp_path):
    for unsafe_value in ["Bearer abc", "Bearer: abc", "Bearer=abc", "https://internal.example", "http://tool.default.svc.cluster.local", "tool.default.svc.cluster.local", "example ghp_not_real", "AWS key AKIAEXAMPLE"]:
        msg = _invalid_message(
            tmp_path,
            lambda spec, unsafe_value=unsafe_value: spec.update(
                tools=[],
                brokeredTools=[
                    {
                        "name": "safe_lookup",
                        "description": "safe schema",
                        "brokeredClass": "read",
                        "parameters": {"type": "object", "properties": {"site": {"type": "string", "default": unsafe_value}}},
                    }
                ],
            ),
        )
        assert "URL or secret-like material" in msg


def test_load_rejects_brokered_enum_combined_with_constraints_for_deterministic_synthesis(tmp_path):
    msg = _invalid_message(
        tmp_path,
        lambda spec: spec.update(
            tools=[],
            brokeredTools=[
                {
                    "name": "safe_lookup",
                    "description": "safe schema",
                    "brokeredClass": "read",
                    "parameters": {"type": "object", "properties": {"n": {"type": "integer", "minimum": 2, "enum": [1, 2]}}},
                }
            ],
        ),
    )
    assert "enum/const/default" in msg


def test_load_rejects_brokered_tool_names_over_model_function_limit(tmp_path):
    msg = _invalid_message(
        tmp_path,
        lambda spec: spec.update(
            tools=[],
            brokeredTools=[
                {
                    "name": "a" * 65,
                    "description": "safe schema",
                    "brokeredClass": "read",
                    "parameters": {"type": "object"},
                }
            ],
        ),
    )
    assert "[A-Za-z0-9_-]{1,64}" in msg


def test_load_rejects_unsupported_brokered_json_schema_composition_keywords(tmp_path):
    msg = _invalid_message(
        tmp_path,
        lambda spec: spec.update(
            tools=[],
            brokeredTools=[
                {
                    "name": "safe_lookup",
                    "description": "safe schema",
                    "brokeredClass": "read",
                    "parameters": {"type": "object", "allOf": [{"required": ["site"]}]},
                }
            ],
        ),
    )
    assert "allOf" in msg


def test_load_rejects_unsupported_brokered_schema_pattern(tmp_path):
    msg = _invalid_message(
        tmp_path,
        lambda spec: spec.update(
            tools=[],
            brokeredTools=[
                {
                    "name": "safe_lookup",
                    "description": "safe schema",
                    "brokeredClass": "read",
                    "parameters": {"type": "object", "properties": {"site": {"type": "string", "pattern": "["}}},
                }
            ],
        ),
    )
    assert "pattern" in msg


@pytest.mark.parametrize("bad_parameters", [
    {"type": "object", "properties": None},
    {"type": "object", "properties": {"site": {"type": "string", "enum": None}}},
    {"type": "object", "properties": {"site": {"type": "string", "pattern": None}}},
])
def test_load_rejects_explicit_null_brokered_json_schema_keywords(tmp_path, bad_parameters: dict):
    msg = _invalid_message(
        tmp_path,
        lambda spec: spec.update(
            tools=[],
            brokeredTools=[
                {
                    "name": "safe_lookup",
                    "description": "safe schema",
                    "brokeredClass": "read",
                    "parameters": bad_parameters,
                }
            ],
        ),
    )
    assert "brokeredTools.0.parameters" in msg


@pytest.mark.parametrize("bad_child", [{"type": 123}, {"type": "strnig"}, {"items": "bad"}, {"items": [{"type": "number"}]}, {"multipleOf": 2}, {"uniqueItems": True}, {"minLength": -1}])
def test_load_rejects_malformed_nested_brokered_json_schema(tmp_path, bad_child: dict):
    msg = _invalid_message(
        tmp_path,
        lambda spec: spec.update(
            tools=[],
            brokeredTools=[
                {
                    "name": "safe_lookup",
                    "description": "safe schema",
                    "brokeredClass": "read",
                    "parameters": {"type": "object", "properties": {"site": bad_child}},
                }
            ],
        ),
    )
    assert "brokeredTools.0.parameters" in msg


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "object", "properties": {"site": {"type": None}}},
        {"type": "object", "properties": {"n": {"type": "integer", "default": "1"}}},
        {"type": "object", "properties": {"site": {"type": "string", "enum": [0, "ok"]}}},
    ],
)
def test_load_rejects_invalid_brokered_schema_type_values_and_defaults(tmp_path, schema: dict):
    msg = _invalid_message(
        tmp_path,
        lambda spec: spec.update(
            tools=[],
            brokeredTools=[
                {
                    "name": "safe_lookup",
                    "description": "safe schema",
                    "brokeredClass": "read",
                    "parameters": schema,
                }
            ],
        ),
    )
    assert "brokeredTools.0.parameters" in msg


def test_load_rejects_unknown_brokered_class_and_malformed_schema(tmp_path):
    unknown_class = _invalid_message(
        tmp_path,
        lambda spec: spec.update(
            tools=[],
            brokeredTools=[
                {
                    "name": "safe_lookup",
                    "description": "safe schema",
                    "brokeredClass": "admin",
                    "parameters": {"type": "object"},
                }
            ],
        ),
    )
    bad_schema = _invalid_message(
        tmp_path,
        lambda spec: spec.update(
            tools=[],
            brokeredTools=[
                {
                    "name": "safe_lookup",
                    "description": "safe schema",
                    "brokeredClass": "read",
                    "parameters": {"type": "array"},
                }
            ],
        ),
    )

    assert "brokeredTools.0.brokeredClass" in unknown_class
    assert "brokeredTools.0.parameters" in bad_schema
    assert "type: object" in bad_schema


def test_load_rejects_brokered_schema_digest_mismatch(tmp_path):
    msg = _invalid_message(
        tmp_path,
        lambda spec: spec.update(
            tools=[],
            brokeredTools=[
                {
                    "name": "safe_lookup",
                    "description": "safe schema",
                    "brokeredClass": "read",
                    "parameters": {"type": "object"},
                    "schemaDigest": "sha256:" + "0" * 64,
                }
            ],
        ),
    )
    assert "brokeredTools.0" in msg
    assert "schemaDigest does not match" in msg


def test_load_rejects_owned_and_brokered_tool_name_overlap(tmp_path):
    msg = _invalid_message(
        tmp_path,
        lambda spec: spec.update(
            brokeredTools=[
                {
                    "name": "fetch",
                    "description": "safe schema",
                    "brokeredClass": "read",
                    "parameters": {"type": "object"},
                }
            ]
        ),
    )
    assert "tools and brokeredTools cannot be mixed" in msg


def test_load_preserves_negative_zero_for_brokered_schema_digest(tmp_path):
    from agentkit_serve_common.config import brokered_tool_schema_digest

    parameters = {
        "type": "object",
        "properties": {"n": {"type": "number", "minimum": -0.0}},
    }
    digest = brokered_tool_schema_digest(
        name="negative-zero-tool",
        description="Negative zero schema.",
        brokered_class="read",
        parameters=parameters,
    )
    spec_dict = deepcopy(_BASE_SPEC)
    spec_dict.update(
        tools=[],
        brokeredTools=[
            {
                "name": "negative-zero-tool",
                "description": "Negative zero schema.",
                "brokeredClass": "read",
                "parameters": parameters,
                "schemaDigest": digest,
            }
        ],
    )

    spec = load(_write_spec(tmp_path, spec_dict))

    minimum = spec.brokered_tools[0].parameters["properties"]["n"]["minimum"]
    assert isinstance(minimum, float)
    assert minimum == 0.0
    assert math.copysign(1.0, minimum) == -1.0
    assert spec.brokered_tools[0].schema_digest == digest


def test_brokered_tool_schema_digest_uses_utf8_canonical_json():
    from agentkit_serve_common.config import brokered_tool_schema_digest

    parameters = {
        "type": "object",
        "properties": {"site": {"type": "string", "description": "São & R&D <safe>"}},
        "required": ["site"],
    }

    assert brokered_tool_schema_digest(
        name="check-network-telemetry",
        description="São & R&D <safe>",
        brokered_class="read",
        parameters=parameters,
    ) == "sha256:7066de4e62dd1a6550701772aad901e1efcf5eb81a4f639252f82e6c7be8d4c1"


def test_brokered_tool_schema_digest_normalizes_integer_valued_floats():
    from agentkit_serve_common.config import brokered_tool_schema_digest

    parameters = {"type": "object", "properties": {"retries": {"type": "number", "minimum": 1.0}}}

    assert brokered_tool_schema_digest(
        name="numeric-tool",
        description="Numeric constraints.",
        brokered_class="read",
        parameters=parameters,
    ) == "sha256:7ad9d43791e157981bcd65fd8452c9e71a64064875cc1330ced42d4956bf7d75"


def test_brokered_tool_schema_digest_canonicalizes_numeric_constraints_cross_language():
    from agentkit_serve_common.config import brokered_tool_schema_digest

    parameters = {
        "type": "object",
        "properties": {"n": {"type": "number", "minimum": 1e-6}},
        "required": ["n"],
    }

    assert brokered_tool_schema_digest(
        name="num-tool",
        description="Numeric schema",
        brokered_class="read",
        parameters=parameters,
    ) == "sha256:83bf12180154a21f8ba19049687e24acee9ef430966af67a83721a10bf7eee50"


def test_brokered_tool_schema_digest_canonicalizes_positive_exponent_numbers():
    from agentkit_serve_common.config import brokered_tool_schema_digest

    parameters = {"type": "object", "properties": {"n": {"type": "number", "maximum": 1e20}}}

    assert brokered_tool_schema_digest(
        name="large-num-tool",
        description="Large numeric schema",
        brokered_class="read",
        parameters=parameters,
    ) == "sha256:51ebe9cdbb967453ba3ae9fb737566028814968d8121ba0d32825f7c8ffb5639"


def test_brokered_tool_schema_digest_canonicalizes_exponent_number_spellings():
    from agentkit_serve_common.config import brokered_tool_schema_digest

    parameters = {
        "type": "object",
        "properties": {
            "small": {"type": "number", "minimum": 1e-7},
            "large": {"type": "number", "maximum": 1e21},
        },
    }

    assert brokered_tool_schema_digest(
        name="exponent-num-tool",
        description="Exponent numeric schema",
        brokered_class="read",
        parameters=parameters,
    ) == "sha256:bb4fac58c3f65a33ed3c4ebfa10e5da9c3dc5a9250a1316f64d25e40e0e645e0"
