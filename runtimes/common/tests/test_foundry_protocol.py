from __future__ import annotations

from types import TracebackType

from fastapi.testclient import TestClient

from agentkit_serve_common.config import AgentSpec
from agentkit_serve_common.conversation import RunRequest
from agentkit_serve_common.foundry import create_foundry_app
from agentkit_serve_common.runtime import RunResult, RuntimeSession


def _spec() -> AgentSpec:
    return AgentSpec.model_validate(
        {
            "abiVersion": "v0",
            "metadata": {"name": "foundry-test"},
            "model": {
                "provider": "openai-compatible",
                "baseURL": "https://api.openai.com/v1",
                "name": "gpt-4o-mini",
            },
            "instructions": "Be helpful.",
            "tools": [],
            "expose": {"openai": True, "port": 8080},
        }
    )


class EchoRuntime:
    def __init__(self) -> None:
        self.requests: list[RunRequest] = []

    async def __aenter__(self) -> RuntimeSession:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        return None

    async def run(self, request: RunRequest) -> RunResult:
        self.requests.append(request)
        return RunResult(text=f"echo: {request.prompt}", usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3})


class EchoFactory:
    def __init__(self) -> None:
        self.runtime = EchoRuntime()

    def build_runtime(self, spec: AgentSpec) -> RuntimeSession:
        return self.runtime


def test_foundry_invocations_and_responses_protocols():
    factory = EchoFactory()
    app = create_foundry_app(_spec(), factory)

    with TestClient(app) as client:
        readiness = client.get("/readiness")
        assert readiness.status_code == 200
        assert readiness.json() == {"ready": True}

        inv = client.post("/invocations", json={"message": "hello"})
        assert inv.status_code == 200
        assert inv.json()["response"] == "echo: hello"

        resp = client.post("/responses", json={"input": "hi"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "completed"
        assert body["output"][0]["content"][0]["text"] == "echo: hi"
        assert body["usage"] == {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}


def test_foundry_non_brokered_ignores_brokered_state_file_env(monkeypatch, tmp_path):
    state_file = tmp_path / "corrupt-state.json"
    state_file.write_text("not json", encoding="utf-8")
    monkeypatch.setenv("AGENTKIT_FOUNDRY_RESPONSE_STATE_FILE", str(state_file))

    app = create_foundry_app(_spec(), EchoFactory())
    with TestClient(app) as client:
        resp = client.post("/responses", json={"input": "hi"})

    assert resp.status_code == 200
    assert resp.json()["output"][0]["content"][0]["text"] == "echo: hi"


def test_foundry_responses_tolerates_stream_flag_with_non_streaming_response():
    app = create_foundry_app(_spec(), EchoFactory())
    with TestClient(app) as client:
        resp = client.post("/responses", json={"input": "hi", "stream": True})
    assert resp.status_code == 200
    assert resp.json()["status"] == "completed"
    assert resp.json()["output"][0]["content"][0]["text"] == "echo: hi"


def test_foundry_protocols_reject_non_object_json():
    app = create_foundry_app(_spec(), EchoFactory())
    with TestClient(app) as client:
        inv = client.post("/invocations", json=[])
        resp = client.post("/responses", json=[])

    assert inv.status_code == 400
    assert "JSON object" in inv.text
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_request"


def test_foundry_protocol_forwards_session_header_and_query_param():
    factory = EchoFactory()
    app = create_foundry_app(_spec(), factory)

    with TestClient(app) as client:
        inv = client.post("/invocations", json={"message": "hello"}, headers={"x-agent-session-id": "session-1"})
        resp = client.post("/responses?agent_session_id=session-2", json={"input": "hi"})

    assert inv.status_code == 200
    assert resp.status_code == 200
    assert factory.runtime.requests[0].session_id == "session-1"
    assert factory.runtime.requests[1].session_id == "session-2"


def test_foundry_responses_prefers_body_session_id_over_local_query_and_header(monkeypatch):
    monkeypatch.delenv("FOUNDRY_AGENT_SESSION_ID", raising=False)
    factory = EchoFactory()
    app = create_foundry_app(_spec(), factory)

    with TestClient(app) as client:
        resp = client.post(
            "/responses?agent_session_id=query-session",
            headers={"x-agent-session-id": "header-session"},
            json={"input": "hi", "agent_session_id": "body-session"},
        )

    assert resp.status_code == 200
    assert factory.runtime.requests[0].session_id == "body-session"
    assert "agent_session_id" not in resp.json()


def test_foundry_responses_prefers_hosted_session_identity_over_caller_fields(monkeypatch):
    monkeypatch.setenv("FOUNDRY_AGENT_SESSION_ID", "hosted-session")
    factory = EchoFactory()
    app = create_foundry_app(_spec(), factory)

    with TestClient(app) as client:
        resp = client.post(
            "/responses?agent_session_id=query-session",
            headers={"x-agent-session-id": "header-session"},
            json={"input": "hi", "agent_session_id": "body-session"},
        )

    assert resp.status_code == 200
    assert factory.runtime.requests[0].session_id == "hosted-session"
    assert "agent_session_id" not in resp.json()


def test_foundry_responses_accepts_body_session_id_compatibility_field(monkeypatch):
    monkeypatch.delenv("FOUNDRY_AGENT_SESSION_ID", raising=False)
    factory = EchoFactory()
    app = create_foundry_app(_spec(), factory)

    with TestClient(app) as client:
        resp = client.post("/responses", json={"input": "hi", "session_id": "compat-session"})

    assert resp.status_code == 200
    assert factory.runtime.requests[0].session_id == "compat-session"


def test_foundry_responses_ignores_portable_optional_fields():
    app = create_foundry_app(_spec(), EchoFactory())
    with TestClient(app) as client:
        resp = client.post("/responses", json={"input": "hi", "max_output_tokens": 500})

    assert resp.status_code == 200
    assert resp.json()["output"][0]["content"][0]["text"] == "echo: hi"


def test_foundry_responses_preserves_message_history_for_list_input():
    factory = EchoFactory()
    app = create_foundry_app(_spec(), factory)

    with TestClient(app) as client:
        resp = client.post(
            "/responses",
            json={
                "input": [
                    {"role": "system", "content": "system context"},
                    {"role": "user", "content": "first turn"},
                    {"role": "assistant", "content": "assistant reply"},
                    {"role": "user", "content": "final prompt"},
                ]
            },
        )

    assert resp.status_code == 200
    request = factory.runtime.requests[0]
    assert request.prompt == "final prompt"
    assert [(turn.role, turn.text) for turn in request.history] == [
        ("system", "system context"),
        ("user", "first turn"),
        ("assistant", "assistant reply"),
    ]


def test_foundry_non_brokered_responses_previous_response_id_does_not_force_continuation():
    factory = EchoFactory()
    app = create_foundry_app(_spec(), factory)

    with TestClient(app) as client:
        resp = client.post("/responses", json={"previous_response_id": "caresp_prior", "input": "next prompt"})

    assert resp.status_code == 200
    assert resp.json()["output"][0]["content"][0]["text"] == "echo: next prompt"
    assert factory.runtime.requests[0].prompt == "next prompt"


def test_foundry_non_brokered_function_call_output_is_not_routed_to_brokered_state_machine():
    factory = EchoFactory()
    app = create_foundry_app(_spec(), factory)

    with TestClient(app) as client:
        resp = client.post(
            "/responses",
            json={
                "previous_response_id": "caresp_prior",
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": "call_prior_1",
                        "output": '{"approved":true,"output":{"ok":true}}',
                    }
                ],
            },
        )

    assert resp.status_code == 200
    assert resp.json()["output"][0]["content"][0]["text"].startswith("echo: ")
    assert "function_call_output" in factory.runtime.requests[0].prompt


def test_foundry_non_brokered_function_call_output_input_stays_on_runtime_path():
    factory = EchoFactory()
    app = create_foundry_app(_spec(), factory)

    with TestClient(app) as client:
        resp = client.post(
            "/responses",
            json={
                "previous_response_id": "caresp_prior",
                "input": [{"type": "function_call_output", "call_id": "call_1", "output": "{}"}],
            },
        )

    assert resp.status_code == 200
    assert "function_call_output" in factory.runtime.requests[0].prompt
    assert resp.json()["output"][0]["content"][0]["text"].startswith("echo:")


def test_foundry_response_id_generator_tolerates_zero_arg_sdk(monkeypatch):
    from agentkit_serve_common import foundry

    class ZeroArgIdGenerator:
        @staticmethod
        def new_response_id():
            return "caresp_zero_arg"

        @staticmethod
        def new_message_item_id(response_id: str):
            return f"msg_{response_id}"

    monkeypatch.setattr(foundry, "_AzureResponsesIdGenerator", ZeroArgIdGenerator)
    app = create_foundry_app(_spec(), EchoFactory())

    with TestClient(app) as client:
        resp = client.post("/responses", json={"previous_response_id": "caresp_previous", "input": "hi"})

    assert resp.status_code == 200
    assert resp.json()["id"] == "caresp_zero_arg"


def test_foundry_responses_rejects_request_supplied_tools():
    app = create_foundry_app(_spec(), EchoFactory())
    with TestClient(app) as client:
        tools = client.post("/responses", json={"input": "hi", "tools": [{"type": "function"}]})
        choice = client.post("/responses", json={"input": "hi", "tool_choice": {"type": "function"}})

    assert tools.status_code == 400
    assert tools.json()["error"]["code"] == "tools_unsupported"
    assert choice.status_code == 400
    assert choice.json()["error"]["code"] == "tool_choice_unsupported"


def test_foundry_protocols_require_auth_when_token_configured_but_readiness_stays_open():
    app = create_foundry_app(_spec(), EchoFactory(), auth_token="foundry-token")
    with TestClient(app) as client:
        readiness = client.get("/readiness")
        unauth_inv = client.post("/invocations", json={"message": "hello"})
        auth_inv = client.post(
            "/invocations",
            json={"message": "hello"},
            headers={"authorization": "Bearer foundry-token"},
        )

    assert readiness.status_code == 200
    assert unauth_inv.status_code == 401
    assert auth_inv.status_code == 200


def test_foundry_protocol_uses_platform_session_env_fallback(monkeypatch):
    factory = EchoFactory()
    app = create_foundry_app(_spec(), factory)
    monkeypatch.setenv("FOUNDRY_AGENT_SESSION_ID", "platform-session")

    with TestClient(app) as client:
        resp = client.post("/invocations", json={"message": "hello"})

    assert resp.status_code == 200
    assert factory.runtime.requests[0].session_id == "platform-session"


def test_foundry_brokered_cli_dry_run_loads_static_brokered_agent(tmp_path, capsys):
    from agentkit_serve_common.foundry_brokered_cli import main

    config = tmp_path / "agent.yaml"
    config.write_text(
        """abiVersion: v0
metadata:
  name: brokered-cli
model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
  name: gpt-4o-mini
instructions: Broker tools.
tools: []
brokeredTools:
  - name: conformance_read
    description: Read conformance data.
    brokeredClass: read
    parameters:
      type: object
      properties:
        probe:
          type: boolean
expose:
  openai: true
  port: 8088
""",
        encoding="utf-8",
    )

    assert main(["--config", str(config), "--dry-run"]) == 0

    output = capsys.readouterr().out
    assert '"agent": "brokered-cli"' in output
    assert '"brokeredTools": ["conformance_read"]' in output


def test_foundry_brokered_cli_rejects_agents_without_brokered_tools(tmp_path):
    from agentkit_serve_common.foundry_brokered_cli import main

    config = tmp_path / "agent.yaml"
    config.write_text(
        """abiVersion: v0
metadata:
  name: not-brokered
model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
  name: gpt-4o-mini
instructions: Be helpful.
tools: []
expose:
  openai: true
  port: 8088
""",
        encoding="utf-8",
    )

    try:
        main(["--config", str(config), "--dry-run"])
    except SystemExit as exc:
        assert "brokeredTools" in str(exc)
    else:  # pragma: no cover - assertion path.
        raise AssertionError("expected missing brokeredTools to fail")
