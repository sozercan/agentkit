from __future__ import annotations

import json
import tomllib
from pathlib import Path

from fastapi.testclient import TestClient
from agentkit_serve_common.foundry_conformance import create_foundry_conformance_app


def test_foundry_conformance_sdk_dependency_is_optional():
    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    sdk_dependency = "azure-ai-agentserver-responses"

    assert not any(dependency.startswith(sdk_dependency) for dependency in project["dependencies"])
    assert any(
        dependency.startswith(sdk_dependency)
        for dependency in project["optional-dependencies"]["foundry-conformance"]
    )


def _function_output(previous_response_id: str | None, call_id: str = "call_conformance_1") -> dict:
    payload = {
        "input": [
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": '{"approved":true,"output":{"success":true}}',
                "status": "completed",
            }
        ]
    }
    if previous_response_id is not None:
        payload["previous_response_id"] = previous_response_id
    return payload


def test_foundry_conformance_sdk_function_call_loop_uses_platform_ids():
    app = create_foundry_conformance_app()

    with TestClient(app) as client:
        readiness = client.get("/readiness")
        initial = client.post("/responses", json={"input": "hello"})
        initial_body = initial.json()
        call = initial_body["output"][0]
        final = client.post("/responses", json=_function_output(initial_body["id"]))

    assert readiness.status_code == 200
    assert readiness.json()["protocols"] == {"responses": "2.0.0"}
    assert initial.status_code == 200, initial.text
    assert initial_body["id"].startswith("caresp_")
    assert not initial_body["id"].startswith("resp_")
    assert call == {
        "type": "function_call",
        "id": call["id"],
        "call_id": "call_conformance_1",
        "name": "conformance_read",
        "arguments": '{"probe":true}',
        "status": "completed",
        "response_id": initial_body["id"],
        "agent_reference": None,
    }
    assert final.status_code == 200, final.text
    final_body = final.json()
    assert final_body["previous_response_id"] == initial_body["id"]
    assert final_body["id"].startswith("caresp_")
    assert final_body["output"][0]["type"] == "message"
    assert "success" in final_body["output"][0]["content"][0]["text"]


def test_foundry_conformance_sdk_rejects_orphan_and_unknown_continuations():
    app = create_foundry_conformance_app()

    with TestClient(app) as client:
        missing_previous = client.post("/responses", json=_function_output(None))
        unknown_previous = client.post("/responses", json=_function_output("caresp_0123456789abcdef00ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef"))
        initial = client.post("/responses", json={"input": "hello"}).json()
        unknown_call = client.post("/responses", json=_function_output(initial["id"], call_id="call_other"))

    assert missing_previous.status_code == 200
    assert missing_previous.json()["status"] == "failed"
    assert missing_previous.json()["error"]["code"] == "missing_previous_response_id"
    assert unknown_previous.status_code == 200
    assert unknown_previous.json()["status"] == "failed"
    assert unknown_previous.json()["error"]["code"] == "unknown_previous_response_id"
    assert unknown_call.status_code == 200
    assert unknown_call.json()["status"] == "failed"
    assert unknown_call.json()["error"]["code"] == "unknown_call_id"


def test_foundry_conformance_sdk_rejects_request_level_tools():
    app = create_foundry_conformance_app()

    with TestClient(app) as client:
        response = client.post(
            "/responses",
            json={
                "input": "hello",
                "tools": [{"type": "function", "name": "unsafe", "parameters": {"type": "object"}}],
            },
        )

    assert response.status_code == 200
    assert response.json()["status"] == "failed"
    assert response.json()["error"]["code"] == "tools_unsupported"


def test_foundry_conformance_sdk_bounds_pending_response_state():
    app = create_foundry_conformance_app(max_pending_responses=1)

    with TestClient(app) as client:
        first = client.post("/responses", json={"input": "first", "store": True}).json()
        second = client.post("/responses", json={"input": "second"}).json()
        completed = client.post(
            "/responses",
            json=_function_output(first["id"], first["output"][0]["call_id"]),
        ).json()
        third = client.post("/responses", json={"input": "third"}).json()

    assert second["status"] == "failed"
    assert second["error"]["code"] == "brokered_response_state_full"
    assert completed["status"] == "completed"
    assert third["output"][0]["type"] == "function_call"
    assert len(app.state.conformance_store._entries) == 0
    assert len(app.state.conformance_store._item_store) == 0
    assert len(app.state.conformance_store._stream_events) == 0


def test_foundry_conformance_sdk_bounds_request_body_before_rewrite(monkeypatch):
    monkeypatch.setenv("FOUNDRY_AGENT_SESSION_ID", "hosted-session")
    app = create_foundry_conformance_app(max_request_body_bytes=64)

    with TestClient(app) as client:
        response = client.post(
            "/responses",
            content=json.dumps({"input": "x" * 128}),
            headers={"content-type": "application/json", "x-request-id": "oversized-request"},
        )
        generated_id_response = client.post(
            "/responses",
            content=json.dumps({"input": "x" * 128}),
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_body_too_large"
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert response.json()["error"]["additionalInfo"]["request_id"] == "oversized-request"
    assert response.headers["x-request-id"] == "oversized-request"
    assert "azure-ai-agentserver-responses" in response.headers["x-platform-server"]
    assert response.headers["x-platform-error-source"] == "user"
    assert response.headers["x-agent-session-id"] == "hosted-session"
    assert generated_id_response.json()["error"]["additionalInfo"]["request_id"] == generated_id_response.headers["x-request-id"]


def test_foundry_conformance_sdk_explicitly_rejects_background_mode():
    app = create_foundry_conformance_app()

    with TestClient(app) as client:
        response = client.post(
            "/responses",
            json={"input": "background", "background": True, "store": True},
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "background_unsupported"
    assert response.headers["x-platform-error-source"] == "user"
    assert len(app.state.conformance_store._entries) == 0


def test_foundry_conformance_sdk_preserves_invalid_store_for_protocol_validation():
    app = create_foundry_conformance_app()

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "invalid store", "store": "not-a-boolean"})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"
    assert len(app.state.conformance_store._entries) == 0


def test_foundry_conformance_sdk_rewrites_nullable_store_to_non_stored():
    app = create_foundry_conformance_app()

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "nullable store", "store": None})

    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert len(app.state.conformance_store._entries) == 0


def test_foundry_conformance_console_dry_run(capsys):
    from agentkit_serve_common.foundry_conformance import main

    assert main(["--host", "127.0.0.1", "--port", "18088", "--model", "conformance-model", "--dry-run"]) == 0

    body = json.loads(capsys.readouterr().out)
    assert body == {
        "host": "127.0.0.1",
        "model": "conformance-model",
        "port": 18088,
        "protocols": {"responses": "2.0.0"},
    }
