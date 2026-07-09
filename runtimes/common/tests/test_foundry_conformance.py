from __future__ import annotations

import json

from fastapi.testclient import TestClient
from agentkit_serve_common.foundry_conformance import create_foundry_conformance_app


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
