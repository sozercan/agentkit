from __future__ import annotations

import json
import time
from types import TracebackType
from typing import Any

import pytest
from fastapi.testclient import TestClient

from agentkit_serve_common.config import AgentSpec
from agentkit_serve_common.conversation import RunRequest
from agentkit_serve_common.orka import ORKA_HARNESS_VERSION, create_orka_app
from agentkit_serve_common.runtime import RunResult, RuntimeSession

AUTH = {"authorization": "Bearer test-token"}


def _spec() -> AgentSpec:
    return AgentSpec.model_validate(
        {
            "abiVersion": "v0",
            "metadata": {"name": "orka-test"},
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
    def __init__(self, *, delay: float = 0) -> None:
        self.requests: list[RunRequest] = []
        self.delay = delay

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
        import asyncio

        self.requests.append(request)
        if self.delay:
            await asyncio.sleep(self.delay)
        return RunResult(text=f"echo: {request.prompt}", usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3})


class EchoFactory:
    def __init__(self, *, delay: float = 0) -> None:
        self.runtime = EchoRuntime(delay=delay)

    def build_runtime(self, spec: AgentSpec) -> RuntimeSession:
        return self.runtime


def _frames(resp_text: str) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for raw in resp_text.strip().split("\n\n"):
        if not raw:
            continue
        data_lines = [line.removeprefix("data: ") for line in raw.splitlines() if line.startswith("data: ")]
        assert len(data_lines) == 1, raw
        frames.append(json.loads(data_lines[0]))
    return frames


def _create_turn(client: TestClient, **overrides: Any) -> str:
    payload = {"version": ORKA_HARNESS_VERSION, "turnID": "turn-1", "prompt": "hello"}
    payload.update(overrides)
    resp = client.post("/v1/turns", json=payload, headers=AUTH)
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["status"] == "accepted"
    return body["turnID"]



def test_orka_app_factory_requires_auth_token():
    with pytest.raises(ValueError, match="requires a bearer auth token"):
        create_orka_app(_spec(), EchoFactory())


def test_orka_health_and_capabilities_are_open():
    app = create_orka_app(_spec(), EchoFactory(), auth_token="test-token")

    with TestClient(app) as client:
        health = client.get("/v1/health")
        caps = client.get("/v1/capabilities")

    assert health.status_code == 200
    assert health.json() == {"version": ORKA_HARNESS_VERSION, "status": "ok", "ready": True}
    assert caps.status_code == 200
    assert caps.json()["toolExecutionModes"] == ["observed"]
    assert caps.json()["supportsCancel"] is True


def test_orka_turn_lifecycle_streams_exactly_one_terminal_frame():
    app = create_orka_app(_spec(), EchoFactory(), auth_token="test-token")

    with TestClient(app) as client:
        turn_id = _create_turn(client)
        resp = client.get(f"/v1/turns/{turn_id}/events", headers=AUTH)

    assert resp.status_code == 200
    frames = _frames(resp.text)
    assert [frame["type"] for frame in frames] == ["TurnStarted", "TurnCompleted"]
    terminals = [frame for frame in frames if frame["type"] in {"TurnCompleted", "TurnFailed", "TurnCancelled"}]
    assert len(terminals) == 1
    assert terminals[0]["payload"]["result"]["text"] == "echo: hello"


def test_orka_events_support_after_seq_replay():
    app = create_orka_app(_spec(), EchoFactory(), auth_token="test-token")

    with TestClient(app) as client:
        turn_id = _create_turn(client)
        all_events = _frames(client.get(f"/v1/turns/{turn_id}/events", headers=AUTH).text)
        replay = client.get(f"/v1/turns/{turn_id}/events?afterSeq=1", headers=AUTH)

    assert [frame["seq"] for frame in all_events] == [1, 2]
    assert replay.status_code == 200
    assert [frame["type"] for frame in _frames(replay.text)] == ["TurnCompleted"]


def test_orka_turn_forwards_per_turn_metadata_env_and_session_fields():
    factory = EchoFactory()
    app = create_orka_app(_spec(), factory, auth_token="test-token")

    with TestClient(app) as client:
        turn_id = _create_turn(
            client,
            turnID="turn-meta",
            sessionID="session-1",
            correlationID="corr-1",
            env={"MODEL_TOKEN": "per-run"},
            metadata={"tenant": "acme"},
            history=[{"role": "system", "text": "be terse"}],
        )
        client.get(f"/v1/turns/{turn_id}/events", headers=AUTH)

    request = factory.runtime.requests[0]
    assert request.turn_id == "turn-meta"
    assert request.session_id == "session-1"
    assert request.correlation_id == "corr-1"
    assert request.env == {"MODEL_TOKEN": "per-run"}
    assert request.metadata == {"tenant": "acme"}
    assert [(turn.role, turn.text) for turn in request.history] == [("system", "be terse")]


def test_orka_protected_endpoints_require_bearer_token():
    app = create_orka_app(_spec(), EchoFactory(), auth_token="test-token")

    with TestClient(app) as client:
        create = client.post("/v1/turns", json={"version": ORKA_HARNESS_VERSION, "prompt": "hi"})
        events = client.get("/v1/turns/missing/events")
        cancel = client.post("/v1/turns/missing/cancel")

    assert create.status_code == 401
    assert events.status_code == 401
    assert cancel.status_code == 401


def test_orka_cancel_produces_cancelled_terminal_frame():
    app = create_orka_app(_spec(), EchoFactory(delay=60), auth_token="test-token")

    with TestClient(app) as client:
        turn_id = _create_turn(client, turnID="turn-cancel", prompt="slow")
        cancel = client.post(f"/v1/turns/{turn_id}/cancel", headers=AUTH)
        # Give the event-loop portal a moment to deliver task cancellation.
        for _ in range(20):
            frames = _frames(client.get(f"/v1/turns/{turn_id}/events", headers=AUTH).text)
            if frames[-1]["type"] == "TurnCancelled":
                break
            time.sleep(0.01)

    assert cancel.status_code == 202
    assert [frame["type"] for frame in frames] == ["TurnStarted", "TurnCancelled"]


def test_orka_terminal_turn_retention_is_bounded():
    app = create_orka_app(_spec(), EchoFactory(), auth_token="test-token", max_terminal_turns=1)

    with TestClient(app) as client:
        first_id = _create_turn(client, turnID="turn-old", prompt="old")
        first_events = client.get(f"/v1/turns/{first_id}/events", headers=AUTH)
        assert first_events.status_code == 200

        second_id = _create_turn(client, turnID="turn-new", prompt="new")
        second_events = client.get(f"/v1/turns/{second_id}/events", headers=AUTH)
        assert second_events.status_code == 200

        evicted = client.get(f"/v1/turns/{first_id}/events", headers=AUTH)
        kept = client.get(f"/v1/turns/{second_id}/events?afterSeq=1", headers=AUTH)

    assert evicted.status_code == 404
    assert kept.status_code == 200
    assert [frame["type"] for frame in _frames(kept.text)] == ["TurnCompleted"]
