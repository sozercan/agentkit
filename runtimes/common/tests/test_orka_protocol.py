from __future__ import annotations

import json
import time
from datetime import UTC, datetime, timedelta
from types import TracebackType
from typing import Any

import pytest
from fastapi.testclient import TestClient

from agentkit_serve_common.config import AgentSpec
from agentkit_serve_common.conversation import RunRequest
from agentkit_serve_common.orka import ORKA_HARNESS_VERSION, create_orka_app
from agentkit_serve_common.runtime import (
    BrokeredToolCall,
    BrokeredToolDefinition,
    BrokeredToolResult,
    OfflineEchoRuntimeFactory,
    RunResult,
    RuntimeSession,
    ToolBroker,
)

AUTH = {"authorization": "Bearer test-token"}
TERMINAL_TYPES = {"TurnCompleted", "TurnFailed", "TurnCancelled"}


def _deadline() -> str:
    return (datetime.now(UTC) + timedelta(minutes=5)).isoformat().replace("+00:00", "Z")


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
            "env": [{"name": "MODEL_TOKEN"}],
            "expose": {"openai": True, "port": 8080},
        }
    )


class EchoRuntime:
    def __init__(self, *, delay: float = 0, delays: dict[str, float] | None = None) -> None:
        self.requests: list[RunRequest] = []
        self.delay = delay
        self.delays = delays or {}

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
        delay = self.delays.get(request.prompt, self.delay)
        if delay:
            await asyncio.sleep(delay)
        return RunResult(text=f"echo: {request.prompt}", usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3})


class EchoFactory:
    def __init__(self, *, delay: float = 0, delays: dict[str, float] | None = None) -> None:
        self.runtime = EchoRuntime(delay=delay, delays=delays)

    def build_runtime(self, spec: AgentSpec) -> RuntimeSession:
        return self.runtime


def _start_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "version": ORKA_HARNESS_VERSION,
        "namespace": "default",
        "taskName": "task-1",
        "sessionName": "session-1",
        "runtimeSessionID": "runtime-session-1",
        "turnID": "turn-1",
        "correlationID": "corr-1",
        "deadline": _deadline(),
        "authIdentity": {"subject": "system:serviceaccount:default:orka"},
        "input": {"prompt": "hello", "contextRefs": [], "env": []},
        "toolExecutionMode": "observed",
        "metadata": {},
    }
    payload.update(overrides)
    return payload


def _cancel_payload(**overrides: Any) -> dict[str, Any]:
    payload = {
        "version": ORKA_HARNESS_VERSION,
        "namespace": "default",
        "taskName": "task-1",
        "sessionName": "session-1",
        "runtimeSessionID": "runtime-session-1",
        "turnID": "turn-1",
        "correlationID": "corr-1",
        "reason": "test requested cancel",
    }
    payload.update(overrides)
    return payload


def _continue_payload(**overrides: Any) -> dict[str, Any]:
    payload = {
        "version": ORKA_HARNESS_VERSION,
        "namespace": "default",
        "taskName": "task-1",
        "sessionName": "session-1",
        "runtimeSessionID": "runtime-session-1",
        "turnID": "turn-1",
        "correlationID": "corr-1",
        "toolResults": [
            {
                "version": ORKA_HARNESS_VERSION,
                "runtimeSessionID": "runtime-session-1",
                "turnID": "turn-1",
                "toolCallID": "tool-call-1",
                "idempotencyKey": "runtime-session-1:turn-1:tool-call-1",
                "approved": True,
                "output": {"success": True, "data": {"answer": "ok"}},
            }
        ],
    }
    payload.update(overrides)
    return payload


def _brokered_input(**overrides: Any) -> dict[str, Any]:
    value = {
        "prompt": "hello",
        "contextRefs": [],
        "env": [],
        "tools": [
            {
                "name": "conformance_read",
                "description": "Synthetic conformance read tool",
                "brokeredClass": "read",
                "parameters": {"type": "object"},
            }
        ],
    }
    value.update(overrides)
    return value


def _brokered_write_input(**overrides: Any) -> dict[str, Any]:
    return _brokered_input(
        tools=[
            {
                "name": "conformance_write",
                "description": "Synthetic conformance write tool",
                "brokeredClass": "write",
                "parameters": {"type": "object"},
            }
        ],
        **overrides,
    )


def _brokered_coordination_input(**overrides: Any) -> dict[str, Any]:
    return _brokered_input(
        tools=[
            {
                "name": "conformance_coordination",
                "description": "Synthetic conformance coordination tool",
                "brokeredClass": "coordination",
                "parameters": {"type": "object"},
            }
        ],
        **overrides,
    )


def _wait_for_event_type(client: TestClient, turn_id: str, event_type: str) -> dict[str, Any]:
    for _ in range(100):
        events = client.app.state.turns[turn_id].events
        for event in events:
            frame = event.as_frame()
            if frame["type"] == event_type:
                return frame
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {event_type}")


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
    payload = _start_payload(**overrides)
    resp = client.post("/v1/turns", json=payload, headers=AUTH)
    assert resp.status_code == 202, resp.text
    assert resp.json() == {
        "version": ORKA_HARNESS_VERSION,
        "accepted": True,
        "runtimeSessionID": payload["runtimeSessionID"],
        "turnID": payload["turnID"],
        "correlationID": payload["correlationID"],
        "eventStreamPath": f"/v1/turns/{payload['turnID']}/events",
    }
    return payload["turnID"]


def _assert_frame_identity(frame: dict[str, Any], *, seq: int, typ: str, runtime_session_id: str = "runtime-session-1", turn_id: str = "turn-1", correlation_id: str = "corr-1") -> None:
    assert frame["version"] == ORKA_HARNESS_VERSION
    assert frame["type"] == typ
    assert frame["runtimeSessionID"] == runtime_session_id
    assert frame["turnID"] == turn_id
    assert frame["correlationID"] == correlation_id
    assert frame["seq"] == seq
    assert frame["severity"] in {"info", "error"}
    assert isinstance(frame["summary"], str)
    assert isinstance(frame["metadata"], dict)
    assert "createdAt" in frame
    assert "timestamp" not in frame
    assert "payload" not in frame


def test_orka_app_factory_requires_auth_token():
    with pytest.raises(ValueError, match="requires a bearer auth token"):
        create_orka_app(_spec(), EchoFactory())


def test_orka_health_and_capabilities_are_open_and_match_contract():
    app = create_orka_app(_spec(), EchoFactory(), auth_token="test-token")

    with TestClient(app) as client:
        health = client.get("/v1/health")
        caps = client.get("/v1/capabilities")

    assert health.status_code == 200
    health_body = health.json()
    assert set(health_body) == {"version", "status", "ready", "checkedAt", "metadata"}
    assert health_body["version"] == ORKA_HARNESS_VERSION
    assert health_body["status"] == "ok"
    assert health_body["ready"] is True
    assert health_body["metadata"] == {"agentName": "orka-test"}
    assert datetime.fromisoformat(health_body["checkedAt"].replace("Z", "+00:00"))

    assert caps.status_code == 200
    assert caps.json() == {
        "version": ORKA_HARNESS_VERSION,
        "protocolVersion": ORKA_HARNESS_VERSION,
        "transport": "http+sse",
        "runtimeName": "agentkit-serve",
        "runtimeVersion": "0.0.0",
        "providerKind": "kubernetes-service",
        "toolExecutionModes": ["observed"],
        "supportsCancel": True,
        "supportsRuntimeSessions": True,
        "supportsSuspend": False,
        "supportsWorkspaceSnapshot": False,
        "maxConcurrentTurns": 1,
        "metadata": {"agentName": "orka-test", "model": "gpt-4o-mini", "agentkitProvider": "openai-compatible"},
    }


def test_orka_turn_lifecycle_streams_contract_frames_and_one_terminal():
    app = create_orka_app(_spec(), EchoFactory(), auth_token="test-token")

    with TestClient(app) as client:
        turn_id = _create_turn(client)
        resp = client.get(f"/v1/turns/{turn_id}/events", headers=AUTH)

    assert resp.status_code == 200
    frames = _frames(resp.text)
    assert [frame["type"] for frame in frames] == ["TurnStarted", "RuntimeOutput", "TurnCompleted"]
    _assert_frame_identity(frames[0], seq=1, typ="TurnStarted")
    _assert_frame_identity(frames[1], seq=2, typ="RuntimeOutput")
    assert frames[1]["contentText"] == "echo: hello"
    assert frames[1]["content"] == {
        "message": "echo: hello",
        "usage": {"completion_tokens": 2, "prompt_tokens": 1, "total_tokens": 3},
    }
    _assert_frame_identity(frames[2], seq=3, typ="TurnCompleted")
    assert frames[2]["completed"] == {"result": "echo: hello", "finalEventSeq": 3}
    assert frames[2]["failed"] is None
    assert frames[2]["error"] is None
    terminals = [frame for frame in frames if frame["type"] in TERMINAL_TYPES]
    assert len(terminals) == 1


def test_orka_events_support_after_seq_replay():
    app = create_orka_app(_spec(), EchoFactory(), auth_token="test-token")

    with TestClient(app) as client:
        turn_id = _create_turn(client)
        all_events = _frames(client.get(f"/v1/turns/{turn_id}/events", headers=AUTH).text)
        replay = client.get(f"/v1/turns/{turn_id}/events?afterSeq=1", headers=AUTH)

    assert [frame["seq"] for frame in all_events] == [1, 2, 3]
    assert replay.status_code == 200
    assert [frame["type"] for frame in _frames(replay.text)] == ["RuntimeOutput", "TurnCompleted"]


def test_orka_duplicate_turn_rejection_matches_orka_conformance_contract():
    app = create_orka_app(_spec(), EchoFactory(delay=60), auth_token="test-token")

    with TestClient(app) as client:
        turn_id = _create_turn(client, turnID="turn-duplicate", input={"prompt": "slow", "contextRefs": [], "env": []})
        duplicate = client.post("/v1/turns", json=_start_payload(turnID=turn_id, input={"prompt": "slow", "contextRefs": [], "env": []}), headers=AUTH)
        cancel = client.post(f"/v1/turns/{turn_id}/cancel", json=_cancel_payload(turnID=turn_id), headers=AUTH)

    assert duplicate.status_code == 409
    assert duplicate.json() == {"detail": "turn already exists"}
    assert cancel.status_code == 202


def test_orka_turn_forwards_per_turn_metadata_env_and_session_fields():
    factory = EchoFactory()
    app = create_orka_app(_spec(), factory, auth_token="test-token")

    with TestClient(app) as client:
        turn_id = _create_turn(
            client,
            turnID="turn-meta",
            runtimeSessionID="runtime-session-meta",
            correlationID="corr-meta",
            input={"prompt": "hello", "contextRefs": [], "env": [{"name": "MODEL_TOKEN", "value": "per-run"}]},
            metadata={"tenant": "acme"},
        )
        client.get(f"/v1/turns/{turn_id}/events", headers=AUTH)

    request = factory.runtime.requests[0]
    assert request.prompt == "hello"
    assert request.turn_id == "turn-meta"
    assert request.session_id == "runtime-session-meta"
    assert request.correlation_id == "corr-meta"
    assert request.env == {"MODEL_TOKEN": "per-run"}
    assert request.metadata == {"tenant": "acme"}


def test_orka_start_turn_requires_contract_fields():
    app = create_orka_app(_spec(), EchoFactory(), auth_token="test-token")

    with TestClient(app) as client:
        missing_namespace = _start_payload()
        missing_namespace.pop("namespace")
        missing = client.post("/v1/turns", json=missing_namespace, headers=AUTH)
        missing_prompt = _start_payload(input={"contextRefs": [], "env": []})
        prompt_resp = client.post("/v1/turns", json=missing_prompt, headers=AUTH)
        bad_env = _start_payload(input={"prompt": "hi", "env": [{"name": "1_BAD", "value": "x"}]})
        env_resp = client.post("/v1/turns", json=bad_env, headers=AUTH)
        undeclared_env = _start_payload(input={"prompt": "hi", "env": [{"name": "OTHER_TOKEN", "value": "x"}]})
        undeclared_env_resp = client.post("/v1/turns", json=undeclared_env, headers=AUTH)
        reserved_env = _start_payload(input={"prompt": "hi", "env": [{"name": "AGENTKIT_WORKLOAD_IDENTITY_TOKEN_COMMAND", "value": "echo nope"}]})
        reserved_env_resp = client.post("/v1/turns", json=reserved_env, headers=AUTH)
        bad_context_refs = _start_payload(input={"prompt": "hi", "contextRefs": [{"kind": "artifact"}], "env": []})
        context_resp = client.post("/v1/turns", json=bad_context_refs, headers=AUTH)
        brokered = _start_payload(toolExecutionMode="brokered")
        brokered_resp = client.post("/v1/turns", json=brokered, headers=AUTH)

    assert missing.status_code == 400
    assert "namespace" in missing.text
    assert prompt_resp.status_code == 400
    assert "input.prompt" in prompt_resp.text
    assert env_resp.status_code == 400
    assert "input.env" in env_resp.text
    assert undeclared_env_resp.status_code == 400
    assert "not declared" in undeclared_env_resp.text
    assert reserved_env_resp.status_code == 400
    assert "reserved" in reserved_env_resp.text
    assert context_resp.status_code == 400
    assert "contextRefs" in context_resp.text
    assert brokered_resp.status_code == 400
    assert "toolExecutionMode" in brokered_resp.text



def test_orka_context_refs_are_validated_and_forwarded_as_safe_references():
    factory = EchoFactory()
    app = create_orka_app(_spec(), factory, auth_token="test-token")

    with TestClient(app) as client:
        turn_id = _create_turn(
            client,
            turnID="turn-context",
            input={
                "prompt": "hello",
                "contextRefs": [{"kind": "artifact", "name": "ctx", "seq": 7}],
                "env": [],
            },
        )
        resp = client.get(f"/v1/turns/{turn_id}/events", headers=AUTH)

    assert resp.status_code == 200
    request = factory.runtime.requests[0]
    assert request.metadata["contextRefs"] == '[{"kind":"artifact","name":"ctx","seq":7}]'
    assert request.history == ()


def test_orka_protected_endpoints_require_bearer_token():
    app = create_orka_app(_spec(), EchoFactory(), auth_token="test-token")

    with TestClient(app) as client:
        create = client.post("/v1/turns", json=_start_payload())
        events = client.get("/v1/turns/missing/events")
        cancel = client.post("/v1/turns/missing/cancel", json=_cancel_payload(turnID="missing"))

    assert create.status_code == 401
    assert events.status_code == 401
    assert cancel.status_code == 401


def test_orka_rejects_turn_ids_that_do_not_fit_route_path():
    app = create_orka_app(_spec(), EchoFactory(), auth_token="test-token")

    with TestClient(app) as client:
        slash = client.post("/v1/turns", json=_start_payload(turnID="bad/id"), headers=AUTH)
        query = client.post("/v1/turns", json=_start_payload(turnID="bad?id"), headers=AUTH)

    assert slash.status_code == 400
    assert query.status_code == 400
    assert "URL-safe" in slash.text


def test_orka_cancel_accepts_contract_request_and_produces_cancelled_terminal_frame():
    app = create_orka_app(_spec(), EchoFactory(delay=60), auth_token="test-token")

    with TestClient(app) as client:
        turn_id = _create_turn(client, turnID="turn-cancel", input={"prompt": "slow", "contextRefs": [], "env": []})
        cancel_payload = _cancel_payload(turnID=turn_id)
        cancel = client.post(f"/v1/turns/{turn_id}/cancel", json=cancel_payload, headers=AUTH)
        for _ in range(20):
            frames = _frames(client.get(f"/v1/turns/{turn_id}/events", headers=AUTH).text)
            if frames[-1]["type"] == "TurnCancelled":
                break
            time.sleep(0.01)

    assert cancel.status_code == 202
    assert cancel.json() == {
        "version": ORKA_HARNESS_VERSION,
        "accepted": True,
        "runtimeSessionID": cancel_payload["runtimeSessionID"],
        "turnID": turn_id,
        "correlationID": cancel_payload["correlationID"],
        "message": "cancel accepted",
    }
    assert [frame["type"] for frame in frames] == ["TurnStarted", "TurnCancelled"]
    _assert_frame_identity(frames[-1], seq=2, typ="TurnCancelled", turn_id=turn_id)



def test_orka_cancel_rejects_runtime_session_or_correlation_mismatch():
    app = create_orka_app(_spec(), EchoFactory(delay=60), auth_token="test-token")

    with TestClient(app) as client:
        turn_id = _create_turn(client, turnID="turn-cancel-identity", input={"prompt": "slow", "contextRefs": [], "env": []})
        wrong_session = client.post(
            f"/v1/turns/{turn_id}/cancel",
            json=_cancel_payload(turnID=turn_id, runtimeSessionID="other-session"),
            headers=AUTH,
        )
        wrong_correlation = client.post(
            f"/v1/turns/{turn_id}/cancel",
            json=_cancel_payload(turnID=turn_id, correlationID="other-corr"),
            headers=AUTH,
        )

    assert wrong_session.status_code == 400
    assert "runtimeSessionID" in wrong_session.text
    assert wrong_correlation.status_code == 400
    assert "correlationID" in wrong_correlation.text


def test_orka_cancel_rejects_body_turn_id_mismatch():
    app = create_orka_app(_spec(), EchoFactory(delay=60), auth_token="test-token")

    with TestClient(app) as client:
        turn_id = _create_turn(client, turnID="turn-cancel-mismatch", input={"prompt": "slow", "contextRefs": [], "env": []})
        mismatch = client.post(f"/v1/turns/{turn_id}/cancel", json=_cancel_payload(turnID="other-turn"), headers=AUTH)

    assert mismatch.status_code == 400
    assert "match route" in mismatch.text



def test_orka_enforces_advertised_single_active_turn_limit():
    app = create_orka_app(_spec(), EchoFactory(delay=60), auth_token="test-token")

    with TestClient(app) as client:
        first_id = _create_turn(client, turnID="turn-active", input={"prompt": "slow", "contextRefs": [], "env": []})
        rejected = client.post(
            "/v1/turns",
            json=_start_payload(turnID="turn-second", input={"prompt": "second", "contextRefs": [], "env": []}),
            headers=AUTH,
        )
        cancel = client.post(f"/v1/turns/{first_id}/cancel", json=_cancel_payload(turnID=first_id), headers=AUTH)
        assert cancel.status_code == 202
        frames = _frames(client.get(f"/v1/turns/{first_id}/events", headers=AUTH).text)
        assert frames[-1]["type"] == "TurnCancelled"
        accepted_after_terminal = client.post(
            "/v1/turns",
            json=_start_payload(turnID="turn-after", input={"prompt": "after", "contextRefs": [], "env": []}),
            headers=AUTH,
        )

    assert rejected.status_code == 429
    assert "maxConcurrentTurns" in rejected.text
    assert accepted_after_terminal.status_code == 202


def test_orka_terminal_turn_retention_is_bounded():
    app = create_orka_app(_spec(), EchoFactory(), auth_token="test-token", max_terminal_turns=1)

    with TestClient(app) as client:
        first_id = _create_turn(client, turnID="turn-old", input={"prompt": "old", "contextRefs": [], "env": []})
        first_events = client.get(f"/v1/turns/{first_id}/events", headers=AUTH)
        assert first_events.status_code == 200

        second_id = _create_turn(client, turnID="turn-new", input={"prompt": "new", "contextRefs": [], "env": []})
        second_events = client.get(f"/v1/turns/{second_id}/events", headers=AUTH)
        assert second_events.status_code == 200

        evicted = client.get(f"/v1/turns/{first_id}/events", headers=AUTH)
        kept = client.get(f"/v1/turns/{second_id}/events?afterSeq=1", headers=AUTH)

    assert evicted.status_code == 404
    assert kept.status_code == 200
    assert [frame["type"] for frame in _frames(kept.text)] == ["RuntimeOutput", "TurnCompleted"]


class EnvRuntime:
    def __init__(self, token: str) -> None:
        self.token = token
        self.requests: list[RunRequest] = []
        self.entered = 0
        self.exited = 0

    async def __aenter__(self) -> RuntimeSession:
        self.entered += 1
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        self.exited += 1
        return None

    async def run(self, request: RunRequest) -> RunResult:
        import os

        self.requests.append(request)
        return RunResult(text=f"token={self.token};run={os.environ.get('MODEL_TOKEN')}")


class EnvFactory:
    def __init__(self) -> None:
        self.runtimes: list[EnvRuntime] = []

    def build_runtime(self, spec: AgentSpec) -> RuntimeSession:
        import os

        token = os.environ["MODEL_TOKEN"]
        runtime = EnvRuntime(token)
        self.runtimes.append(runtime)
        return runtime


def test_orka_runtime_build_is_deferred_until_turn_env_is_available(monkeypatch):
    monkeypatch.delenv("MODEL_TOKEN", raising=False)
    factory = EnvFactory()
    app = create_orka_app(_spec(), factory, auth_token="test-token")
    assert factory.runtimes == []

    with TestClient(app) as client:
        turn_id = _create_turn(
            client,
            turnID="turn-env-build",
            input={"prompt": "hello", "contextRefs": [], "env": [{"name": "MODEL_TOKEN", "value": "turn-token"}]},
        )
        frames = _frames(client.get(f"/v1/turns/{turn_id}/events", headers=AUTH).text)
        assert frames[-1]["type"] == "TurnCompleted"
        assert frames[-1]["completed"]["result"] == "token=turn-token;run=turn-token"

    assert len(factory.runtimes) == 1
    assert factory.runtimes[0].entered == 1
    assert factory.runtimes[0].exited == 1
    assert "MODEL_TOKEN" not in __import__("os").environ


def test_orka_runtime_session_cache_is_bounded_and_closes_evicted_runtime(monkeypatch):
    monkeypatch.delenv("MODEL_TOKEN", raising=False)
    factory = EnvFactory()
    app = create_orka_app(_spec(), factory, auth_token="test-token", max_runtime_sessions=1)

    with TestClient(app) as client:
        first_id = _create_turn(
            client,
            turnID="turn-session-one",
            runtimeSessionID="runtime-session-one",
            correlationID="corr-one",
            input={"prompt": "one", "contextRefs": [], "env": [{"name": "MODEL_TOKEN", "value": "one-token"}]},
        )
        first_frames = _frames(client.get(f"/v1/turns/{first_id}/events", headers=AUTH).text)
        assert first_frames[-1]["completed"]["result"] == "token=one-token;run=one-token"

        second_id = _create_turn(
            client,
            turnID="turn-session-two",
            runtimeSessionID="runtime-session-two",
            correlationID="corr-two",
            input={"prompt": "two", "contextRefs": [], "env": [{"name": "MODEL_TOKEN", "value": "two-token"}]},
        )
        second_frames = _frames(client.get(f"/v1/turns/{second_id}/events", headers=AUTH).text)
        assert second_frames[-1]["completed"]["result"] == "token=two-token;run=two-token"
        assert len(factory.runtimes) == 2
        assert factory.runtimes[0].exited == 1
        assert factory.runtimes[1].exited == 0

    assert factory.runtimes[1].exited == 1


def test_orka_required_env_must_be_supplied_per_turn_or_process(monkeypatch):
    spec = AgentSpec.model_validate(
        {
            "abiVersion": "v0",
            "metadata": {"name": "orka-required-env"},
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
    monkeypatch.delenv("MODEL_TOKEN", raising=False)
    app = create_orka_app(spec, EchoFactory(), auth_token="test-token")

    with TestClient(app) as client:
        missing = client.post(
            "/v1/turns",
            json=_start_payload(turnID="turn-missing-env", input={"prompt": "hello", "contextRefs": [], "env": []}),
            headers=AUTH,
        )
        supplied = client.post(
            "/v1/turns",
            json=_start_payload(
                turnID="turn-supplied-env",
                input={"prompt": "hello", "contextRefs": [], "env": [{"name": "MODEL_TOKEN", "value": "turn-token"}]},
            ),
            headers=AUTH,
        )

    assert missing.status_code == 400
    assert "MODEL_TOKEN" in missing.text
    assert supplied.status_code == 202


def test_orka_required_env_rejects_empty_turn_override_even_when_process_has_value(monkeypatch):
    spec = AgentSpec.model_validate(
        {
            "abiVersion": "v0",
            "metadata": {"name": "orka-required-env"},
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
    monkeypatch.setenv("MODEL_TOKEN", "process-token")
    app = create_orka_app(spec, EchoFactory(), auth_token="test-token")

    with TestClient(app) as client:
        empty_override = client.post(
            "/v1/turns",
            json=_start_payload(
                turnID="turn-empty-env",
                input={"prompt": "hello", "contextRefs": [], "env": [{"name": "MODEL_TOKEN", "value": ""}]},
            ),
            headers=AUTH,
        )
        process_fallback = client.post(
            "/v1/turns",
            json=_start_payload(turnID="turn-process-env", input={"prompt": "hello", "contextRefs": [], "env": []}),
            headers=AUTH,
        )

    assert empty_override.status_code == 400
    assert "MODEL_TOKEN" in empty_override.text
    assert process_fallback.status_code == 202


def test_orka_runtime_session_rebuilds_when_turn_env_changes(monkeypatch):
    monkeypatch.delenv("MODEL_TOKEN", raising=False)
    factory = EnvFactory()
    app = create_orka_app(_spec(), factory, auth_token="test-token", max_runtime_sessions=2)

    with TestClient(app) as client:
        first_id = _create_turn(
            client,
            turnID="turn-rotated-env-1",
            runtimeSessionID="runtime-session-rotating",
            correlationID="corr-rotating-1",
            input={"prompt": "one", "contextRefs": [], "env": [{"name": "MODEL_TOKEN", "value": "one-token"}]},
        )
        first_frames = _frames(client.get(f"/v1/turns/{first_id}/events", headers=AUTH).text)
        assert first_frames[-1]["completed"]["result"] == "token=one-token;run=one-token"

        second_id = _create_turn(
            client,
            turnID="turn-rotated-env-2",
            runtimeSessionID="runtime-session-rotating",
            correlationID="corr-rotating-2",
            input={"prompt": "two", "contextRefs": [], "env": [{"name": "MODEL_TOKEN", "value": "two-token"}]},
        )
        second_frames = _frames(client.get(f"/v1/turns/{second_id}/events", headers=AUTH).text)
        assert second_frames[-1]["completed"]["result"] == "token=two-token;run=two-token"

    assert len(factory.runtimes) == 2
    assert factory.runtimes[0].exited == 1
    assert factory.runtimes[1].exited == 1




def test_orka_continue_is_unavailable_when_brokered_is_not_enabled():
    app = create_orka_app(_spec(), EchoFactory(), AUTH["authorization"].removeprefix("Bearer "))

    with TestClient(app) as client:
        turn_id = _create_turn(client, turnID="turn-observed-no-continue")
        cont = client.post(f"/v1/turns/{turn_id}/continue", json=_continue_payload(turnID=turn_id), headers=AUTH)
        frames = _frames(client.get(f"/v1/turns/{turn_id}/events", headers=AUTH).text)

    assert cont.status_code == 404
    assert "continuation is not enabled" in cont.text
    assert frames[-1]["type"] == "TurnCompleted"


def test_orka_brokered_read_capability_is_feature_gated():
    default_app = create_orka_app(_spec(), EchoFactory(), AUTH["authorization"].removeprefix("Bearer "))
    enabled_app = create_orka_app(_spec(), OfflineEchoRuntimeFactory(), "test-token", enable_brokered_read=True)
    write_app = create_orka_app(
        _spec(),
        OfflineEchoRuntimeFactory(),
        "test-token",
        enable_brokered_read=True,
        enable_brokered_write=True,
        enable_brokered_coordination=True,
    )

    with TestClient(default_app) as client:
        default_caps = client.get("/v1/capabilities").json()
    with TestClient(enabled_app) as client:
        enabled_caps = client.get("/v1/capabilities").json()
    with TestClient(write_app) as client:
        write_caps = client.get("/v1/capabilities").json()

    assert default_caps["toolExecutionModes"] == ["observed"]
    assert "brokeredToolClasses" not in default_caps
    assert "supportsContinuation" not in default_caps
    assert enabled_caps["toolExecutionModes"] == ["observed", "brokered"]
    assert enabled_caps["brokeredToolClasses"] == ["read"]
    assert enabled_caps["supportsContinuation"] is True
    assert write_caps["brokeredToolClasses"] == ["read", "write", "coordination"]


def test_orka_brokered_read_round_trip_emits_tool_frames_and_completes():
    app = create_orka_app(_spec(), OfflineEchoRuntimeFactory(), "test-token", enable_brokered_read=True)

    with TestClient(app) as client:
        turn_id = _create_turn(client, toolExecutionMode="brokered", input=_brokered_input())
        tool_frame = _wait_for_event_type(client, turn_id, "ToolCallRequested")
        assert tool_frame["toolName"] == "conformance_read"
        assert tool_frame["toolCallID"] == "tool-call-1"
        assert tool_frame["content"] == {"prompt": "hello"}

        cont = client.post(f"/v1/turns/{turn_id}/continue", json=_continue_payload(turnID=turn_id), headers=AUTH)
        duplicate = client.post(f"/v1/turns/{turn_id}/continue", json=_continue_payload(turnID=turn_id), headers=AUTH)
        conflicting_payload = _continue_payload(turnID=turn_id)
        conflicting_payload["toolResults"][0]["output"] = {"success": True, "data": {"answer": "different"}}
        conflicting = client.post(f"/v1/turns/{turn_id}/continue", json=conflicting_payload, headers=AUTH)
        frames = _frames(client.get(f"/v1/turns/{turn_id}/events", headers=AUTH).text)

    assert cont.status_code == 202, cont.text
    assert duplicate.status_code == 202, duplicate.text
    assert conflicting.status_code == 409, conflicting.text
    assert [frame["type"] for frame in frames] == ["TurnStarted", "ToolCallRequested", "ToolResultReceived", "RuntimeOutput", "TurnCompleted"]
    assert frames[2]["toolName"] == "conformance_read"
    assert frames[2]["toolCallID"] == "tool-call-1"
    assert frames[2]["content"] == {"success": True, "data": {"answer": "ok"}}
    assert "offline brokered echo" in frames[-1]["completed"]["result"]




def test_orka_brokered_write_round_trip_emits_tool_frames_and_completes():
    app = create_orka_app(_spec(), OfflineEchoRuntimeFactory(), "test-token", enable_brokered_write=True)

    with TestClient(app) as client:
        turn_id = _create_turn(client, toolExecutionMode="brokered", input=_brokered_write_input())
        tool_frame = _wait_for_event_type(client, turn_id, "ToolCallRequested")
        assert tool_frame["toolName"] == "conformance_write"
        assert tool_frame["toolCallID"] == "tool-call-1"

        cont = client.post(f"/v1/turns/{turn_id}/continue", json=_continue_payload(turnID=turn_id), headers=AUTH)
        frames = _frames(client.get(f"/v1/turns/{turn_id}/events", headers=AUTH).text)

    assert cont.status_code == 202, cont.text
    assert [frame["type"] for frame in frames] == ["TurnStarted", "ToolCallRequested", "ToolResultReceived", "RuntimeOutput", "TurnCompleted"]
    assert frames[1]["toolName"] == "conformance_write"
    assert frames[2]["toolName"] == "conformance_write"
    assert "offline brokered echo" in frames[-1]["completed"]["result"]


def test_orka_brokered_coordination_round_trip_emits_tool_frames_and_completes():
    app = create_orka_app(_spec(), OfflineEchoRuntimeFactory(), "test-token", enable_brokered_coordination=True)

    with TestClient(app) as client:
        turn_id = _create_turn(client, toolExecutionMode="brokered", input=_brokered_coordination_input())
        tool_frame = _wait_for_event_type(client, turn_id, "ToolCallRequested")
        assert tool_frame["toolName"] == "conformance_coordination"
        assert tool_frame["toolCallID"] == "tool-call-1"

        cont = client.post(f"/v1/turns/{turn_id}/continue", json=_continue_payload(turnID=turn_id), headers=AUTH)
        frames = _frames(client.get(f"/v1/turns/{turn_id}/events", headers=AUTH).text)

    assert cont.status_code == 202, cont.text
    assert [frame["type"] for frame in frames] == ["TurnStarted", "ToolCallRequested", "ToolResultReceived", "RuntimeOutput", "TurnCompleted"]
    assert frames[1]["toolName"] == "conformance_coordination"
    assert frames[2]["toolName"] == "conformance_coordination"
    assert "offline brokered echo" in frames[-1]["completed"]["result"]


def test_orka_brokered_continue_accepts_declined_result_without_output_or_error():
    app = create_orka_app(_spec(), OfflineEchoRuntimeFactory(), "test-token", enable_brokered_read=True)

    with TestClient(app) as client:
        turn_id = _create_turn(client, toolExecutionMode="brokered", input=_brokered_input())
        _wait_for_event_type(client, turn_id, "ToolCallRequested")
        declined = _continue_payload(turnID=turn_id)
        declined["toolResults"][0].pop("output")
        declined["toolResults"][0]["approved"] = False
        cont = client.post(f"/v1/turns/{turn_id}/continue", json=declined, headers=AUTH)
        frames = _frames(client.get(f"/v1/turns/{turn_id}/events", headers=AUTH).text)

    assert cont.status_code == 202, cont.text
    assert frames[2]["type"] == "ToolResultReceived"
    assert frames[2]["error"] == {"code": "ToolCallDenied", "message": "tool call was not approved", "retryable": False}
    assert frames[-1]["type"] == "TurnCompleted"
    assert "tool error" in frames[-1]["completed"]["result"]


def test_orka_brokered_continue_rejects_wrong_identity_or_unknown_tool_call():
    app = create_orka_app(_spec(), OfflineEchoRuntimeFactory(), "test-token", enable_brokered_read=True)

    with TestClient(app) as client:
        turn_id = _create_turn(client, toolExecutionMode="brokered", input=_brokered_input())
        _wait_for_event_type(client, turn_id, "ToolCallRequested")
        wrong_correlation = client.post(
            f"/v1/turns/{turn_id}/continue",
            json=_continue_payload(turnID=turn_id, correlationID="other-corr"),
            headers=AUTH,
        )
        unknown_tool = _continue_payload(turnID=turn_id)
        unknown_tool["toolResults"][0]["toolCallID"] = "other-tool"
        unknown_tool["toolResults"][0]["idempotencyKey"] = f"runtime-session-1:{turn_id}:other-tool"
        unknown = client.post(f"/v1/turns/{turn_id}/continue", json=unknown_tool, headers=AUTH)
        mixed_batch = _continue_payload(turnID=turn_id)
        mixed_batch["toolResults"].append(unknown_tool["toolResults"][0])
        mixed = client.post(f"/v1/turns/{turn_id}/continue", json=mixed_batch, headers=AUTH)
        denied_with_output = _continue_payload(turnID=turn_id)
        denied_with_output["toolResults"][0]["approved"] = False
        denied_output = client.post(f"/v1/turns/{turn_id}/continue", json=denied_with_output, headers=AUTH)
        assert all(event.type != "ToolResultReceived" for event in client.app.state.turns[turn_id].events)
        cancel = client.post(f"/v1/turns/{turn_id}/cancel", json=_cancel_payload(turnID=turn_id), headers=AUTH)

    assert wrong_correlation.status_code == 400
    assert "correlationID" in wrong_correlation.text
    assert unknown.status_code == 400
    assert "unknown toolCallID" in unknown.text
    assert mixed.status_code == 400
    assert "unknown toolCallID" in mixed.text
    assert denied_output.status_code == 400
    assert "approved is false" in denied_output.text
    assert cancel.status_code == 202


def test_orka_brokered_start_validates_safe_read_tool_schemas():
    app = create_orka_app(_spec(), OfflineEchoRuntimeFactory(), "test-token", enable_brokered_read=True)

    with TestClient(app) as client:
        unsupported_class = client.post(
            "/v1/turns",
            json=_start_payload(
                turnID="turn-brokered-write",
                toolExecutionMode="brokered",
                input=_brokered_input(tools=[{"name": "write_tool", "brokeredClass": "write", "parameters": {"type": "object"}}]),
            ),
            headers=AUTH,
        )
        bad_parameters = client.post(
            "/v1/turns",
            json=_start_payload(
                turnID="turn-brokered-bad-schema",
                toolExecutionMode="brokered",
                input=_brokered_input(tools=[{"name": "read_tool", "brokeredClass": "read", "parameters": []}]),
            ),
            headers=AUTH,
        )

    assert unsupported_class.status_code == 400
    assert "brokered write tools are not enabled" in unsupported_class.text
    assert bad_parameters.status_code == 400
    assert "parameters" in bad_parameters.text


class CapturingBrokeredRuntime:
    def __init__(self) -> None:
        self.tools: list[BrokeredToolDefinition] = []

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
        raise AssertionError("brokered mode must not call direct run()")

    async def run_brokered(self, request: RunRequest, tools: list[BrokeredToolDefinition], broker: ToolBroker) -> RunResult:
        self.tools = list(tools)
        result = await broker.request_tool(
            BrokeredToolCall(
                tool_call_id="tool-call-1",
                name=tools[0].name,
                arguments={"incident": "INC-1"},
                brokered_class="read",
            )
        )
        return RunResult(text=f"captured {result.output}")


class CapturingBrokeredFactory:
    def __init__(self) -> None:
        self.runtime = CapturingBrokeredRuntime()

    def supports_brokered_read(self) -> bool:
        return True

    def build_runtime(self, spec: AgentSpec) -> RuntimeSession:
        return self.runtime


def test_orka_brokered_runtime_receives_only_safe_tool_definition_fields():
    factory = CapturingBrokeredFactory()
    app = create_orka_app(_spec(), factory, "test-token", enable_brokered_read=True)

    with TestClient(app) as client:
        turn_id = _create_turn(
            client,
            toolExecutionMode="brokered",
            input=_brokered_input(
                tools=[
                    {
                        "name": "safe_lookup",
                        "description": "safe schema",
                        "brokeredClass": "read",
                        "parameters": {"type": "object", "properties": {"incident": {"type": "string"}}},
                        "url": "http://tool.default.svc.cluster.local",
                        "secretRef": {"name": "should-not-cross"},
                        "headers": {"Authorization": "should-not-cross"},
                    }
                ]
            ),
        )
        _wait_for_event_type(client, turn_id, "ToolCallRequested")
        client.post(f"/v1/turns/{turn_id}/continue", json=_continue_payload(turnID=turn_id), headers=AUTH)
        frames = _frames(client.get(f"/v1/turns/{turn_id}/events", headers=AUTH).text)

    assert frames[-1]["type"] == "TurnCompleted"
    assert factory.runtime.tools == [
        BrokeredToolDefinition(
            name="safe_lookup",
            description="safe schema",
            brokered_class="read",
            parameters={"properties": {"incident": {"type": "string"}}, "type": "object"},
        )
    ]
    assert not hasattr(factory.runtime.tools[0], "url")
    assert not hasattr(factory.runtime.tools[0], "secretRef")
    assert not hasattr(factory.runtime.tools[0], "headers")


class BadBrokeredRuntime(CapturingBrokeredRuntime):
    def __init__(self, call: BrokeredToolCall) -> None:
        super().__init__()
        self.call = call

    async def run_brokered(self, request: RunRequest, tools: list[BrokeredToolDefinition], broker: ToolBroker) -> RunResult:
        await broker.request_tool(self.call)
        return RunResult(text="should not complete")


class BadBrokeredFactory:
    def __init__(self, call: BrokeredToolCall) -> None:
        self.runtime = BadBrokeredRuntime(call)

    def supports_brokered_read(self) -> bool:
        return True

    def build_runtime(self, spec: AgentSpec) -> RuntimeSession:
        return self.runtime


def test_orka_brokered_unknown_tool_and_invalid_arguments_fail_safely():
    cases = [
        BrokeredToolCall(tool_call_id="tool-call-1", name="unknown", arguments={}, brokered_class="read"),
        BrokeredToolCall(tool_call_id="tool-call-1", name="conformance_read", arguments="bad", brokered_class="read"),  # type: ignore[arg-type]
    ]
    for idx, call in enumerate(cases):
        app = create_orka_app(_spec(), BadBrokeredFactory(call), "test-token", enable_brokered_read=True)
        with TestClient(app) as client:
            turn_id = _create_turn(
                client,
                turnID=f"turn-bad-brokered-{idx}",
                toolExecutionMode="brokered",
                input=_brokered_input(),
            )
            frames = _frames(client.get(f"/v1/turns/{turn_id}/events", headers=AUTH).text)
        assert [frame["type"] for frame in frames] == ["TurnStarted", "TurnFailed"]
        assert frames[-1]["failed"]["reason"] in {"UnknownBrokeredTool", "InvalidToolArguments"}






def test_orka_brokered_pending_tool_wait_is_bounded_by_deadline():
    app = create_orka_app(_spec(), OfflineEchoRuntimeFactory(), "test-token", enable_brokered_write=True)
    deadline = (datetime.now(UTC) + timedelta(milliseconds=150)).isoformat().replace("+00:00", "Z")

    with TestClient(app) as client:
        turn_id = _create_turn(
            client,
            turnID="turn-brokered-deadline",
            toolExecutionMode="brokered",
            input=_brokered_write_input(),
            deadline=deadline,
        )
        _wait_for_event_type(client, turn_id, "ToolCallRequested")
        frames = _frames(client.get(f"/v1/turns/{turn_id}/events", headers=AUTH).text)
        accepted_after_terminal = client.post(
            "/v1/turns",
            json=_start_payload(turnID="turn-after-brokered-deadline", input={"prompt": "after", "contextRefs": [], "env": []}),
            headers=AUTH,
        )

    assert frames[-1]["type"] == "TurnFailed"
    assert frames[-1]["failed"]["reason"] == "DeadlineExceeded"
    assert accepted_after_terminal.status_code == 202


def test_orka_brokered_late_continue_after_cancel_is_rejected():
    app = create_orka_app(_spec(), OfflineEchoRuntimeFactory(), "test-token", enable_brokered_write=True)

    with TestClient(app) as client:
        turn_id = _create_turn(client, toolExecutionMode="brokered", input=_brokered_write_input())
        _wait_for_event_type(client, turn_id, "ToolCallRequested")
        cancel = client.post(f"/v1/turns/{turn_id}/cancel", json=_cancel_payload(turnID=turn_id), headers=AUTH)
        frames = _frames(client.get(f"/v1/turns/{turn_id}/events", headers=AUTH).text)
        late = client.post(f"/v1/turns/{turn_id}/continue", json=_continue_payload(turnID=turn_id), headers=AUTH)

    assert cancel.status_code == 202
    assert frames[-1]["type"] == "TurnCancelled"
    assert late.status_code == 409
    assert "already terminal" in late.text


def test_orka_brokered_mode_does_not_advertise_or_fall_back_to_direct_runtime_run():
    with pytest.raises(ValueError, match="requires a runtime factory that supports brokered tools"):
        create_orka_app(_spec(), EchoFactory(), "test-token", enable_brokered_read=True)
