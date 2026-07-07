from __future__ import annotations

import pytest

from agentkit_serve import agent_factory
from agentkit_serve_common.config import AgentSpec


def _spec_data() -> dict:
    return {
        "abiVersion": "v0",
        "metadata": {"name": "x"},
        "model": {"provider": "openai-compatible", "baseURL": "https://api.openai.com/v1", "name": "gpt-4o-mini"},
        "instructions": "hi",
        "tools": [],
        "expose": {"openai": True, "port": 8080},
    }


def test_pydantic_rejects_model_workload_identity_auth():
    data = _spec_data()
    data["model"]["auth"] = {"type": "workload-identity-token", "audience": "https://ai.azure.com/.default"}
    spec = AgentSpec.model_validate(data)

    with pytest.raises(agent_factory.AgentBuildError, match="model.auth"):
        agent_factory.build_runtime(spec)


def test_pydantic_rejects_context_providers():
    data = _spec_data()
    data["context"] = {"providers": [{"type": "skills", "source": "filesystem", "path": "/agent/skills"}]}
    spec = AgentSpec.model_validate(data)

    with pytest.raises(agent_factory.AgentBuildError, match="context providers"):
        agent_factory.build_runtime(spec)


def test_pydantic_orka_offline_echo_bypasses_provider_runtime(monkeypatch):
    spec = AgentSpec.model_validate(_spec_data())
    monkeypatch.setenv("AGENTKIT_PROTOCOL", "orka")
    monkeypatch.setenv("AGENTKIT_ORKA_OFFLINE_ECHO", "1")

    runtime = agent_factory.build_runtime(spec)

    assert runtime.__class__.__name__ == "OfflineEchoRuntime"


def test_pydantic_orka_offline_echo_completes_without_provider(monkeypatch):
    from fastapi.testclient import TestClient

    from agentkit_serve_common.orka import ORKA_HARNESS_VERSION, create_orka_app

    spec = AgentSpec.model_validate(_spec_data())
    monkeypatch.setenv("AGENTKIT_PROTOCOL", "orka")
    monkeypatch.setenv("AGENTKIT_ORKA_OFFLINE_ECHO", "1")
    app = create_orka_app(spec, agent_factory, auth_token="example")
    payload = {
        "version": ORKA_HARNESS_VERSION,
        "namespace": "default",
        "taskName": "task-1",
        "sessionName": "session-1",
        "runtimeSessionID": "runtime-session-1",
        "turnID": "offline-turn-1",
        "correlationID": "corr-1",
        "deadline": "2099-01-01T00:00:00Z",
        "authIdentity": {"subject": "system:serviceaccount:default:orka"},
        "input": {"prompt": "hello", "contextRefs": [], "env": []},
        "toolExecutionMode": "observed",
        "metadata": {},
    }

    with TestClient(app) as client:
        start = client.post("/v1/turns", json=payload, headers={"authorization": "Bearer example"})
        events = client.get("/v1/turns/offline-turn-1/events", headers={"authorization": "Bearer example"})

    assert start.status_code == 202
    assert events.status_code == 200
    assert "offline echo: hello" in events.text
