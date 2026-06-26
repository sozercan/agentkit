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


def test_foundry_responses_rejects_streaming():
    app = create_foundry_app(_spec(), EchoFactory())
    with TestClient(app) as client:
        resp = client.post("/responses", json={"input": "hi", "stream": True})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "stream_unsupported"


def test_foundry_protocols_reject_non_object_json():
    app = create_foundry_app(_spec(), EchoFactory())
    with TestClient(app) as client:
        inv = client.post("/invocations", json=[])
        resp = client.post("/responses", json=[])

    assert inv.status_code == 400
    assert "JSON object" in inv.text
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_request"
