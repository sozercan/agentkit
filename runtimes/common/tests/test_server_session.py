from __future__ import annotations

from types import TracebackType

from fastapi.testclient import TestClient

from agentkit_serve_common.config import AgentSpec
from agentkit_serve_common.conversation import RunRequest
from agentkit_serve_common.runtime import RunResult, RuntimeSession
from agentkit_serve_common.server import create_app


def _spec() -> AgentSpec:
    return AgentSpec.model_validate({
        "abiVersion": "v0",
        "metadata": {"name": "server-test"},
        "model": {"provider": "openai-compatible", "baseURL": "https://api.openai.com/v1", "name": "gpt-4o-mini"},
        "instructions": "hi",
        "tools": [],
        "expose": {"openai": True, "port": 8080},
    })


class Runtime:
    def __init__(self):
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
        return RunResult(text="ok")


class Factory:
    def __init__(self):
        self.runtime = Runtime()

    def build_runtime(self, spec: AgentSpec) -> RuntimeSession:
        return self.runtime


def test_openai_facade_forwards_agentkit_session_header():
    factory = Factory()
    app = create_app(_spec(), factory)
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}]},
            headers={"X-AgentKit-Session-Id": "local-session"},
        )

    assert resp.status_code == 200
    assert factory.runtime.requests[0].session_id == "local-session"
