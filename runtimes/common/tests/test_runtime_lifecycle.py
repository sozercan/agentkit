from __future__ import annotations

import asyncio
from types import TracebackType

from agentkit_serve_common.config import AgentSpec
from agentkit_serve_common.conversation import ConversationTurn, RunRequest
from agentkit_serve_common.runtime import RunResult, RuntimeSession
from agentkit_serve_common.server import create_app


def _spec() -> AgentSpec:
    return AgentSpec.model_validate(
        {
            "abiVersion": "v0",
            "metadata": {"name": "test-agent"},
            "model": {
                "provider": "openai-compatible",
                "baseURL": "https://api.openai.com/v1",
                "name": "gpt-4o-mini",
                "apiKeyEnv": "OPENAI_API_KEY",
            },
            "instructions": "Be helpful.",
            "tools": [],
            "expose": {"openai": True, "port": 8080},
        }
    )


class RecordingRuntime:
    def __init__(self) -> None:
        self.entered = 0
        self.exited = 0
        self.requests: list[RunRequest] = []

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
        self.requests.append(request)
        return RunResult(text=f"echo: {request.prompt}")


class RecordingFactory:
    def __init__(self) -> None:
        self.runtime = RecordingRuntime()
        self.spec: AgentSpec | None = None

    def build_runtime(self, spec: AgentSpec) -> RuntimeSession:
        self.spec = spec
        return self.runtime


def test_server_lifespan_enters_runtime_session():
    factory = RecordingFactory()
    app = create_app(_spec(), factory)

    async def exercise_lifespan() -> None:
        async with app.router.lifespan_context(app):
            assert app.state.runtime is factory.runtime
            assert factory.runtime.entered == 1
            result = await app.state.runtime.run(
                RunRequest(prompt="hello", history=(ConversationTurn("system", "be terse"),))
            )
            assert result.text == "echo: hello"

    asyncio.run(exercise_lifespan())

    assert factory.spec is not None
    assert factory.runtime.exited == 1
    assert factory.runtime.requests == [
        RunRequest(prompt="hello", history=(ConversationTurn("system", "be terse"),))
    ]
