from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi.testclient import TestClient
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import AbstractToolset

from agentkit_serve import agent_factory
from agentkit_serve_common.config import AgentSpec
from agentkit_serve_common.orka import ORKA_HARNESS_VERSION, create_orka_app
from agentkit_serve_common.runtime import RuntimeSession

AUTH = {"authorization": "Bearer mock-token"}


def _spec() -> AgentSpec:
    return AgentSpec.model_validate(
        {
            "abiVersion": "v0",
            "metadata": {"name": "pydantic-orka-lifecycle"},
            "model": {
                "provider": "openai-compatible",
                "baseURL": "https://api.openai.com/v1",
                "name": "gpt-4o-mini",
            },
            "instructions": "Be helpful.",
            "tools": [],
            "env": [],
            "expose": {"openai": True, "port": 8080},
        }
    )


def _start_payload(*, turn_id: str, runtime_session_id: str, correlation_id: str, deadline: str) -> dict[str, Any]:
    return {
        "version": ORKA_HARNESS_VERSION,
        "namespace": "default",
        "taskName": "task-1",
        "sessionName": "session-1",
        "runtimeSessionID": runtime_session_id,
        "turnID": turn_id,
        "correlationID": correlation_id,
        "deadline": deadline,
        "authIdentity": {"subject": "system:serviceaccount:default:orka"},
        "input": {"prompt": turn_id, "contextRefs": [], "env": []},
        "toolExecutionMode": "observed",
        "metadata": {},
    }


def _frames(response_text: str) -> list[dict[str, Any]]:
    return [json.loads(line.removeprefix("data: ")) for line in response_text.splitlines() if line.startswith("data: ")]


class SlowExitToolset(AbstractToolset[None]):
    def __init__(self, toolset_id: str, *, close_delay: float, slow_close_call: int | None = None) -> None:
        self.toolset_id = toolset_id
        self.close_delay = close_delay
        self.slow_close_call = slow_close_call
        self.entered = 0
        self.close_calls = 0
        self.close_completed = 0
        self.close_cancelled = 0

    @property
    def id(self) -> str:
        return self.toolset_id

    async def __aenter__(self) -> SlowExitToolset:
        self.entered += 1
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        self.close_calls += 1
        try:
            if self.close_calls == self.slow_close_call:
                await asyncio.sleep(self.close_delay)
        except asyncio.CancelledError:
            self.close_cancelled += 1
            raise
        self.close_completed += 1
        return None

    async def get_tools(self, ctx: Any) -> dict[str, Any]:
        return {}

    async def call_tool(self, name: str, tool_args: dict[str, Any], ctx: Any, tool: Any) -> Any:
        raise AssertionError("the lifecycle reproduction exposes no callable tools")


class PydanticAgentFactory:
    def __init__(self, *, first_close_delay: float = 1.0) -> None:
        self.first_close_delay = first_close_delay
        self.toolsets: list[SlowExitToolset] = []

    def build_runtime(self, spec: AgentSpec) -> RuntimeSession:
        del spec
        index = len(self.toolsets)
        toolset = SlowExitToolset(
            f"toolset-{index}",
            close_delay=self.first_close_delay if index == 0 else 0,
            # Pydantic enters/exits the toolset for the first run, then exits it
            # again when the long-lived Agent context is evicted from Orka.
            slow_close_call=2 if index == 0 else None,
        )
        self.toolsets.append(toolset)
        agent = Agent(TestModel(custom_output_text=f"runtime-{index}"), instructions="Be helpful.", toolsets=[toolset])
        return agent_factory.PydanticRuntime(agent)


def test_orka_cache_limit_preserves_real_pydantic_agent_toolset_cleanup_after_deadline():
    factory = PydanticAgentFactory()
    app = create_orka_app(_spec(), factory, auth_token=AUTH["authorization"].removeprefix("Bearer "), max_runtime_sessions=1)
    long_deadline = (datetime.now(UTC) + timedelta(minutes=1)).isoformat().replace("+00:00", "Z")

    with TestClient(app) as client:
        first = client.post(
            "/v1/turns",
            json=_start_payload(
                turn_id="turn-pydantic-one",
                runtime_session_id="runtime-session-one",
                correlation_id="corr-one",
                deadline=long_deadline,
            ),
            headers=AUTH,
        )
        assert first.status_code == 202
        first_frames = _frames(client.get("/v1/turns/turn-pydantic-one/events", headers=AUTH).text)
        assert first_frames[-1]["type"] == "TurnCompleted"

        short_deadline = (datetime.now(UTC) + timedelta(milliseconds=500)).isoformat().replace("+00:00", "Z")
        second = client.post(
            "/v1/turns",
            json=_start_payload(
                turn_id="turn-pydantic-two",
                runtime_session_id="runtime-session-two",
                correlation_id="corr-two",
                deadline=short_deadline,
            ),
            headers=AUTH,
        )
        assert second.status_code == 202
        second_frames = _frames(client.get("/v1/turns/turn-pydantic-two/events", headers=AUTH).text)
        assert second_frames[-1]["type"] == "TurnFailed"
        assert second_frames[-1]["failed"]["reason"] == "DeadlineExceeded"

    assert len(factory.toolsets) == 1
    assert factory.toolsets[0].entered >= 2
    assert factory.toolsets[0].close_calls == factory.toolsets[0].entered
    assert factory.toolsets[0].close_cancelled == 0
    assert factory.toolsets[0].close_completed == factory.toolsets[0].close_calls
