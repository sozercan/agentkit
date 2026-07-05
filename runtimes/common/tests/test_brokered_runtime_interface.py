from __future__ import annotations

import asyncio
from types import TracebackType
from typing import Any

from agentkit_serve_common.conversation import RunRequest
from agentkit_serve_common.runtime import (
    BrokeredRuntimeSession,
    BrokeredToolCall,
    BrokeredToolDefinition,
    BrokeredToolResult,
    RunResult,
    RuntimeSession,
    ToolBroker,
)


class RecordingBroker:
    def __init__(self) -> None:
        self.calls: list[BrokeredToolCall] = []

    async def request_tool(self, call: BrokeredToolCall) -> BrokeredToolResult:
        self.calls.append(call)
        return BrokeredToolResult(tool_call_id=call.tool_call_id, approved=True, output={"answer": "ok"})


class FakeBrokeredRuntime:
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
        return RunResult(text=request.prompt)

    async def run_brokered(
        self,
        request: RunRequest,
        tools: list[BrokeredToolDefinition],
        broker: ToolBroker,
    ) -> RunResult:
        tool = tools[0]
        result = await broker.request_tool(
            BrokeredToolCall(
                tool_call_id="tool-call-1",
                name=tool.name,
                arguments={"prompt": request.prompt},
                brokered_class=tool.brokered_class,
            )
        )
        return RunResult(text=f"approved={result.approved}; output={result.output}")


def test_brokered_runtime_interface_is_framework_neutral() -> None:
    async def exercise() -> tuple[FakeBrokeredRuntime, RecordingBroker, RunResult]:
        runtime = FakeBrokeredRuntime()
        broker = RecordingBroker()
        result = await runtime.run_brokered(
            RunRequest(prompt="hello"),
            [
                BrokeredToolDefinition(
                    name="conformance_read",
                    description="safe schema only",
                    brokered_class="read",
                    parameters={"type": "object"},
                )
            ],
            broker,
        )
        return runtime, broker, result

    runtime, broker, result = asyncio.run(exercise())

    assert isinstance(runtime, BrokeredRuntimeSession)
    assert result.text == "approved=True; output={'answer': 'ok'}"
    assert broker.calls == [
        BrokeredToolCall(
            tool_call_id="tool-call-1",
            name="conformance_read",
            arguments={"prompt": "hello"},
            brokered_class="read",
        )
    ]


def test_brokered_tool_definition_is_schema_only() -> None:
    tool = BrokeredToolDefinition(
        name="safe_lookup",
        description="No URLs or credentials are represented in the neutral type.",
        brokered_class="read",
        parameters={"type": "object", "properties": {"incident": {"type": "string"}}},
    )

    serialized: dict[str, Any] = {
        "name": tool.name,
        "description": tool.description,
        "brokered_class": tool.brokered_class,
        "parameters": dict(tool.parameters),
    }
    forbidden = {"url", "headers", "secretRef", "token", "txToken", "kubeconfig"}
    assert forbidden.isdisjoint(serialized)


def test_offline_echo_runtime_requests_delegate_then_wait_for_coordination_tools(monkeypatch) -> None:
    from agentkit_serve_common.runtime import OfflineEchoRuntime

    class SequencedBroker:
        def __init__(self) -> None:
            self.calls: list[BrokeredToolCall] = []

        async def request_tool(self, call: BrokeredToolCall) -> BrokeredToolResult:
            self.calls.append(call)
            if call.name == "delegate_task":
                return BrokeredToolResult(tool_call_id=call.tool_call_id, approved=True, output={"taskName": "child-1", "status": "created"})
            if call.name == "wait_for_tasks":
                return BrokeredToolResult(tool_call_id=call.tool_call_id, approved=True, output={"completed": True, "results": []})
            raise AssertionError(call.name)

    async def exercise() -> SequencedBroker:
        monkeypatch.setenv("AGENTKIT_ORKA_OFFLINE_DELEGATE_AGENT", "worker-agent")
        broker = SequencedBroker()
        runtime = OfflineEchoRuntime()
        result = await runtime.run_brokered(
            RunRequest(prompt="coordinate work"),
            [
                BrokeredToolDefinition(name="delegate_task", description="delegate", brokered_class="coordination", parameters={}),
                BrokeredToolDefinition(name="wait_for_tasks", description="wait", brokered_class="coordination", parameters={}),
            ],
            broker,
        )
        assert "offline coordinator delegated child-1" in result.text
        return broker

    broker = asyncio.run(exercise())

    assert [call.name for call in broker.calls] == ["delegate_task", "wait_for_tasks"]
    assert broker.calls[0].arguments == {"agent": "worker-agent", "prompt": "Offline child proof for: coordinate work"}
    assert broker.calls[1].arguments == {"tasks": ["child-1"], "timeout": "2m"}


def test_offline_echo_runtime_can_wait_for_existing_task_from_prompt() -> None:
    from agentkit_serve_common.runtime import OfflineEchoRuntime

    class WaitBroker:
        def __init__(self) -> None:
            self.calls: list[BrokeredToolCall] = []

        async def request_tool(self, call: BrokeredToolCall) -> BrokeredToolResult:
            self.calls.append(call)
            return BrokeredToolResult(tool_call_id=call.tool_call_id, approved=True, output={"completed": True})

    async def exercise() -> WaitBroker:
        broker = WaitBroker()
        runtime = OfflineEchoRuntime()
        result = await runtime.run_brokered(
            RunRequest(prompt="please wait_for_tasks:child-123 now"),
            [BrokeredToolDefinition(name="wait_for_tasks", description="wait", brokered_class="coordination", parameters={})],
            broker,
        )
        assert "offline coordinator waited for child-123" in result.text
        return broker

    broker = asyncio.run(exercise())
    assert broker.calls[0].arguments == {"tasks": ["child-123"], "timeout": "30s"}
