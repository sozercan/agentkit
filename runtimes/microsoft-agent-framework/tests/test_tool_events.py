from __future__ import annotations

import asyncio
from dataclasses import asdict

import pytest
from agent_framework import Agent, BaseChatClient, ChatResponse, Content, FunctionInvocationLayer, Message, tool

from agentkit_serve import agent_factory
from agentkit_serve_common.conversation import RunRequest, ToolCallEvent
from agentkit_serve_common.runtime import AgentRunError


class _ToolClient(FunctionInvocationLayer, BaseChatClient):
    def __init__(self, calls=1):
        super().__init__()
        self.remaining = calls

    async def _inner_get_response(self, *, messages, stream, options, **kwargs):
        if self.remaining:
            self.remaining -= 1
            content = Content.from_function_call(
                call_id=f"private-provider-id-{self.remaining}", name="probe", arguments={}
            )
        else:
            content = Content.from_text("done")
        return ChatResponse(messages=[Message(role="assistant", contents=[content])])


def _agent(probe, *, calls=1):
    return Agent(client=_ToolClient(calls), tools=[tool(probe, name="probe", description="Controlled fixture.")])


def test_real_tool_execution_awaits_redacted_start_and_result():
    async def exercise():
        events: list[ToolCallEvent] = []
        order = []

        async def probe():
            order.append("side-effect")
            return {"private-payload": ["héllo 世界 🌍", {"nested": True}]}

        async def observe(event):
            await asyncio.sleep(0)
            events.append(event)
            order.append(event.status)

        async with _agent(probe) as agent:
            result = await agent_factory.run_agent(agent, RunRequest("probe", on_tool_event=observe))
        assert result.text == "done"
        assert order == ["in_progress", "side-effect", "completed"]
        assert events[0].tool_call_id == events[1].tool_call_id
        assert [event.tool_name for event in events] == ["probe", "probe"]
        assert all(set(asdict(event)) == {"tool_call_id", "tool_name", "status"} for event in events)
        assert "private" not in str(events)

    asyncio.run(exercise())


def test_failed_tool_emits_failure_before_successful_retry():
    async def exercise():
        events = []
        calls = 0

        async def probe():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("private error details")
            return "private successful result"

        async def observe(event):
            events.append(event)

        async with _agent(probe, calls=2) as agent:
            result = await agent_factory.run_agent(agent, RunRequest("recover", on_tool_event=observe))
        assert result.text == "done" and calls == 2
        assert [event.status for event in events] == ["in_progress", "failed", "in_progress", "completed"]
        assert events[0].tool_call_id == events[1].tool_call_id
        assert events[2].tool_call_id == events[3].tool_call_id
        assert events[0].tool_call_id != events[2].tool_call_id
        assert "private" not in str(events)

    asyncio.run(exercise())


def test_cancellation_stops_real_tool_and_leaves_terminal_to_acp():
    async def exercise():
        events = []
        started = asyncio.Event()
        stopped = asyncio.Event()

        async def probe():
            started.set()
            try:
                await asyncio.Future()
            finally:
                stopped.set()

        async def observe(event):
            events.append(event)

        async with _agent(probe) as agent:
            run = asyncio.create_task(agent_factory.run_agent(agent, RunRequest("delay", on_tool_event=observe)))
            try:
                await asyncio.wait_for(started.wait(), 3)
            finally:
                run.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(run, 3)
        assert stopped.is_set()
        assert [event.status for event in events] == ["in_progress"]

    asyncio.run(exercise())


@pytest.mark.parametrize("failure_status", ["in_progress", "completed", "failed"])
def test_observer_failure_aborts_run_without_exposing_its_error(failure_status, caplog):
    async def exercise():
        calls = []

        async def probe():
            calls.append("called")
            if failure_status == "failed":
                raise RuntimeError("synthetic tool failure")
            return "private payload"

        async def observe(event):
            if event.status == failure_status:
                raise RuntimeError("private observer failure")

        async with _agent(probe) as agent:
            with pytest.raises(AgentRunError, match="tool lifecycle observer failed"):
                await agent_factory.run_agent(agent, RunRequest("probe", on_tool_event=observe))
        assert calls == ([] if failure_status == "in_progress" else ["called"])
        assert "private observer failure" not in caplog.text

    asyncio.run(exercise())


def test_no_observer_preserves_plain_run_contract():
    async def exercise():
        calls = []

        class PlainAgent:
            async def run(self, messages, *, session):
                calls.append((messages, session))
                return type("Result", (), {"text": "plain"})()

        result = await agent_factory.run_agent(PlainAgent(), RunRequest("plain prompt"))
        assert result.text == "plain" and len(calls) == 1
        assert calls[0][0][-1].text == "plain prompt" and calls[0][1] is None

    asyncio.run(exercise())
