from __future__ import annotations

import asyncio
from dataclasses import asdict
from types import SimpleNamespace

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool, ToolException

from agentkit_serve import agent_factory
from agentkit_serve_common.conversation import RunRequest, ToolCallEvent
from agentkit_serve_common.runtime import AgentRunError


class _ToolModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def _runtime(probe, *, calls=1, handle_tool_error=True):
    messages = [
        AIMessage(content="", tool_calls=[{"name": "probe", "args": {}, "id": f"private-provider-id-{i}"}])
        for i in range(calls)
    ]
    messages.append(AIMessage(content="done"))
    tool = StructuredTool.from_function(
        coroutine=probe, name="probe", description="Controlled fixture.", handle_tool_error=handle_tool_error
    )
    return SimpleNamespace(graph=create_agent(model=_ToolModel(messages=iter(messages)), tools=[tool]))


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

        result = await agent_factory.run_agent(_runtime(probe), RunRequest("probe", on_tool_event=observe))
        assert result.text == "done"
        assert order == ["in_progress", "side-effect", "completed"]
        assert events[0].tool_call_id == events[1].tool_call_id
        assert [event.tool_name for event in events] == ["probe", "probe"]
        assert all(set(asdict(event)) == {"tool_call_id", "tool_name", "status"} for event in events)
        assert "private" not in str(events)

    asyncio.run(exercise())


def test_handled_tool_error_emits_failure_before_successful_retry():
    async def exercise():
        events = []
        calls = 0

        async def probe():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ToolException("private error details")
            return "private successful result"

        async def observe(event):
            events.append(event)

        result = await agent_factory.run_agent(_runtime(probe, calls=2), RunRequest("recover", on_tool_event=observe))
        assert result.text == "done" and calls == 2
        assert [event.status for event in events] == ["in_progress", "failed", "in_progress", "completed"]
        assert events[0].tool_call_id == events[1].tool_call_id
        assert events[2].tool_call_id == events[3].tool_call_id
        assert events[0].tool_call_id != events[2].tool_call_id
        assert "private" not in str(events)

    asyncio.run(exercise())


def test_unhandled_tool_error_emits_failure_and_propagates():
    async def exercise():
        events = []

        async def probe():
            raise RuntimeError("private error details")

        async def observe(event):
            events.append(event)

        with pytest.raises(AgentRunError):
            await agent_factory.run_agent(_runtime(probe), RunRequest("fail", on_tool_event=observe))
        assert [event.status for event in events] == ["in_progress", "failed"]
        assert events[0].tool_call_id == events[1].tool_call_id
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

        run = asyncio.create_task(agent_factory.run_agent(_runtime(probe), RunRequest("delay", on_tool_event=observe)))
        try:
            await asyncio.wait_for(started.wait(), 3)
        finally:
            run.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(run, 3)
        assert stopped.is_set()
        assert [event.status for event in events] == ["in_progress"]

    asyncio.run(exercise())


@pytest.mark.parametrize("failure_status", ["in_progress", "completed"])
def test_observer_failure_aborts_run_without_exposing_its_error(failure_status, caplog):
    async def exercise():
        calls = []

        async def probe():
            calls.append("called")
            return "private payload"

        async def observe(event):
            if event.status == failure_status:
                raise RuntimeError("private observer failure")

        with pytest.raises(AgentRunError, match="tool lifecycle observer failed"):
            await agent_factory.run_agent(_runtime(probe), RunRequest("probe", on_tool_event=observe))
        assert calls == ([] if failure_status == "in_progress" else ["called"])
        assert "private observer failure" not in caplog.text

    asyncio.run(exercise())


def test_no_observer_preserves_plain_ainvoke_contract():
    async def exercise():
        calls = []

        class PlainGraph:
            async def ainvoke(self, state):
                calls.append(state)
                return {"messages": [AIMessage(content="plain")]}

        result = await agent_factory.run_agent(SimpleNamespace(graph=PlainGraph()), RunRequest("plain prompt"))
        assert result.text == "plain" and len(calls) == 1
        assert calls[0]["messages"][-1].content == "plain prompt"

    asyncio.run(exercise())
