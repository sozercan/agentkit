from __future__ import annotations

import asyncio
import json
import logging
from contextlib import AsyncExitStack
from dataclasses import asdict
from unittest import mock

import httpx
import pytest
from agent_framework import AgentSession, BaseChatClient, ChatResponse, Content, FunctionInvocationLayer, Message
from anyio import ClosedResourceError
from mcp import types
from mcp.shared.exceptions import McpError

from agentkit_serve import agent_factory
from agentkit_serve_common.config import AgentSpec, ToolSpec
from agentkit_serve_common.conversation import RunRequest
from agentkit_serve_common.runtime import AgentRunError


_PRIVATE = "PRIVATE_MCP_DIAGNOSTIC_91f42"
_PAYLOAD = {"text": "héllo 世界 🌍", "nested": [42, True, None]}


class _Session:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.calls = []

    async def list_tools(self, **kwargs):
        return types.ListToolsResult(tools=[types.Tool(
            name="probe",
            inputSchema={"type": "object", "properties": {"payload": {"type": "object"}}},
            _meta={"fixture.example/authority": "frozen-metadata"},
        )])

    async def call_tool(self, name, *, arguments, meta):
        self.calls.append({"name": name, "arguments": arguments, "meta": meta})
        outcome = next(self.outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        if callable(outcome):
            return await outcome()
        return outcome


class _Client(FunctionInvocationLayer, BaseChatClient):
    def __init__(self, function_name, *, calls=1):
        super().__init__()
        self.function_name = function_name
        self.tool_calls = calls
        self.requests = {}

    async def _inner_get_response(self, *, messages, stream, options, **kwargs):
        prompt = next(message.text for message in reversed(messages) if message.role == "user")
        requests = self.requests.setdefault(prompt, [])
        requests.append(messages)
        number = len(requests)
        if number <= self.tool_calls:
            item = Content.from_function_call(
                call_id=f"private-provider-id-{prompt}-{number}", name=self.function_name,
                arguments={"payload": {**_PAYLOAD, "prompt": prompt}, "undeclared": "must-be-filtered"},
            )
        else:
            item = Content.from_text("done")
        return ChatResponse(messages=[Message(role="assistant", contents=[item])])


def _success():
    return types.CallToolResult(content=[types.TextContent(type="text", text=json.dumps(_PAYLOAD))])


def _failure(kind):
    if kind == "jsonrpc":
        return McpError(types.ErrorData(code=-32000, message=_PRIVATE))
    if kind == "closed":
        return ClosedResourceError(_PRIVATE)
    if kind == "terminated":
        return McpError(types.ErrorData(code=-32000, message="session terminated: " + _PRIVATE))
    if kind == "auth":
        return httpx.HTTPStatusError(_PRIVATE, request=httpx.Request("POST", "https://example.invalid"),
                                     response=httpx.Response(403))
    if kind == "timeout":
        return httpx.ReadTimeout(_PRIVATE)
    if kind == "malformed":
        try:
            types.CallToolResult.model_validate({"content": _PRIVATE})
        except ValueError as exc:
            return exc
    raise AssertionError("unknown controlled failure")


async def _setup(stack, monkeypatch, outcomes, *, transport="streamable-http", calls=1):
    monkeypatch.setenv("TEST_MCP_URL", "http://example.invalid/mcp")
    tool = (
        ToolSpec(name="fixture", type="mcp", url_env="TEST_MCP_URL", transport=transport)
        if transport == "streamable-http"
        else ToolSpec(name="fixture", command=["unused-fixture"])
    )
    server = agent_factory.build_tool(tool, stack=stack)
    server.session = session = _Session(outcomes)
    server._supports_tools = True
    server._ping_available = False
    server.connect = mock.AsyncMock()
    await server.load_tools()
    client = _Client(server.functions[0].name, calls=calls)
    spec = AgentSpec.model_validate({
        "abiVersion": "v0", "metadata": {"name": "test-package"},
        "model": {"provider": "openai-compatible", "name": "test-model", "baseURL": "http://model.invalid/v1"},
        "instructions": "Follow the user's request.", "tools": [tool.model_dump()],
        "expose": {"openai": True, "port": 8080},
    })
    # Keep the real SDK-generated MCP callable, but supply its already-loaded
    # function to avoid opening a network transport for the in-memory session.
    with mock.patch.object(agent_factory, "build_tool", return_value=server.functions[0]):
        agent = await stack.enter_async_context(agent_factory.build_agent(spec, client=client))
    return agent, server, session, client


@pytest.mark.parametrize("transport", ["streamable-http", "stdio"])
@pytest.mark.parametrize("kind", ["jsonrpc", "closed", "terminated", "auth", "timeout", "malformed"])
@pytest.mark.parametrize("observed", [True, False])
def test_mcp_protocol_failure_is_fatal_without_retry_or_model_continuation(
    monkeypatch, caplog, transport, kind, observed
):
    async def exercise():
        events = []

        async def observe(event):
            events.append(event)

        async with AsyncExitStack() as stack:
            agent, server, session, client = await _setup(stack, monkeypatch, [_failure(kind), _success()],
                                                         transport=transport)
            with pytest.raises(AgentRunError, match="^MCP tool protocol failed$") as caught:
                await agent_factory.run_agent(agent, RunRequest("fatal", on_tool_event=observe if observed else None))
            assert len(session.calls) == 1 and len(client.requests["fatal"]) == 1
            assert server.connect.await_count == 0
            assert session.calls[0] == {
                "name": "probe", "arguments": {"payload": {**_PAYLOAD, "prompt": "fatal"}},
                "meta": {"fixture.example/authority": "frozen-metadata"},
            }
            assert caught.value.__cause__ is None
        assert [event.status for event in events] == (["in_progress", "failed"] if observed else [])
        if observed:
            assert events[0].tool_call_id == events[1].tool_call_id
            assert all(set(asdict(event)) == {"tool_call_id", "tool_name", "status"} for event in events)
            assert "private" not in str(events)
        assert _PRIVATE not in caplog.text

    caplog.set_level(logging.DEBUG, logger="agent_framework")
    asyncio.run(exercise())


@pytest.mark.parametrize("transport", ["streamable-http", "stdio"])
def test_admitted_mcp_error_can_recover_with_correlated_events(monkeypatch, caplog, transport):
    async def exercise():
        events = []

        async def observe(event):
            events.append(event)

        admitted = types.CallToolResult(content=[types.TextContent(type="text", text=_PRIVATE)], isError=True)
        async with AsyncExitStack() as stack:
            agent, server, session, client = await _setup(stack, monkeypatch, [admitted, _success()],
                                                         transport=transport, calls=2)
            result = await agent_factory.run_agent(agent, RunRequest("recover", on_tool_event=observe))
        assert result.text == "done"
        assert len(session.calls) == 2 and len(client.requests["recover"]) == 3
        assert server.connect.await_count == 0
        assert [event.status for event in events] == ["in_progress", "failed", "in_progress", "completed"]
        assert events[0].tool_call_id == events[1].tool_call_id
        assert events[2].tool_call_id == events[3].tool_call_id != events[0].tool_call_id
        assert _PRIVATE not in caplog.text

    caplog.set_level(logging.DEBUG, logger="agent_framework")
    asyncio.run(exercise())


def test_mcp_cancellation_propagates_without_retry_or_failure_event(monkeypatch):
    async def exercise():
        started, stopped = asyncio.Event(), asyncio.Event()
        events = []

        async def pending():
            started.set()
            try:
                await asyncio.Future()
            finally:
                stopped.set()

        async def observe(event):
            events.append(event)

        async with AsyncExitStack() as stack:
            agent, server, session, client = await _setup(stack, monkeypatch, [pending])
            run = asyncio.create_task(agent_factory.run_agent(agent, RunRequest("cancel", on_tool_event=observe)))
            try:
                await asyncio.wait_for(started.wait(), 3)
            finally:
                run.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(run, 3)
        assert stopped.is_set() and len(session.calls) == 1 and len(client.requests["cancel"]) == 1
        assert [event.status for event in events] == ["in_progress"] and server.connect.await_count == 0

    asyncio.run(exercise())


@pytest.mark.parametrize("status", ["in_progress", "completed", "failed"])
def test_mcp_observer_failure_remains_fatal_and_redacted(monkeypatch, caplog, status):
    async def exercise():
        async def observe(event):
            if event.status == status:
                raise RuntimeError(_PRIVATE)

        outcome = (types.CallToolResult(content=[], isError=True) if status == "failed" else _success())
        async with AsyncExitStack() as stack:
            agent, server, session, client = await _setup(stack, monkeypatch, [outcome])
            with pytest.raises(AgentRunError, match="tool lifecycle observer failed"):
                await agent_factory.run_agent(agent, RunRequest("observer", on_tool_event=observe))
        assert len(session.calls) == (0 if status == "in_progress" else 1)
        assert len(client.requests["observer"]) == 1 and server.connect.await_count == 0
        assert _PRIVATE not in caplog.text

    caplog.set_level(logging.DEBUG, logger="agent_framework")
    asyncio.run(exercise())


def test_protocol_failure_isolated_across_concurrent_and_subsequent_runs(monkeypatch):
    async def exercise():
        held, release = asyncio.Event(), asyncio.Event()

        async def pending_success():
            held.set()
            await release.wait()
            return _success()

        async with AsyncExitStack() as stack:
            agent, _, session, client = await _setup(stack, monkeypatch, [pending_success, _failure("jsonrpc"), _success()])
            good_session = AgentSession(session_id="good-session")
            failed_session = AgentSession(session_id="failed-session")
            good = asyncio.create_task(agent_factory.run_agent(
                agent, RunRequest("concurrent-success"), session=good_session,
            ))
            try:
                await asyncio.wait_for(held.wait(), 3)
                with pytest.raises(AgentRunError, match="MCP tool protocol failed"):
                    await agent_factory.run_agent(agent, RunRequest("concurrent-failure"), session=failed_session)
            finally:
                release.set()
            assert (await asyncio.wait_for(good, 3)).text == "done"
            assert (await agent_factory.run_agent(
                agent, RunRequest("subsequent-success"), session=good_session,
            )).text == "done"
        assert len(session.calls) == 3
        assert {name: len(requests) for name, requests in client.requests.items()} == {
            "concurrent-success": 2, "concurrent-failure": 1, "subsequent-success": 2,
        }

    asyncio.run(exercise())
