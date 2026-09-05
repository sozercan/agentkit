from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from openai import AsyncOpenAI
from pydantic_ai import Agent
from pydantic_ai.providers.openai import OpenAIProvider

from agentkit_serve import agent_factory
from agentkit_serve_common.config import AgentSpec
from agentkit_serve_common.conversation import RunRequest
from agentkit_serve_common.runtime import AgentRunError


PRIVATE = "private-incomplete-model-output"


def _frame(delta, finish=None):
    return (
        "data: "
        + json.dumps(
            {
                "id": "completion-test",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
        )
        + "\n\n"
    ).encode()


def _response(*, tool=False, complete=True, done=True):
    delta = (
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call-probe",
                    "type": "function",
                    "function": {"name": "probe", "arguments": "{}"},
                }
            ]
        }
        if tool
        else {"content": PRIVATE}
    )
    content = _frame({"role": "assistant"}) + _frame(delta)
    if complete:
        content += _frame({}, "tool_calls" if tool else "stop")
    if done:
        content += b"data: [DONE]\n\n"
    return httpx.Response(
        200, headers={"content-type": "text/event-stream"}, content=content
    )


def _model(monkeypatch, client):
    provider = OpenAIProvider(
        openai_client=AsyncOpenAI(
            api_key="synthetic-test-key",
            base_url="https://fixture.invalid/v1",
            http_client=client,
            max_retries=0,
        )
    )
    monkeypatch.setattr(agent_factory, "OpenAIProvider", lambda **kwargs: provider)
    return agent_factory.build_model(
        AgentSpec.model_validate(
            {
                "abiVersion": "v0",
                "metadata": {"name": "provider-test"},
                "model": {
                    "provider": "openai-compatible",
                    "baseURL": "https://fixture.invalid/v1",
                    "name": "gpt-4o-mini",
                    "apiKeyEnv": "OPENAI_API_KEY",
                },
                "instructions": "Use the probe.",
                "tools": [],
                "expose": {"openai": True, "port": 8080},
            }
        )
    )


@pytest.mark.parametrize("after_tool", [True, False])
@pytest.mark.parametrize("incomplete_part", ["text", "tool"])
def test_incomplete_stream_fails_before_committing_text_or_executing_its_tools(
    monkeypatch, after_tool, incomplete_part
):
    async def exercise():
        requests, invocations, events = [], [], []

        async def probe():
            invocations.append("probe")
            return "success"

        async def observe(event):
            events.append(event)

        def handle(request):
            requests.append(json.loads(request.content))
            if after_tool and len(requests) == 1:
                return _response(tool=True)
            if len(requests) > (2 if after_tool else 1):
                return _response()
            return _response(tool=incomplete_part == "tool", complete=False, done=False)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            agent = Agent(_model(monkeypatch, client), tools=[probe])
            async with agent:
                with pytest.raises(AgentRunError) as caught:
                    await agent_factory.run_agent(
                        agent, RunRequest("test", on_tool_event=observe)
                    )
        assert PRIVATE not in str(caught.value)
        assert len(requests) == (2 if after_tool else 1)
        assert invocations == (["probe"] if after_tool else [])
        assert [event.status for event in events] == (
            ["in_progress", "completed"] if after_tool else []
        )

    asyncio.run(exercise())


@pytest.mark.parametrize("done", [True, False])
def test_explicit_completion_reason_preserves_tool_loop_and_output(monkeypatch, done):
    async def exercise():
        requests, invocations, events = [], [], []

        async def probe():
            invocations.append("probe")
            return "success"

        async def observe(event):
            events.append(event)

        def handle(request):
            requests.append(json.loads(request.content))
            return _response(tool=len(requests) == 1, done=done)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            agent = Agent(_model(monkeypatch, client), tools=[probe])
            async with agent:
                result = await agent_factory.run_agent(
                    agent, RunRequest("test", on_tool_event=observe)
                )
        assert result.text == PRIVATE
        assert len(requests) == 2 and invocations == ["probe"]
        assert [event.status for event in events] == ["in_progress", "completed"]

    asyncio.run(exercise())


def test_cancelling_incomplete_stream_preserves_cancellation(monkeypatch):
    async def exercise():
        started, closed = asyncio.Event(), asyncio.Event()

        class PendingStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield _frame({"content": PRIVATE})
                started.set()
                await asyncio.Future()

            async def aclose(self):
                closed.set()

        async def observe(event):
            pass

        def handle(request):
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=PendingStream(),
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            agent = Agent(_model(monkeypatch, client))
            async with agent:
                run = asyncio.create_task(
                    agent_factory.run_agent(
                        agent, RunRequest("test", on_tool_event=observe)
                    )
                )
                await asyncio.wait_for(started.wait(), 3)
                run.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(run, 3)
        assert closed.is_set()

    asyncio.run(exercise())
