from __future__ import annotations

import asyncio
from unittest import mock

import pytest

from agentkit_serve import agent_factory
from agentkit_serve_common.config import AgentSpec, ToolSpec


def _runtime() -> agent_factory.MAFRuntime:
    spec = AgentSpec.model_validate(
        {
            "abiVersion": "v0",
            "metadata": {"name": "lifecycle-test"},
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
    return agent_factory.MAFRuntime(spec)


def test_agent_partial_enter_failure_closes_agent_and_context_without_masking_startup_error():
    runtime = _runtime()
    startup_error = RuntimeError("agent startup failed")
    agent_cleanup_error = RuntimeError("agent cleanup failed")
    events: list[str] = []
    owner_tasks: dict[str, asyncio.Task] = {}

    class _ContextResource:
        async def __aenter__(self):
            events.append("context-enter")
            owner_tasks["context"] = asyncio.current_task()
            return self

        async def __aexit__(self, exc_type, exc, tb):
            assert asyncio.current_task() is owner_tasks["context"]
            events.append("context-exit")

    class _PartialAgent:
        async def __aenter__(self):
            events.append("agent-enter")
            owner_tasks["agent"] = asyncio.current_task()
            raise startup_error

        async def __aexit__(self, exc_type, exc, tb):
            assert asyncio.current_task() is owner_tasks["agent"]
            events.append("agent-exit")
            raise agent_cleanup_error

    async def _build_context_providers():
        await runtime.stack.enter_async_context(_ContextResource())
        return None

    runtime._build_context_providers = _build_context_providers  # type: ignore[method-assign]
    with mock.patch("agentkit_serve.agent_factory.build_agent", return_value=_PartialAgent()):
        with pytest.raises(RuntimeError) as exc_info:
            asyncio.run(runtime.__aenter__())

    assert exc_info.value is startup_error
    assert events == ["context-enter", "agent-enter", "agent-exit", "context-exit"]
    assert runtime.agent is None


def test_runtime_exit_closes_context_stack_even_when_agent_exit_raises():
    runtime = _runtime()
    agent_exit_error = RuntimeError("agent exit failed")
    events: list[str] = []
    owner_tasks: dict[str, asyncio.Task] = {}

    class _ContextResource:
        async def __aenter__(self):
            events.append("context-enter")
            owner_tasks["context"] = asyncio.current_task()
            return self

        async def __aexit__(self, exc_type, exc, tb):
            assert asyncio.current_task() is owner_tasks["context"]
            events.append("context-exit")

    class _Agent:
        async def __aenter__(self):
            events.append("agent-enter")
            owner_tasks["agent"] = asyncio.current_task()
            return self

        async def __aexit__(self, exc_type, exc, tb):
            assert asyncio.current_task() is owner_tasks["agent"]
            events.append("agent-exit")
            raise agent_exit_error

    async def _build_context_providers():
        await runtime.stack.enter_async_context(_ContextResource())
        return None

    async def exercise() -> None:
        runtime._build_context_providers = _build_context_providers  # type: ignore[method-assign]
        with mock.patch("agentkit_serve.agent_factory.build_agent", return_value=_Agent()):
            await runtime.__aenter__()
        with pytest.raises(RuntimeError) as exc_info:
            await runtime.__aexit__(None, None, None)
        assert exc_info.value is agent_exit_error

    asyncio.run(exercise())

    assert events == ["context-enter", "agent-enter", "agent-exit", "context-exit"]
    assert runtime.agent is None


def test_cancellation_while_opening_second_context_resource_finishes_cleanup():
    runtime = _runtime()
    second_opened = asyncio.Event()
    second_exit_started = asyncio.Event()
    allow_second_exit = asyncio.Event()
    exited: list[str] = []
    owner_tasks: dict[str, asyncio.Task] = {}

    class _ContextResource:
        def __init__(self, name: str) -> None:
            self.name = name

        async def __aenter__(self):
            owner_tasks[self.name] = asyncio.current_task()
            return self

        async def __aexit__(self, exc_type, exc, tb):
            assert asyncio.current_task() is owner_tasks[self.name]
            if self.name == "second":
                second_exit_started.set()
                await allow_second_exit.wait()
            exited.append(self.name)

    async def _build_context_providers():
        await runtime.stack.enter_async_context(_ContextResource("first"))
        await runtime.stack.enter_async_context(_ContextResource("second"))
        second_opened.set()
        await asyncio.Event().wait()

    async def exercise() -> None:
        runtime._build_context_providers = _build_context_providers  # type: ignore[method-assign]
        task = asyncio.create_task(runtime.__aenter__())
        try:
            await asyncio.wait_for(second_opened.wait(), timeout=1)
            task.cancel()
            await asyncio.wait_for(second_exit_started.wait(), timeout=1)

            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()

            allow_second_exit.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            allow_second_exit.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(exercise())

    assert exited == ["second", "first"]


def test_exit_cancellation_waits_for_owner_cleanup_and_keeps_cancellation_primary():
    runtime = _runtime()
    cleanup_error = RuntimeError("context cleanup failed")
    exit_started = asyncio.Event()
    allow_exit = asyncio.Event()
    events: list[str] = []
    owner_tasks: dict[str, asyncio.Task] = {}

    class _ContextResource:
        async def __aenter__(self):
            events.append("context-enter")
            owner_tasks["context"] = asyncio.current_task()
            return self

        async def __aexit__(self, exc_type, exc, tb):
            assert asyncio.current_task() is owner_tasks["context"]
            exit_started.set()
            await allow_exit.wait()
            events.append("context-exit")
            raise cleanup_error

    class _Agent:
        async def __aenter__(self):
            events.append("agent-enter")
            owner_tasks["agent"] = asyncio.current_task()
            return self

        async def __aexit__(self, exc_type, exc, tb):
            assert asyncio.current_task() is owner_tasks["agent"]
            events.append("agent-exit")

    async def _build_context_providers():
        await runtime.stack.enter_async_context(_ContextResource())
        return None

    async def exercise() -> None:
        runtime._build_context_providers = _build_context_providers  # type: ignore[method-assign]
        with mock.patch("agentkit_serve.agent_factory.build_agent", return_value=_Agent()):
            await runtime.__aenter__()

        exit_task = asyncio.create_task(runtime.__aexit__(None, None, None))
        try:
            await asyncio.wait_for(exit_started.wait(), timeout=1)
            exit_task.cancel()
            allow_exit.set()
            with pytest.raises(asyncio.CancelledError) as exc_info:
                await exit_task
            assert exc_info.value.__cause__ is cleanup_error
        finally:
            allow_exit.set()
            if not exit_task.done():
                exit_task.cancel()
            await asyncio.gather(exit_task, return_exceptions=True)

    asyncio.run(exercise())

    assert events == ["context-enter", "agent-enter", "agent-exit", "context-exit"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, 120), ("garbage", 120), ("-1", 120), ("7.9", 7)],
)
def test_stdio_mcp_timeout_defaults_to_120_seconds_and_honors_override(
    monkeypatch,
    raw: str | None,
    expected: int,
):
    tool = ToolSpec(name="fetch", command=["fake-mcp"], env=[])

    if raw is None:
        monkeypatch.delenv("AGENTKIT_MCP_TIMEOUT", raising=False)
    else:
        monkeypatch.setenv("AGENTKIT_MCP_TIMEOUT", raw)
    assert agent_factory.build_tool(tool).request_timeout == expected
