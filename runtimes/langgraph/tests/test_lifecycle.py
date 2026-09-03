from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest import mock

import pytest

from agentkit_serve import agent_factory
from agentkit_serve_common.config import AgentSpec


def _runtime_with_tools(*names: str) -> agent_factory.LangGraphRuntime:
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
            "tools": [{"name": name, "command": ["fake-mcp"], "env": []} for name in names],
            "expose": {"openai": True, "port": 8080},
        }
    )
    return agent_factory.LangGraphRuntime(spec)


def test_second_mcp_session_failure_preserves_startup_error_while_unwinding_first():
    runtime = _runtime_with_tools("first", "second")
    startup_error = RuntimeError("second session failed")
    cleanup_error = RuntimeError("first session cleanup failed")
    exited: list[str] = []
    owner_tasks: dict[str, asyncio.Task] = {}

    class _Session:
        async def initialize(self) -> None:
            return None

    class _SessionContext:
        def __init__(self, name: str) -> None:
            self.name = name

        async def __aenter__(self):
            if self.name == "second":
                raise startup_error
            owner_tasks[self.name] = asyncio.current_task()
            return _Session()

        async def __aexit__(self, exc_type, exc, tb):
            assert asyncio.current_task() is owner_tasks[self.name]
            exited.append(self.name)
            raise cleanup_error

    class _FakeClient:
        def __init__(self, connections, tool_name_prefix=False) -> None:
            pass

        def session(self, server_name, auto_initialize=True):
            return _SessionContext(server_name)

    async def _fake_load(session, *, server_name, tool_name_prefix):
        return [SimpleNamespace(name=f"{server_name}_tool")]

    with (
        mock.patch("agentkit_serve.agent_factory.MultiServerMCPClient", _FakeClient),
        mock.patch("agentkit_serve.agent_factory.load_mcp_tools", _fake_load),
    ):
        with pytest.raises(RuntimeError) as exc_info:
            asyncio.run(runtime.__aenter__())

    assert exc_info.value is startup_error
    assert exited == ["first"]


def test_cancellation_while_initializing_second_mcp_session_finishes_cleanup():
    runtime = _runtime_with_tools("first", "second")
    cleanup_error = RuntimeError("second session cleanup failed")
    second_initialize_started = asyncio.Event()
    second_exit_started = asyncio.Event()
    allow_second_exit = asyncio.Event()
    exited: list[str] = []
    owner_tasks: dict[str, asyncio.Task] = {}

    class _Session:
        def __init__(self, name: str) -> None:
            self.name = name

        async def initialize(self) -> None:
            if self.name == "second":
                second_initialize_started.set()
                await asyncio.Event().wait()

    class _SessionContext:
        def __init__(self, name: str) -> None:
            self.name = name

        async def __aenter__(self):
            owner_tasks[self.name] = asyncio.current_task()
            return _Session(self.name)

        async def __aexit__(self, exc_type, exc, tb):
            assert asyncio.current_task() is owner_tasks[self.name]
            if self.name == "second":
                second_exit_started.set()
                await allow_second_exit.wait()
            exited.append(self.name)
            if self.name == "second":
                raise cleanup_error

    class _FakeClient:
        def __init__(self, connections, tool_name_prefix=False) -> None:
            pass

        def session(self, server_name, auto_initialize=True):
            return _SessionContext(server_name)

    async def _fake_load(session, *, server_name, tool_name_prefix):
        return []

    async def exercise() -> None:
        task = asyncio.create_task(runtime.__aenter__())
        try:
            await asyncio.wait_for(second_initialize_started.wait(), timeout=1)
            task.cancel()
            await asyncio.wait_for(second_exit_started.wait(), timeout=1)

            # A second cancellation must not interrupt the already-running cleanup.
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()

            allow_second_exit.set()
            with pytest.raises(asyncio.CancelledError) as exc_info:
                await task
            assert any(
                "cleanup also failed with RuntimeError" in note
                for note in getattr(exc_info.value, "__notes__", ())
            )
        finally:
            allow_second_exit.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    with (
        mock.patch("agentkit_serve.agent_factory.MultiServerMCPClient", _FakeClient),
        mock.patch("agentkit_serve.agent_factory.load_mcp_tools", _fake_load),
    ):
        asyncio.run(exercise())

    assert exited == ["second", "first"]
