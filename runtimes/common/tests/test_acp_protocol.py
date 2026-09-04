from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import queue
import threading
from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import Any

import pytest

from agentkit_serve_common import acp
from agentkit_serve_common.acp import ACPConfigurationError, ACPStdioServer
from agentkit_serve_common.config import AgentSpec
from agentkit_serve_common.conversation import ConversationTurn, RunRequest
from agentkit_serve_common.runtime import (
    AgentRunError,
    OfflineEchoRuntimeFactory,
    RunResult,
    RuntimeSession,
    offline_orka_echo_enabled,
)


def _spec(**overrides: Any) -> AgentSpec:
    data: dict[str, Any] = {
        "abiVersion": "v0",
        "metadata": {"name": "acp-test"},
        "model": {
            "provider": "openai-compatible",
            "baseURL": "https://baked.example.invalid/v1",
            "name": "test-model",
            "apiKeyEnv": "BAKED_MODEL_TOKEN",
        },
        "instructions": "Be concise.",
        "tools": [],
        "expose": {"openai": True, "port": 8080},
    }
    data.update(overrides)
    return AgentSpec.model_validate(data)


def _set_provider_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(acp.ACP_PROVIDER_BASE_URL_ENV, "http://127.0.0.1:43123/v1")
    monkeypatch.setenv(acp.ACP_PROVIDER_TOKEN_ENV, "provider-session-token")


def _initialize(request_id: int = 1) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "initialize",
        "params": {
            "protocolVersion": 1,
            "clientCapabilities": {
                "fs": {"readTextFile": False, "writeTextFile": False},
                "terminal": False,
            },
            "clientInfo": {"name": "orka", "version": "test"},
        },
    }


def _new_session(request_id: int = 2, *, mcp_servers: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "session/new",
        "params": {
            "cwd": os.getcwd(),
            "mcpServers": [] if mcp_servers is None else mcp_servers,
        },
    }


def _prompt(request_id: int, session_id: str, text: str) -> dict[str, Any]:
    return _prompt_content(
        request_id,
        session_id,
        [{"type": "text", "text": text}],
    )


def _prompt_content(
    request_id: int,
    session_id: str,
    content: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "session/prompt",
        "params": {
            "sessionId": session_id,
            "prompt": content,
        },
    }


def _orka_mcp_server(url: str = "http://127.0.0.1:43124/session") -> dict[str, Any]:
    return {
        "type": "http",
        "name": "orka",
        "url": url,
        "headers": [{"name": "Authorization", "value": "Bearer mcp-session-token"}],
    }


def _response(messages: list[dict[str, Any]], request_id: int) -> dict[str, Any]:
    matches = [message for message in messages if message.get("id") == request_id]
    assert len(matches) == 1
    return matches[0]


async def _send_to(
    server: ACPStdioServer,
    messages: list[dict[str, Any]],
    request: dict[str, Any],
) -> dict[str, Any]:
    await server.accept(request)
    await server.wait_idle()
    return _response(messages, request["id"])


async def _wait_for_response(
    messages: list[dict[str, Any]],
    request_id: int,
) -> dict[str, Any]:
    async def wait() -> dict[str, Any]:
        while True:
            matches = [message for message in messages if message.get("id") == request_id]
            if matches:
                assert len(matches) == 1
                return matches[0]
            await asyncio.sleep(0)

    return await asyncio.wait_for(wait(), timeout=1)


class RecordingRuntime:
    def __init__(self, outcomes: list[RunResult | BaseException]) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[RunRequest] = []
        self.discarded_sessions: list[str] = []
        self.entered = 0
        self.exited = 0

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
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def discard_session(self, session_id: str) -> None:
        self.discarded_sessions.append(session_id)


class RecordingFactory:
    def __init__(
        self,
        runtime_builder: Callable[[], RuntimeSession],
        *,
        supports_http_mcp: bool = False,
    ) -> None:
        self.runtime_builder = runtime_builder
        self.supports_http_mcp = supports_http_mcp
        self.specs: list[AgentSpec] = []
        self.runtimes: list[RuntimeSession] = []
        self.environment_snapshots: list[dict[str, str | None]] = []

    def supports_acp_http_mcp(self) -> bool:
        return self.supports_http_mcp

    def build_runtime(self, spec: AgentSpec) -> RuntimeSession:
        self.specs.append(spec)
        environment_names = [
            name
            for tool in spec.tools
            for name in [tool.url_env, *(header.value_env for header in tool.headers)]
            if name is not None
        ]
        self.environment_snapshots.append({name: os.environ.get(name) for name in environment_names})
        runtime = self.runtime_builder()
        self.runtimes.append(runtime)
        return runtime


class BlockingEnterRuntime:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.exited = 0

    async def __aenter__(self) -> RuntimeSession:
        self.started.set()
        await self.release.wait()
        return self

    async def __aexit__(self, exc_type, exc, tb):  # noqa: ANN001
        self.exited += 1
        return None

    async def run(self, request: RunRequest) -> RunResult:
        raise AssertionError(f"unexpected prompt for {request.session_id}")


class BlockingExitRuntime(BlockingEnterRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.exit_started = asyncio.Event()
        self.release_exit = asyncio.Event()
        self.exit_completed = False

    async def __aexit__(self, exc_type, exc, tb):  # noqa: ANN001
        self.exited += 1
        self.exit_started.set()
        await self.release_exit.wait()
        self.exit_completed = True
        return None


class FeedingReader:
    def __init__(self) -> None:
        self.lines: queue.Queue[bytes] = queue.Queue()

    def readline(self, limit: int) -> bytes:  # noqa: ARG002
        return self.lines.get()

    def feed(self, message: dict[str, Any]) -> None:
        self.lines.put(json.dumps(message, separators=(",", ":")).encode() + b"\n")

    def close(self) -> None:
        self.lines.put(b"")


class RecordingWriter:
    def __init__(self, *, block_first_update: bool = False) -> None:
        self.block_first_update = block_first_update
        self.blocked = threading.Event()
        self.release = threading.Event()
        self.frames: list[bytes] = []
        self.lock = threading.Lock()

    def write(self, frame: bytes) -> int:
        message = json.loads(frame)
        if (
            self.block_first_update
            and message.get("method") == "session/update"
            and not self.blocked.is_set()
        ):
            self.blocked.set()
            if not self.release.wait(timeout=5):
                raise TimeoutError("test did not release blocked ACP writer")
        with self.lock:
            self.frames.append(bytes(frame))
        return len(frame)

    def flush(self) -> None:
        return None

    def messages(self) -> list[dict[str, Any]]:
        with self.lock:
            frames = list(self.frames)
        return [json.loads(frame) for frame in frames]


async def _wait_for_stdio_response(writer: RecordingWriter, request_id: int) -> dict[str, Any]:
    async def wait() -> dict[str, Any]:
        while True:
            matches = [message for message in writer.messages() if message.get("id") == request_id]
            if matches:
                assert len(matches) == 1
                return matches[0]
            await asyncio.sleep(0)

    return await asyncio.wait_for(wait(), timeout=5)


def test_offline_echo_round_trip_uses_canonical_acp_shapes(monkeypatch):
    _set_provider_environment(monkeypatch)

    async def exercise() -> None:
        messages: list[dict[str, Any]] = []

        async def send(message):  # noqa: ANN001
            messages.append(dict(message))

        server = ACPStdioServer(_spec(), OfflineEchoRuntimeFactory(), send)
        initialized = await _send_to(server, messages, _initialize())
        assert initialized["result"]["protocolVersion"] == 1
        assert initialized["result"]["agentCapabilities"]["mcpCapabilities"] == {}

        created = await _send_to(server, messages, _new_session())
        session_id = created["result"]["sessionId"]
        completed = await _send_to(server, messages, _prompt(3, session_id, "hello"))

        assert completed["result"] == {"stopReason": "end_turn"}
        assert {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": session_id,
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "offline echo: hello"},
                },
            },
        } in messages
        assert server.sessions[session_id].history == [
            ConversationTurn(role="user", text="hello"),
            ConversationTurn(role="assistant", text="offline echo: hello"),
        ]
        await server.close()

    asyncio.run(exercise())


def test_stdio_server_frames_json_rpc_and_closes_on_eof(monkeypatch):
    _set_provider_environment(monkeypatch)
    monkeypatch.setattr(acp.secrets, "token_hex", lambda size: "b" * (size * 2))

    async def exercise() -> list[dict[str, Any]]:
        reader = FeedingReader()
        writer = RecordingWriter()
        serve_task = asyncio.create_task(
            acp.serve_acp_stdio(
                _spec(),
                OfflineEchoRuntimeFactory(),
                reader=reader,
                writer=writer,
            )
        )
        try:
            reader.feed(_initialize())
            await _wait_for_stdio_response(writer, 1)
            reader.feed(_new_session())
            created = await _wait_for_stdio_response(writer, 2)
            reader.feed(_prompt(3, created["result"]["sessionId"], "through stdio"))
            await _wait_for_stdio_response(writer, 3)
        finally:
            reader.close()
            await asyncio.wait_for(serve_task, timeout=1)
        return writer.messages()

    output = asyncio.run(exercise())
    assert _response(output, 3)["result"] == {"stopReason": "end_turn"}
    assert any(message.get("method") == "session/update" for message in output)


def test_stdio_server_splits_output_larger_than_orka_acp_reader_limit(monkeypatch):
    _set_provider_environment(monkeypatch)
    monkeypatch.setattr(acp.secrets, "token_hex", lambda size: "c" * (size * 2))
    session_id = "agentkit-" + "c" * 32
    result_text = "x" * ((2 << 20) + 1)
    runtime = RecordingRuntime([RunResult(result_text)])

    async def exercise() -> RecordingWriter:
        reader = FeedingReader()
        writer = RecordingWriter()
        serve_task = asyncio.create_task(
            acp.serve_acp_stdio(
                _spec(),
                RecordingFactory(lambda: runtime),
                reader=reader,
                writer=writer,
            )
        )
        try:
            reader.feed(_initialize())
            await _wait_for_stdio_response(writer, 1)
            reader.feed(_new_session())
            await _wait_for_stdio_response(writer, 2)
            reader.feed(_prompt(3, session_id, "large output"))
            await _wait_for_stdio_response(writer, 3)
        finally:
            reader.close()
            await asyncio.wait_for(serve_task, timeout=1)
        return writer

    writer = asyncio.run(exercise())
    lines = list(writer.frames)
    output = writer.messages()
    updates = [message for message in output if message.get("method") == "session/update"]
    chunks = [message["params"]["update"]["content"]["text"] for message in updates]

    assert len(updates) > 1
    assert "".join(chunks) == result_text
    assert all(
        len(chunk.encode()) <= acp._MAX_ASSISTANT_MESSAGE_CHUNK_BYTES  # noqa: SLF001
        for chunk in chunks
    )
    assert all(len(line) < 512 << 10 for line in lines)
    assert _response(output, 3)["result"] == {"stopReason": "end_turn"}


def test_stdio_server_drains_oversized_line_before_parsing_next_frame(monkeypatch):
    monkeypatch.setattr(acp, "_MAX_MESSAGE_BYTES", 1024)
    injected = json.dumps(_initialize(request_id=99), separators=(",", ":")).encode()
    oversized_line = b"x" * (acp._MAX_MESSAGE_BYTES + 2) + injected + b"\n"
    valid_line = json.dumps(_initialize(), separators=(",", ":")).encode() + b"\n"
    reader = io.BytesIO(oversized_line + valid_line)
    writer = io.BytesIO()

    asyncio.run(
        acp.serve_acp_stdio(
            _spec(),
            OfflineEchoRuntimeFactory(),
            reader=reader,
            writer=writer,
        )
    )

    output = [json.loads(line) for line in writer.getvalue().splitlines()]
    oversized_errors = [
        message
        for message in output
        if message.get("error", {}).get("message") == "ACP message exceeds the 8 MiB limit"
    ]
    assert len(oversized_errors) == 1
    assert not any(message.get("id") == 99 for message in output)
    assert _response(output, 1)["result"]["protocolVersion"] == 1


def test_stdio_blocked_writer_does_not_block_cancel_dispatch(monkeypatch):
    _set_provider_environment(monkeypatch)
    runtime = RecordingRuntime([RunResult("late answer"), RunResult("clean answer")])
    factory = RecordingFactory(lambda: runtime)
    cancel_dispatched = asyncio.Event()
    original_cancel_state = ACPStdioServer._cancel_state  # noqa: SLF001

    def record_cancel(server: ACPStdioServer, state) -> None:  # noqa: ANN001
        cancel_dispatched.set()
        original_cancel_state(server, state)

    monkeypatch.setattr(ACPStdioServer, "_cancel_state", record_cancel)

    async def exercise() -> None:
        reader = FeedingReader()
        writer = RecordingWriter(block_first_update=True)
        serve_task = asyncio.create_task(
            acp.serve_acp_stdio(_spec(), factory, reader=reader, writer=writer)
        )
        try:
            reader.feed(_initialize())
            await _wait_for_stdio_response(writer, 1)
            reader.feed(_new_session())
            created = await _wait_for_stdio_response(writer, 2)
            session_id = created["result"]["sessionId"]

            reader.feed(_prompt(3, session_id, "cancel under backpressure"))
            assert await asyncio.to_thread(writer.blocked.wait, 1)
            reader.feed(
                {
                    "jsonrpc": "2.0",
                    "method": "session/cancel",
                    "params": {"sessionId": session_id},
                }
            )
            await asyncio.wait_for(cancel_dispatched.wait(), timeout=1)

            writer.release.set()
            cancelled = await _wait_for_stdio_response(writer, 3)
            assert cancelled["result"] == {"stopReason": "cancelled"}
            assert runtime.discarded_sessions == [session_id]

            reader.feed(_prompt(4, session_id, "try again"))
            completed = await _wait_for_stdio_response(writer, 4)
            assert completed["result"] == {"stopReason": "end_turn"}
            assert runtime.requests[1].history == ()
        finally:
            writer.release.set()
            reader.close()
            await asyncio.wait_for(serve_task, timeout=1)

    asyncio.run(exercise())


def test_stdio_writer_close_waits_for_in_flight_write_after_cancellation():
    async def exercise() -> None:
        output = RecordingWriter(block_first_update=True)
        writer = acp._ACPStdioWriter(output)  # noqa: SLF001
        send_task = asyncio.create_task(
            writer.send({"jsonrpc": "2.0", "method": "session/update"})
        )
        assert await asyncio.to_thread(output.blocked.wait, 1)

        close_task = asyncio.create_task(writer.close())
        await asyncio.sleep(0)
        close_task.cancel()
        await asyncio.sleep(0)
        assert not close_task.done()

        output.release.set()
        await send_task
        with pytest.raises(asyncio.CancelledError):
            await close_task

    asyncio.run(exercise())


def test_stdio_writer_retains_write_ownership_after_sender_cancellation():
    class BlockingWriter:
        def __init__(self) -> None:
            self.first_started = threading.Event()
            self.second_started = threading.Event()
            self.release_first = threading.Event()
            self.frames: list[bytes] = []
            self.active_writes = 0
            self.max_active_writes = 0
            self.lock = threading.Lock()

        def write(self, frame: bytes) -> int:
            message = json.loads(frame)
            with self.lock:
                self.active_writes += 1
                self.max_active_writes = max(self.max_active_writes, self.active_writes)
            try:
                if message["id"] == 1:
                    self.first_started.set()
                    if not self.release_first.wait(timeout=5):
                        raise TimeoutError("test did not release first ACP writer call")
                else:
                    self.second_started.set()
                with self.lock:
                    self.frames.append(bytes(frame))
                return len(frame)
            finally:
                with self.lock:
                    self.active_writes -= 1

        def flush(self) -> None:
            return None

    async def exercise() -> None:
        output = BlockingWriter()
        writer = acp._ACPStdioWriter(output)  # noqa: SLF001
        first_send = asyncio.create_task(
            writer.send({"jsonrpc": "2.0", "id": 1, "result": {}})
        )
        assert await asyncio.to_thread(output.first_started.wait, 1)

        first_send.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first_send

        second_send = asyncio.create_task(
            writer.send({"jsonrpc": "2.0", "id": 2, "result": {}})
        )
        await asyncio.sleep(0)
        assert not second_send.done()

        close_task = asyncio.create_task(writer.close())
        await asyncio.sleep(0)
        assert not output.second_started.is_set()
        assert not close_task.done()

        output.release_first.set()
        await second_send
        await close_task

        assert [json.loads(frame)["id"] for frame in output.frames] == [1, 2]
        assert output.max_active_writes == 1

    asyncio.run(exercise())


def test_stdio_server_surfaces_writer_failure():
    class FailingWriter:
        def write(self, frame: bytes) -> int:  # noqa: ARG002
            raise BrokenPipeError("supervisor closed stdout")

        def flush(self) -> None:
            return None

    request = json.dumps(_initialize(), separators=(",", ":")).encode() + b"\n"
    with pytest.raises(BrokenPipeError, match="supervisor closed stdout"):
        asyncio.run(
            acp.serve_acp_stdio(
                _spec(),
                OfflineEchoRuntimeFactory(),
                reader=io.BytesIO(request),
                writer=FailingWriter(),
            )
        )


def test_offline_echo_gate_applies_to_acp(monkeypatch):
    monkeypatch.setenv("AGENTKIT_PROTOCOL", "acp")
    monkeypatch.setenv("AGENTKIT_ORKA_OFFLINE_ECHO", "true")

    assert offline_orka_echo_enabled() is True


def test_session_reuses_runtime_and_passes_full_successful_history(monkeypatch):
    _set_provider_environment(monkeypatch)
    runtime = RecordingRuntime([RunResult("first answer"), RunResult("second answer")])
    factory = RecordingFactory(lambda: runtime, supports_http_mcp=True)

    async def exercise() -> None:
        messages: list[dict[str, Any]] = []

        async def send(message):  # noqa: ANN001
            messages.append(dict(message))

        server = ACPStdioServer(_spec(), factory, send)
        initialized = await _send_to(server, messages, _initialize())
        assert initialized["result"]["agentCapabilities"]["mcpCapabilities"] == {"http": True}
        created = await _send_to(
            server,
            messages,
            _new_session(mcp_servers=[_orka_mcp_server()]),
        )
        session_id = created["result"]["sessionId"]

        assert (await _send_to(server, messages, _prompt(3, session_id, "  first question  ")))[
            "result"
        ] == {"stopReason": "end_turn"}
        assert (await _send_to(server, messages, _prompt(4, session_id, "second question")))[
            "result"
        ] == {"stopReason": "end_turn"}

        assert len(factory.specs) == 1
        assert runtime.entered == 1
        assert runtime.requests == [
            RunRequest(prompt="  first question  ", history=(), session_id=session_id),
            RunRequest(
                prompt="second question",
                history=(
                    ConversationTurn(role="user", text="  first question  "),
                    ConversationTurn(role="assistant", text="first answer"),
                ),
                session_id=session_id,
            ),
        ]

        projected = factory.specs[0]
        assert projected.model.name == "test-model"
        assert projected.model.base_url == "http://127.0.0.1:43123/v1"
        assert projected.model.api_key_env == acp.ACP_PROVIDER_TOKEN_ENV
        assert projected.model.auth is None
        assert len(projected.tools) == 1
        tool = projected.tools[0]
        assert tool.transport == "streamable-http"
        assert tool.url_env is not None
        assert factory.environment_snapshots[0][tool.url_env] == _orka_mcp_server()["url"]
        assert tool.headers[0].value_env is not None
        assert factory.environment_snapshots[0][tool.headers[0].value_env] == "Bearer mcp-session-token"

        generated_names = set(factory.environment_snapshots[0])
        assert all(os.environ.get(name) for name in generated_names)
        await server.close()
        assert runtime.exited == 1
        assert all(name not in os.environ for name in generated_names)

    asyncio.run(exercise())


def test_resource_links_are_projected_and_preserved_in_history(monkeypatch):
    _set_provider_environment(monkeypatch)
    runtime = RecordingRuntime([RunResult("first answer"), RunResult("second answer")])
    factory = RecordingFactory(lambda: runtime)
    projected_prompt = (
        "Review this resource.\n"
        "Resource link: design notes\n"
        "URI: https://example.com/design.md\n"
        "MIME type: text/markdown\n"
        "Summarize it."
    )

    async def exercise() -> None:
        messages: list[dict[str, Any]] = []

        async def send(message):  # noqa: ANN001
            messages.append(dict(message))

        server = ACPStdioServer(_spec(), factory, send)
        await _send_to(server, messages, _initialize())
        created = await _send_to(server, messages, _new_session())
        session_id = created["result"]["sessionId"]
        prompt = _prompt_content(
            3,
            session_id,
            [
                {"type": "text", "text": "Review this resource."},
                {
                    "type": "resource_link",
                    "name": "design notes",
                    "uri": "https://example.com/design.md",
                    "mimeType": "text/markdown",
                },
                {"type": "text", "text": "Summarize it."},
            ],
        )

        assert (await _send_to(server, messages, prompt))["result"] == {
            "stopReason": "end_turn"
        }
        assert (
            await _send_to(server, messages, _prompt(4, session_id, "What changed?"))
        )["result"] == {"stopReason": "end_turn"}
        assert runtime.requests == [
            RunRequest(prompt=projected_prompt, history=(), session_id=session_id),
            RunRequest(
                prompt="What changed?",
                history=(
                    ConversationTurn(role="user", text=projected_prompt),
                    ConversationTurn(role="assistant", text="first answer"),
                ),
                session_id=session_id,
            ),
        ]
        await server.close()

    asyncio.run(exercise())


def test_child_rejects_a_second_session(monkeypatch):
    _set_provider_environment(monkeypatch)
    factory = RecordingFactory(lambda: RecordingRuntime([RunResult("unused")]))

    async def exercise() -> None:
        messages: list[dict[str, Any]] = []

        async def send(message):  # noqa: ANN001
            messages.append(dict(message))

        server = ACPStdioServer(_spec(), factory, send)
        await _send_to(server, messages, _initialize())
        assert "sessionId" in (await _send_to(server, messages, _new_session()))["result"]
        rejected = await _send_to(server, messages, _new_session(request_id=3))

        assert rejected["error"]["code"] == -32600
        assert "already owns a session" in rejected["error"]["message"]
        assert len(factory.specs) == 1
        await server.close()

    asyncio.run(exercise())


def test_cancel_request_stops_blocked_session_creation_and_restores_environment(monkeypatch):
    _set_provider_environment(monkeypatch)
    runtime = BlockingEnterRuntime()
    factory = RecordingFactory(lambda: runtime, supports_http_mcp=True)

    async def exercise() -> None:
        messages: list[dict[str, Any]] = []

        async def send(message):  # noqa: ANN001
            messages.append(dict(message))

        server = ACPStdioServer(_spec(), factory, send)
        await _send_to(server, messages, _initialize())
        await server.accept(_new_session(mcp_servers=[_orka_mcp_server()]))
        await asyncio.wait_for(runtime.started.wait(), timeout=1)
        projected_names = set(factory.environment_snapshots[0])
        assert projected_names
        assert all(os.environ.get(name) for name in projected_names)

        await server.accept(
            {
                "jsonrpc": "2.0",
                "method": "$/cancel_request",
                "params": {"requestId": 2},
            }
        )
        await asyncio.wait_for(server.wait_idle(), timeout=1)

        assert _response(messages, 2)["error"] == {
            "code": -32800,
            "message": "ACP request cancelled",
        }
        assert server.sessions == {}
        assert runtime.exited == 1
        assert all(name not in os.environ for name in projected_names)
        await asyncio.wait_for(server.close(), timeout=1)

    asyncio.run(exercise())


def test_cancel_request_before_session_dispatch_prevents_runtime_start(monkeypatch):
    _set_provider_environment(monkeypatch)
    runtime = BlockingEnterRuntime()
    factory = RecordingFactory(lambda: runtime)

    async def exercise() -> None:
        messages: list[dict[str, Any]] = []

        async def send(message):  # noqa: ANN001
            messages.append(dict(message))

        server = ACPStdioServer(_spec(), factory, send)
        await _send_to(server, messages, _initialize())
        accept_task = asyncio.create_task(server.accept(_new_session()))
        await asyncio.sleep(0)
        assert not runtime.started.is_set()

        await server.accept(
            {
                "jsonrpc": "2.0",
                "method": "$/cancel_request",
                "params": {"requestId": 2},
            }
        )
        await accept_task
        await asyncio.wait_for(server.wait_idle(), timeout=1)

        assert _response(messages, 2)["error"]["code"] == -32800
        assert factory.runtimes == []
        assert server.sessions == {}
        await server.close()

    asyncio.run(exercise())


def test_close_cancels_blocked_session_creation(monkeypatch):
    _set_provider_environment(monkeypatch)
    runtime = BlockingEnterRuntime()
    factory = RecordingFactory(lambda: runtime)

    async def exercise() -> None:
        messages: list[dict[str, Any]] = []

        async def send(message):  # noqa: ANN001
            messages.append(dict(message))

        server = ACPStdioServer(_spec(), factory, send)
        await _send_to(server, messages, _initialize())
        accept_task = asyncio.create_task(server.accept(_new_session()))
        await asyncio.sleep(0)
        assert not runtime.started.is_set()

        async with asyncio.timeout(1):
            await server.close()
        await accept_task

        assert _response(messages, 2)["error"]["code"] == -32800
        assert server.sessions == {}
        assert factory.runtimes == []

    asyncio.run(exercise())


def test_close_does_not_interrupt_cancelled_session_cleanup(monkeypatch):
    _set_provider_environment(monkeypatch)
    runtime = BlockingExitRuntime()
    factory = RecordingFactory(lambda: runtime)

    async def exercise() -> None:
        messages: list[dict[str, Any]] = []

        async def send(message):  # noqa: ANN001
            messages.append(dict(message))

        server = ACPStdioServer(_spec(), factory, send)
        await _send_to(server, messages, _initialize())
        await server.accept(_new_session())
        await asyncio.wait_for(runtime.started.wait(), timeout=1)
        await server.accept(
            {
                "jsonrpc": "2.0",
                "method": "$/cancel_request",
                "params": {"requestId": 2},
            }
        )
        await asyncio.wait_for(runtime.exit_started.wait(), timeout=1)

        close_task = asyncio.create_task(server.close())
        await asyncio.sleep(0)
        assert not close_task.done()
        assert not runtime.exit_completed

        runtime.release_exit.set()
        await asyncio.wait_for(close_task, timeout=1)

        assert runtime.exit_completed
        assert runtime.exited == 1
        assert _response(messages, 2)["error"]["code"] == -32800
        assert server.sessions == {}

    asyncio.run(exercise())


def test_cancel_request_after_session_commit_does_not_replace_success(monkeypatch):
    _set_provider_environment(monkeypatch)
    runtime = RecordingRuntime([RunResult("unused")])
    factory = RecordingFactory(lambda: runtime)

    async def exercise() -> None:
        messages: list[dict[str, Any]] = []
        response_started = asyncio.Event()
        release_response = asyncio.Event()

        async def send(message):  # noqa: ANN001
            if message.get("id") == 2 and "result" in message:
                response_started.set()
                await release_response.wait()
            messages.append(dict(message))

        server = ACPStdioServer(_spec(), factory, send)
        await _send_to(server, messages, _initialize())
        await server.accept(_new_session())
        await asyncio.wait_for(response_started.wait(), timeout=1)

        for request_id in (2, 2, 999):
            await server.accept(
                {
                    "jsonrpc": "2.0",
                    "method": "$/cancel_request",
                    "params": {"requestId": request_id},
                }
            )
        release_response.set()
        await asyncio.wait_for(server.wait_idle(), timeout=1)

        created = _response(messages, 2)
        assert "error" not in created
        assert created["result"]["sessionId"] in server.sessions
        assert runtime.entered == 1
        await server.close()

    asyncio.run(exercise())


class CancellingRuntime:
    def __init__(self, *, swallow_cancellation: bool) -> None:
        self.swallow_cancellation = swallow_cancellation
        self.started = asyncio.Event()
        self.block = asyncio.Event()
        self.requests: list[RunRequest] = []
        self.discarded_sessions: list[str] = []

    async def __aenter__(self) -> RuntimeSession:
        return self

    async def __aexit__(self, exc_type, exc, tb):  # noqa: ANN001
        return None

    async def run(self, request: RunRequest) -> RunResult:
        self.requests.append(request)
        if len(self.requests) > 1:
            return RunResult("clean answer")
        self.started.set()
        try:
            await self.block.wait()
        except asyncio.CancelledError:
            if self.swallow_cancellation:
                return RunResult("partial answer that must be discarded")
            raise
        return RunResult("unexpected answer")

    async def discard_session(self, session_id: str) -> None:
        self.discarded_sessions.append(session_id)


def test_concurrent_prompt_is_rejected_as_cancelled_without_disturbing_active_prompt(
    monkeypatch,
):
    _set_provider_environment(monkeypatch)
    runtime = CancellingRuntime(swallow_cancellation=False)
    factory = RecordingFactory(lambda: runtime)

    async def exercise() -> None:
        messages: list[dict[str, Any]] = []

        async def send(message):  # noqa: ANN001
            messages.append(dict(message))

        server = ACPStdioServer(_spec(), factory, send)
        await _send_to(server, messages, _initialize())
        created = await _send_to(server, messages, _new_session())
        session_id = created["result"]["sessionId"]

        await server.accept(_prompt(3, session_id, "keep running"))
        await asyncio.wait_for(runtime.started.wait(), timeout=1)
        await server.accept(_prompt(4, session_id, "reject me"))
        rejected = await _wait_for_response(messages, 4)

        assert rejected["error"] == {
            "code": -32800,
            "message": "ACP session already has an active prompt",
        }
        assert not any(message.get("id") == 3 for message in messages)
        assert len(runtime.requests) == 1
        assert server.sessions[session_id].history == []

        await server.accept(
            {
                "jsonrpc": "2.0",
                "method": "session/cancel",
                "params": {"sessionId": session_id},
            }
        )
        await asyncio.wait_for(server.wait_idle(), timeout=1)
        assert _response(messages, 3)["result"] == {"stopReason": "cancelled"}
        assert server.sessions[session_id].history == []

        completed = await _send_to(server, messages, _prompt(5, session_id, "try again"))
        assert completed["result"] == {"stopReason": "end_turn"}
        assert runtime.requests[1].history == ()
        await server.close()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("cancel_method", "swallow_cancellation"),
    [("session/cancel", False), ("$/cancel_request", True)],
)
def test_cancellation_returns_cancelled_and_does_not_commit_history(
    monkeypatch,
    cancel_method: str,
    swallow_cancellation: bool,
):
    _set_provider_environment(monkeypatch)
    runtime = CancellingRuntime(swallow_cancellation=swallow_cancellation)
    factory = RecordingFactory(lambda: runtime)

    async def exercise() -> None:
        messages: list[dict[str, Any]] = []

        async def send(message):  # noqa: ANN001
            messages.append(dict(message))

        server = ACPStdioServer(_spec(), factory, send)
        await _send_to(server, messages, _initialize())
        created = await _send_to(server, messages, _new_session())
        session_id = created["result"]["sessionId"]

        await server.accept(_prompt(3, session_id, "cancel me"))
        await asyncio.wait_for(runtime.started.wait(), timeout=1)
        params = {"sessionId": session_id} if cancel_method == "session/cancel" else {"requestId": 3}
        await server.accept(
            {"jsonrpc": "2.0", "method": cancel_method, "params": params}
        )
        await server.wait_idle()

        assert _response(messages, 3)["result"] == {"stopReason": "cancelled"}
        assert not [message for message in messages if message.get("method") == "session/update"]
        assert server.sessions[session_id].history == []
        assert runtime.discarded_sessions == [session_id]

        completed = await _send_to(server, messages, _prompt(4, session_id, "try again"))
        assert completed["result"] == {"stopReason": "end_turn"}
        assert runtime.requests[1].history == ()
        await server.close()

    asyncio.run(exercise())


def test_runtime_error_is_redacted_and_does_not_commit_history(monkeypatch):
    _set_provider_environment(monkeypatch)
    runtime = RecordingRuntime(
        [
            AgentRunError("provider leaked secret-token", code="ProviderFailure"),
            RunResult("recovered"),
        ]
    )
    factory = RecordingFactory(lambda: runtime)

    async def exercise() -> None:
        messages: list[dict[str, Any]] = []

        async def send(message):  # noqa: ANN001
            messages.append(dict(message))

        server = ACPStdioServer(_spec(), factory, send)
        await _send_to(server, messages, _initialize())
        created = await _send_to(server, messages, _new_session())
        session_id = created["result"]["sessionId"]

        failed = await _send_to(server, messages, _prompt(3, session_id, "fail"))
        assert failed["error"] == {
            "code": -32603,
            "message": "AgentKit runtime prompt failed",
            "data": {"code": "ProviderFailure"},
        }
        assert "secret-token" not in str(failed)
        assert server.sessions[session_id].history == []
        assert runtime.discarded_sessions == [session_id]

        completed = await _send_to(server, messages, _prompt(4, session_id, "recover"))
        assert completed["result"] == {"stopReason": "end_turn"}
        assert runtime.requests[1].history == ()
        await server.close()

    asyncio.run(exercise())


def test_update_send_failure_discards_runtime_state(monkeypatch):
    _set_provider_environment(monkeypatch)
    runtime = RecordingRuntime([RunResult("oversized answer"), RunResult("recovered")])
    factory = RecordingFactory(lambda: runtime)

    async def exercise() -> None:
        messages: list[dict[str, Any]] = []
        fail_update = True

        async def send(message):  # noqa: ANN001
            nonlocal fail_update
            if message.get("method") == "session/update" and fail_update:
                fail_update = False
                raise RuntimeError("ACP response exceeds the 8 MiB limit")
            messages.append(dict(message))

        server = ACPStdioServer(_spec(), factory, send)
        await _send_to(server, messages, _initialize())
        created = await _send_to(server, messages, _new_session())
        session_id = created["result"]["sessionId"]

        failed = await _send_to(server, messages, _prompt(3, session_id, "too large"))
        assert failed["error"] == {
            "code": -32603,
            "message": "AgentKit ACP request failed",
            "data": {"code": "RuntimeError"},
        }
        assert server.sessions[session_id].history == []
        assert runtime.discarded_sessions == [session_id]

        completed = await _send_to(server, messages, _prompt(4, session_id, "recover"))
        assert completed["result"] == {"stopReason": "end_turn"}
        assert runtime.requests[1].history == ()
        await server.close()

    asyncio.run(exercise())


def test_multibyte_output_is_split_without_partial_commit(monkeypatch):
    _set_provider_environment(monkeypatch)
    result_text = "a" + "🙂" * (acp._MAX_ASSISTANT_MESSAGE_CHUNK_BYTES // 4 + 1)  # noqa: SLF001
    runtime = RecordingRuntime([RunResult(result_text)])
    factory = RecordingFactory(lambda: runtime)

    async def exercise() -> None:
        messages: list[dict[str, Any]] = []
        second_update_started = asyncio.Event()
        release_second_update = asyncio.Event()
        update_count = 0

        async def send(message):  # noqa: ANN001
            nonlocal update_count
            if message.get("method") == "session/update":
                update_count += 1
                if update_count == 2:
                    second_update_started.set()
                    await release_second_update.wait()
            messages.append(dict(message))

        server = ACPStdioServer(_spec(), factory, send)
        await _send_to(server, messages, _initialize())
        created = await _send_to(server, messages, _new_session())
        session_id = created["result"]["sessionId"]

        await server.accept(_prompt(3, session_id, "multibyte output"))
        await asyncio.wait_for(second_update_started.wait(), timeout=1)
        updates = [message for message in messages if message.get("method") == "session/update"]
        assert len(updates) == 1
        assert server.sessions[session_id].history == []

        release_second_update.set()
        await server.wait_idle()
        updates = [message for message in messages if message.get("method") == "session/update"]
        chunks = [message["params"]["update"]["content"]["text"] for message in updates]

        assert "".join(chunks) == result_text
        assert [len(chunk.encode()) for chunk in chunks] == [
            acp._MAX_ASSISTANT_MESSAGE_CHUNK_BYTES - 3,  # noqa: SLF001
            8,
        ]
        assert server.sessions[session_id].history == [
            ConversationTurn(role="user", text="multibyte output"),
            ConversationTurn(role="assistant", text=result_text),
        ]
        assert _response(messages, 3)["result"] == {"stopReason": "end_turn"}
        await server.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("cancel_method", ["session/cancel", "$/cancel_request"])
def test_cancellation_while_update_send_waits_does_not_commit_history(
    monkeypatch,
    cancel_method: str,
):
    _set_provider_environment(monkeypatch)
    runtime = RecordingRuntime([RunResult("late answer"), RunResult("clean answer")])
    factory = RecordingFactory(lambda: runtime)

    async def exercise() -> None:
        messages: list[dict[str, Any]] = []
        update_started = asyncio.Event()
        release_update = asyncio.Event()

        async def send(message):  # noqa: ANN001
            if (
                message.get("method") == "session/update"
                and not update_started.is_set()
            ):
                update_started.set()
                await release_update.wait()
            messages.append(dict(message))

        server = ACPStdioServer(_spec(), factory, send)
        await _send_to(server, messages, _initialize())
        created = await _send_to(server, messages, _new_session())
        session_id = created["result"]["sessionId"]

        await server.accept(_prompt(3, session_id, "cancel during update"))
        await asyncio.wait_for(update_started.wait(), timeout=1)
        params = (
            {"sessionId": session_id}
            if cancel_method == "session/cancel"
            else {"requestId": 3}
        )
        await server.accept({"jsonrpc": "2.0", "method": cancel_method, "params": params})
        release_update.set()
        await server.wait_idle()

        assert _response(messages, 3)["result"] == {"stopReason": "cancelled"}
        assert server.sessions[session_id].history == []
        assert runtime.discarded_sessions == [session_id]

        completed = await _send_to(server, messages, _prompt(4, session_id, "try again"))
        assert completed["result"] == {"stopReason": "end_turn"}
        assert runtime.requests[1].history == ()
        await server.close()

    asyncio.run(exercise())


def test_prompt_rejects_capability_gated_content_without_running_model(monkeypatch):
    _set_provider_environment(monkeypatch)
    runtime = RecordingRuntime([RunResult("must not run")])
    factory = RecordingFactory(lambda: runtime)

    async def exercise() -> None:
        messages: list[dict[str, Any]] = []

        async def send(message):  # noqa: ANN001
            messages.append(dict(message))

        server = ACPStdioServer(_spec(), factory, send)
        await _send_to(server, messages, _initialize())
        created = await _send_to(server, messages, _new_session())
        session_id = created["result"]["sessionId"]
        invalid = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "session/prompt",
            "params": {
                "sessionId": session_id,
                "prompt": [{"type": "image", "data": "ignored", "mimeType": "image/png"}],
            },
        }

        failed = await _send_to(server, messages, invalid)
        assert failed["error"]["code"] == -32602
        assert "only text and resource_link" in failed["error"]["message"]
        assert runtime.requests == []
        await server.close()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "block",
    [
        {"type": "resource_link", "uri": "https://example.com/file.txt"},
        {"type": "resource_link", "name": "file"},
        {
            "type": "resource_link",
            "name": "file",
            "uri": "https://example.com/file.txt",
            "mimeType": "",
        },
    ],
)
def test_prompt_rejects_malformed_resource_links_without_running_model(monkeypatch, block):
    _set_provider_environment(monkeypatch)
    runtime = RecordingRuntime([RunResult("must not run")])
    factory = RecordingFactory(lambda: runtime)

    async def exercise() -> None:
        messages: list[dict[str, Any]] = []

        async def send(message):  # noqa: ANN001
            messages.append(dict(message))

        server = ACPStdioServer(_spec(), factory, send)
        await _send_to(server, messages, _initialize())
        created = await _send_to(server, messages, _new_session())
        session_id = created["result"]["sessionId"]

        failed = await _send_to(
            server,
            messages,
            _prompt_content(3, session_id, [block]),
        )
        assert failed["error"]["code"] == -32602
        assert runtime.requests == []
        await server.close()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "server",
    [
        _orka_mcp_server("https://example.com/mcp"),
        {**_orka_mcp_server(), "headers": []},
        {**_orka_mcp_server(), "type": "stdio", "command": "tool"},
    ],
)
def test_session_rejects_unsafe_or_unsupported_mcp_without_building_runtime(monkeypatch, server):
    _set_provider_environment(monkeypatch)
    factory = RecordingFactory(lambda: RecordingRuntime([RunResult("unused")]), supports_http_mcp=True)

    async def exercise() -> None:
        messages: list[dict[str, Any]] = []

        async def send(message):  # noqa: ANN001
            messages.append(dict(message))

        dispatcher = ACPStdioServer(_spec(), factory, send)
        await _send_to(dispatcher, messages, _initialize())
        failed = await _send_to(dispatcher, messages, _new_session(mcp_servers=[server]))
        assert failed["error"]["code"] == -32602
        assert factory.specs == []
        await dispatcher.close()

    asyncio.run(exercise())


def test_failed_runtime_build_restores_projected_environment(monkeypatch):
    _set_provider_environment(monkeypatch)
    monkeypatch.setattr(acp.secrets, "token_hex", lambda size: "a" * (size * 2))
    prefix = "AGENTKIT_ACP_SESSION_" + "A" * 32 + "_MCP_0"
    url_env = prefix + "_URL"
    header_env = prefix + "_HEADER_0"
    monkeypatch.setenv(url_env, "previous-url")
    monkeypatch.setenv(header_env, "previous-header")

    class FailingFactory:
        def supports_acp_http_mcp(self) -> bool:
            return True

        def build_runtime(self, spec):  # noqa: ANN001
            assert os.environ[url_env] == _orka_mcp_server()["url"]
            assert os.environ[header_env] == "Bearer mcp-session-token"
            raise RuntimeError("build failed")

    async def exercise() -> None:
        messages: list[dict[str, Any]] = []

        async def send(message):  # noqa: ANN001
            messages.append(dict(message))

        server = ACPStdioServer(_spec(), FailingFactory(), send)
        await _send_to(server, messages, _initialize())
        failed = await _send_to(
            server,
            messages,
            _new_session(mcp_servers=[_orka_mcp_server()]),
        )
        assert failed["error"]["code"] == -32603
        assert os.environ[url_env] == "previous-url"
        assert os.environ[header_env] == "previous-header"
        await server.close()

    asyncio.run(exercise())


def test_runtime_binding_verifies_exact_config_digest_model_and_provider(monkeypatch, tmp_path):
    config = tmp_path / "agent.yaml"
    config_bytes = b"abiVersion: v0\nmetadata:\n  name: exact-bytes\n"
    config.write_bytes(config_bytes)
    monkeypatch.setenv(
        acp.ACP_AGENT_CONFIGURATION_DIGEST_ENV,
        "sha256:" + hashlib.sha256(config_bytes).hexdigest(),
    )
    monkeypatch.setenv(acp.ACP_MODEL_ENV, "test-model")
    _set_provider_environment(monkeypatch)

    acp.validate_acp_runtime_binding(config_bytes, _spec())

    with pytest.raises(ACPConfigurationError, match="exact agent config bytes"):
        acp.validate_acp_runtime_binding(config_bytes + b"# changed\n", _spec())


def test_runtime_binding_accepts_unicode_model_name(monkeypatch):
    config_bytes = b"exact"
    model_name = "模型"
    monkeypatch.setenv(
        acp.ACP_AGENT_CONFIGURATION_DIGEST_ENV,
        "sha256:" + hashlib.sha256(config_bytes).hexdigest(),
    )
    monkeypatch.setenv(acp.ACP_MODEL_ENV, model_name)
    _set_provider_environment(monkeypatch)

    acp.validate_acp_runtime_binding(
        config_bytes,
        _spec(
            model={
                "provider": "openai-compatible",
                "baseURL": "https://baked.example.invalid/v1",
                "name": model_name,
                "apiKeyEnv": "BAKED_MODEL_TOKEN",
            }
        ),
    )


def test_verified_runtime_binding_parses_the_hashed_bytes_after_file_replacement(monkeypatch, tmp_path):
    config = tmp_path / "agent.yaml"
    verified_bytes = b"""abiVersion: v0
metadata:
  name: verified
model:
  provider: openai-compatible
  baseURL: https://baked.example.invalid/v1
  name: test-model
  apiKeyEnv: TEST
instructions: Verified instructions.
tools: []
expose:
  openai: true
  port: 8080
"""
    replaced_bytes = verified_bytes.replace(b"Verified instructions.", b"Replaced instructions.")
    config.write_bytes(verified_bytes)
    monkeypatch.setenv(
        acp.ACP_AGENT_CONFIGURATION_DIGEST_ENV,
        "sha256:" + hashlib.sha256(verified_bytes).hexdigest(),
    )
    monkeypatch.setenv(acp.ACP_MODEL_ENV, "test-model")
    _set_provider_environment(monkeypatch)

    original_read_bytes = Path.read_bytes
    reads = 0

    def read_then_replace(path: Path) -> bytes:
        nonlocal reads
        raw = original_read_bytes(path)
        if path == config:
            reads += 1
            config.write_bytes(replaced_bytes)
        return raw

    monkeypatch.setattr(Path, "read_bytes", read_then_replace)

    spec = acp.load_verified_acp_runtime_binding(config)

    assert reads == 1
    assert spec.instructions == "Verified instructions."
    assert original_read_bytes(config) == replaced_bytes


@pytest.mark.parametrize(
    ("environment_name", "environment_value", "message"),
    [
        (acp.ACP_AGENT_CONFIGURATION_DIGEST_ENV, "sha256:short", "lowercase sha256"),
        (acp.ACP_MODEL_ENV, "other-model", "does not match model.name"),
        (acp.ACP_PROVIDER_BASE_URL_ENV, "https://provider.example.com/v1", "loopback"),
        (acp.ACP_PROVIDER_TOKEN_ENV, "", "non-empty"),
    ],
)
def test_runtime_binding_rejects_profile_or_provider_mismatch(
    monkeypatch,
    tmp_path,
    environment_name: str,
    environment_value: str,
    message: str,
):
    config = tmp_path / "agent.yaml"
    config.write_bytes(b"exact")
    monkeypatch.setenv(
        acp.ACP_AGENT_CONFIGURATION_DIGEST_ENV,
        "sha256:" + hashlib.sha256(b"exact").hexdigest(),
    )
    monkeypatch.setenv(acp.ACP_MODEL_ENV, "test-model")
    _set_provider_environment(monkeypatch)
    monkeypatch.setenv(environment_name, environment_value)

    with pytest.raises(ACPConfigurationError, match=message):
        acp.validate_acp_runtime_binding(b"exact", _spec())


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"tools": [{"name": "local", "command": ["local-tool"]}]}, "direct tools"),
        (
            {
                "brokeredTools": [
                    {
                        "name": "lookup",
                        "description": "Look up a value.",
                        "brokeredClass": "read",
                        "parameters": {"type": "object"},
                    }
                ]
            },
            "brokeredTools",
        ),
        (
            {
                "context": {
                    "providers": [
                        {
                            "name": "skills",
                            "type": "skills",
                            "source": "filesystem",
                            "path": "/agent/skills",
                        }
                    ]
                }
            },
            "context providers",
        ),
    ],
)
def test_runtime_binding_rejects_baked_tool_and_context_paths(tmp_path, override, message):
    config = tmp_path / "agent.yaml"
    config.write_bytes(b"unused")

    with pytest.raises(ACPConfigurationError, match=message):
        acp.validate_acp_runtime_binding(b"unused", _spec(**override))
