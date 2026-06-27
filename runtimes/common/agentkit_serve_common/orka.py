"""Orka harness protocol adapter over the AgentKit RuntimeSession seam.

This module exposes the observed-mode ``orka.harness.v1`` HTTP+SSE contract while
reusing the same loaded ``AgentSpec`` and runtime adapter used by the standalone
OpenAI and Foundry protocol skins. AgentKit does not enforce Orka policy here: it
accepts a turn, runs the local AgentKit-owned runtime/tools, and reports lifecycle
frames honestly for Orka to govern upstream.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Awaitable, Callable, Mapping

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from .config import AgentSpec
from .conversation import RunRequest
from .runtime import AgentRunError, RunResult, RuntimeFactory
from .server import make_auth_dependency

ORKA_HARNESS_VERSION = "orka.harness.v1"
HTTP_TRANSPORT = "http+sse"
PROVIDER_KIND_KUBERNETES_SERVICE = "kubernetes-service"
TOOL_MODE_OBSERVED = "observed"
_TERMINAL_TYPES = frozenset({"TurnCompleted", "TurnFailed", "TurnCancelled"})
_DEFAULT_MAX_TERMINAL_TURNS = 256
_DEFAULT_MAX_RUNTIME_SESSIONS = 64
_MAX_TERMINAL_TURNS_ENV = "AGENTKIT_ORKA_MAX_TERMINAL_TURNS"
_MAX_RUNTIME_SESSIONS_ENV = "AGENTKIT_ORKA_MAX_RUNTIME_SESSIONS"
_TURN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass
class ActiveRuntime:
    context: Any
    session: Any


@contextmanager
def _scoped_process_env(env: Mapping[str, str]):
    if not env:
        yield
        return
    old_values: dict[str, str | None] = {name: os.environ.get(name) for name in env}
    try:
        os.environ.update(env)
        yield
    finally:
        for name, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old_value


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class TurnEvent:
    seq: int
    type: str
    runtime_session_id: str
    turn_id: str
    correlation_id: str
    created_at: str = field(default_factory=_now_iso)
    severity: str = "info"
    summary: str = ""
    content: Mapping[str, Any] | None = None
    content_text: str = ""
    completed: Mapping[str, Any] | None = None
    failed: Mapping[str, Any] | None = None
    error: Mapping[str, Any] | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    @property
    def terminal(self) -> bool:
        return self.type in _TERMINAL_TYPES

    def as_frame(self) -> dict[str, Any]:
        return {
            "version": ORKA_HARNESS_VERSION,
            "type": self.type,
            "runtimeSessionID": self.runtime_session_id,
            "turnID": self.turn_id,
            "correlationID": self.correlation_id,
            "seq": self.seq,
            "createdAt": self.created_at,
            "severity": self.severity,
            "summary": self.summary,
            "content": dict(self.content or {}),
            "contentText": self.content_text,
            "completed": dict(self.completed) if self.completed is not None else None,
            "failed": dict(self.failed) if self.failed is not None else None,
            "error": dict(self.error) if self.error is not None else None,
            "metadata": dict(self.metadata),
        }


class TurnState:
    """Buffered turn event state with replay support for SSE clients."""

    def __init__(
        self,
        *,
        runtime_session_id: str,
        turn_id: str,
        correlation_id: str,
        metadata: Mapping[str, str] | None = None,
        task: asyncio.Task[None] | None = None,
    ) -> None:
        self.runtime_session_id = runtime_session_id
        self.turn_id = turn_id
        self.correlation_id = correlation_id
        self.metadata = dict(metadata or {})
        self.task = task
        self.events: list[TurnEvent] = []
        self.condition = asyncio.Condition()
        self.terminal_event: TurnEvent | None = None

    async def append(
        self,
        event_type: str,
        *,
        severity: str = "info",
        summary: str = "",
        content: Mapping[str, Any] | None = None,
        content_text: str = "",
        completed: Mapping[str, Any] | None = None,
        failed: Mapping[str, Any] | None = None,
        error: Mapping[str, Any] | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> tuple[TurnEvent, bool]:
        async with self.condition:
            if event_type in _TERMINAL_TYPES and self.terminal_event is not None:
                return self.terminal_event, False
            seq = len(self.events) + 1
            if event_type == "TurnCompleted":
                completed = dict(completed or {})
                completed.setdefault("finalEventSeq", seq)
            event = TurnEvent(
                seq=seq,
                type=event_type,
                runtime_session_id=self.runtime_session_id,
                turn_id=self.turn_id,
                correlation_id=self.correlation_id,
                severity=severity,
                summary=summary,
                content=content,
                content_text=content_text,
                completed=completed,
                failed=failed,
                error=error,
                metadata=metadata or self.metadata,
            )
            self.events.append(event)
            if event.terminal:
                self.terminal_event = event
            self.condition.notify_all()
            return event, True

    async def events_after(self, seq: int) -> list[TurnEvent]:
        async with self.condition:
            while True:
                events = [event for event in self.events if event.seq > seq]
                if events or self.terminal_event is not None:
                    return events
                await self.condition.wait()


def _positive_int_setting(value: int | None, *, env_name: str, default: int, field_name: str) -> int:
    if value is not None:
        if value < 1:
            raise ValueError(f"{field_name} must be at least 1")
        return value
    raw = os.environ.get(env_name)
    if not raw:
        return default
    try:
        parsed = int(raw)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _max_terminal_turns(value: int | None = None) -> int:
    return _positive_int_setting(
        value,
        env_name=_MAX_TERMINAL_TURNS_ENV,
        default=_DEFAULT_MAX_TERMINAL_TURNS,
        field_name="max_terminal_turns",
    )


def _max_runtime_sessions(value: int | None = None) -> int:
    return _positive_int_setting(
        value,
        env_name=_MAX_RUNTIME_SESSIONS_ENV,
        default=_DEFAULT_MAX_RUNTIME_SESSIONS,
        field_name="max_runtime_sessions",
    )


def _record_terminal_turn(
    turn_id: str,
    terminal_order: list[str],
    turns: dict[str, TurnState],
    max_terminal_turns: int,
) -> None:
    if turn_id in terminal_order:
        terminal_order.remove(turn_id)
    terminal_order.append(turn_id)
    overflow = len(terminal_order) - max_terminal_turns
    if overflow <= 0:
        return
    for evict_id in terminal_order[:overflow]:
        turns.pop(evict_id, None)
    del terminal_order[:overflow]


async def _append_terminal_if_missing(
    state: TurnState,
    terminal_order: list[str],
    turns: dict[str, TurnState],
    max_terminal_turns: int,
    event_type: str,
    *,
    summary: str,
    failed: Mapping[str, Any] | None = None,
    completed: Mapping[str, Any] | None = None,
    error: Mapping[str, Any] | None = None,
) -> None:
    _, created = await state.append(
        event_type,
        severity="error" if event_type == "TurnFailed" else "info",
        summary=summary,
        completed=completed,
        failed=failed,
        error=error,
    )
    if created:
        _record_terminal_turn(state.turn_id, terminal_order, turns, max_terminal_turns)


def _ensure_terminal_on_task_done(
    task: asyncio.Task[None],
    state: TurnState,
    terminal_order: list[str],
    turns: dict[str, TurnState],
    max_terminal_turns: int,
) -> None:
    if state.terminal_event is not None:
        return
    if task.cancelled():
        event_type = "TurnCancelled"
        summary = "turn cancelled"
        failed = None
        error = None
    else:
        exc = task.exception()
        event_type = "TurnFailed"
        code = exc.__class__.__name__ if exc is not None else "RuntimeTaskEndedWithoutTerminal"
        message = str(exc) if exc is not None else "runtime task ended without a terminal frame"
        summary = "turn failed"
        failed = {"reason": code, "message": message, "retryable": False}
        error = {"code": code, "message": message, "retryable": False}
    asyncio.create_task(
        _append_terminal_if_missing(
            state,
            terminal_order,
            turns,
            max_terminal_turns,
            event_type,
            summary=summary,
            failed=failed,
            error=error,
        )
    )


def _sse_frame(event: TurnEvent) -> str:
    data = json.dumps(event.as_frame(), separators=(",", ":"), sort_keys=True)
    return f"id: {event.seq}\nevent: {event.type}\ndata: {data}\n\n"


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _required_string(data: Mapping[str, Any], field_name: str) -> str:
    value = _clean(data.get(field_name))
    if value is None:
        raise HTTPException(status_code=400, detail=f"{field_name} is required")
    return value


def _turn_id_from_payload(data: Mapping[str, Any]) -> str:
    turn_id = _required_string(data, "turnID")
    if not _TURN_ID_RE.fullmatch(turn_id):
        raise HTTPException(
            status_code=400,
            detail="turnID must be URL-safe: letters, numbers, '.', '_', or '-'",
        )
    return turn_id


def _mapping_of_strings(data: Any, *, field_name: str) -> dict[str, str]:
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail=f"{field_name} must be an object")
    out: dict[str, str] = {}
    for key, value in data.items():
        if not isinstance(key, str):
            raise HTTPException(status_code=400, detail=f"{field_name} keys must be strings")
        if not isinstance(value, str):
            raise HTTPException(status_code=400, detail=f"{field_name}.{key} must be a string")
        out[key] = value
    return out


def _parse_deadline(value: Any) -> datetime:
    if value in (None, ""):
        raise HTTPException(status_code=400, detail="deadline is required")
    if not isinstance(value, str):
        raise HTTPException(status_code=400, detail="deadline must be an RFC3339 timestamp string")
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="deadline must be an RFC3339 timestamp string") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _allowed_turn_env_names(spec: AgentSpec) -> set[str]:
    names = {entry.name for entry in spec.env}
    if spec.model.api_key_env:
        names.add(spec.model.api_key_env)
    if spec.model.auth and spec.model.auth.token_env:
        names.add(spec.model.auth.token_env)
    for tool in spec.tools:
        names.update(tool.env)
        if tool.url_env:
            names.add(tool.url_env)
        for header in tool.headers:
            if header.value_env:
                names.add(header.value_env)
        if tool.auth and tool.auth.token_env:
            names.add(tool.auth.token_env)
    for provider in spec.context.providers:
        for name in (provider.endpoint_env, provider.index_env, provider.store_name_env):
            if name:
                names.add(name)
        if provider.auth and provider.auth.token_env:
            names.add(provider.auth.token_env)
    for name in (spec.observability.otel.endpoint_env, spec.observability.logs.level_env):
        if name:
            names.add(name)
    return names


def _env_from_input(input_value: Mapping[str, Any], *, allowed_names: set[str]) -> dict[str, str]:
    raw_env = input_value.get("env", [])
    if raw_env in (None, ""):
        return {}
    if not isinstance(raw_env, list):
        raise HTTPException(status_code=400, detail="input.env must be an array")
    env: dict[str, str] = {}
    for idx, item in enumerate(raw_env):
        if not isinstance(item, dict):
            raise HTTPException(status_code=400, detail=f"input.env[{idx}] must be an object")
        name = _clean(item.get("name"))
        if name is None:
            raise HTTPException(status_code=400, detail=f"input.env[{idx}].name is required")
        if not _ENV_NAME_RE.fullmatch(name):
            raise HTTPException(status_code=400, detail=f"input.env[{idx}].name is invalid")
        if name.startswith("AGENTKIT_"):
            raise HTTPException(status_code=400, detail=f"input.env[{idx}].name {name!r} is reserved")
        if name not in allowed_names:
            raise HTTPException(status_code=400, detail=f"input.env[{idx}].name {name!r} is not declared by this agent")
        value = item.get("value", "")
        if not isinstance(value, str):
            raise HTTPException(status_code=400, detail=f"input.env[{idx}].value must be a string")
        env[name] = value
    return env


def _context_refs_from_input(input_value: Mapping[str, Any]) -> list[dict[str, Any]]:
    refs = input_value.get("contextRefs", [])
    if refs in (None, ""):
        return []
    if not isinstance(refs, list):
        raise HTTPException(status_code=400, detail="input.contextRefs must be an array")
    out: list[dict[str, Any]] = []
    for idx, item in enumerate(refs):
        if not isinstance(item, dict):
            raise HTTPException(status_code=400, detail=f"input.contextRefs[{idx}] must be an object")
        kind = _clean(item.get("kind"))
        if kind is None:
            raise HTTPException(status_code=400, detail=f"input.contextRefs[{idx}].kind is required")
        name = _clean(item.get("name"))
        if name is None:
            raise HTTPException(status_code=400, detail=f"input.contextRefs[{idx}].name is required")
        seq = item.get("seq", 0)
        if not isinstance(seq, int) or seq < 0:
            raise HTTPException(status_code=400, detail=f"input.contextRefs[{idx}].seq must be non-negative")
        ref = {"kind": kind, "name": name}
        if seq:
            ref["seq"] = seq
        out.append(ref)
    return out


def _validate_auth_identity(data: Mapping[str, Any]) -> None:
    raw = data.get("authIdentity")
    if not isinstance(raw, dict):
        raise HTTPException(status_code=400, detail="authIdentity is required")
    if _clean(raw.get("subject")) is None and _clean(raw.get("username")) is None:
        raise HTTPException(status_code=400, detail="authIdentity.subject or authIdentity.username is required")


def _request_to_run_request(data: dict[str, Any], *, turn_id: str, spec: AgentSpec) -> RunRequest:
    version = data.get("version")
    if version != ORKA_HARNESS_VERSION:
        raise HTTPException(status_code=400, detail=f"version must be {ORKA_HARNESS_VERSION!r}")
    namespace = _required_string(data, "namespace")
    task_name = _required_string(data, "taskName")
    session_name = _required_string(data, "sessionName")
    runtime_session_id = _required_string(data, "runtimeSessionID")
    correlation_id = _required_string(data, "correlationID")
    deadline = _parse_deadline(data.get("deadline"))
    _validate_auth_identity(data)
    tool_mode = _clean(data.get("toolExecutionMode"))
    if tool_mode not in (None, "", TOOL_MODE_OBSERVED):
        raise HTTPException(status_code=400, detail=f"unsupported toolExecutionMode {tool_mode!r}")
    if data.get("eventCursor", 0) not in (None, ""):
        event_cursor = data.get("eventCursor", 0)
        if not isinstance(event_cursor, int) or event_cursor < 0:
            raise HTTPException(status_code=400, detail="eventCursor must be non-negative")

    input_value = data.get("input")
    if not isinstance(input_value, dict):
        raise HTTPException(status_code=400, detail="input must be an object")
    if "prompt" not in input_value:
        raise HTTPException(status_code=400, detail="input.prompt is required")
    prompt = input_value.get("prompt")
    if not isinstance(prompt, str):
        raise HTTPException(status_code=400, detail="input.prompt must be a string")
    context_refs = _context_refs_from_input(input_value)
    metadata = _mapping_of_strings(data.get("metadata"), field_name="metadata")
    if context_refs:
        metadata["contextRefs"] = json.dumps(context_refs, separators=(",", ":"), sort_keys=True)
    _ = (namespace, task_name, session_name)

    return RunRequest(
        prompt=prompt,
        history=(),
        session_id=runtime_session_id,
        env=_env_from_input(input_value, allowed_names=_allowed_turn_env_names(spec)),
        deadline=deadline,
        turn_id=turn_id,
        correlation_id=correlation_id,
        metadata=metadata,
    )


def _usage_payload(result: RunResult) -> dict[str, int]:
    usage = result.usage or {}
    return {key: int(usage.get(key, 0) or 0) for key in sorted(usage)}


def health_response(spec: AgentSpec) -> dict[str, Any]:
    return {
        "version": ORKA_HARNESS_VERSION,
        "status": "ok",
        "ready": True,
        "checkedAt": _now_iso(),
        "metadata": {"agentName": spec.metadata.name},
    }


def capabilities_response(spec: AgentSpec) -> dict[str, Any]:
    return {
        "version": ORKA_HARNESS_VERSION,
        "protocolVersion": ORKA_HARNESS_VERSION,
        "transport": HTTP_TRANSPORT,
        "runtimeName": "agentkit-serve",
        "runtimeVersion": "0.0.0",
        "providerKind": PROVIDER_KIND_KUBERNETES_SERVICE,
        "toolExecutionModes": [TOOL_MODE_OBSERVED],
        "supportsCancel": True,
        "supportsRuntimeSessions": True,
        "supportsSuspend": False,
        "supportsWorkspaceSnapshot": False,
        "maxConcurrentTurns": 1,
        "metadata": {
            "agentName": spec.metadata.name,
            "model": spec.model.name,
            "agentkitProvider": spec.model.provider,
        },
    }


def start_turn_response(turn: TurnState) -> dict[str, Any]:
    return {
        "version": ORKA_HARNESS_VERSION,
        "accepted": True,
        "runtimeSessionID": turn.runtime_session_id,
        "turnID": turn.turn_id,
        "correlationID": turn.correlation_id,
        "eventStreamPath": f"/v1/turns/{turn.turn_id}/events",
    }


def cancel_turn_response(turn: TurnState, request_data: Mapping[str, Any]) -> dict[str, Any]:  # noqa: ARG001 - request is validated before response.
    return {
        "version": ORKA_HARNESS_VERSION,
        "accepted": True,
        "runtimeSessionID": turn.runtime_session_id,
        "turnID": turn.turn_id,
        "correlationID": turn.correlation_id,
        "message": "cancel accepted",
    }


def _frame_base(
    turn: TurnState,
    seq: int,
    frame_type: str,
    *,
    severity: str = "info",
    summary: str = "",
    content: Mapping[str, Any] | None = None,
    content_text: str = "",
    completed: Mapping[str, Any] | None = None,
    failed: Mapping[str, Any] | None = None,
    error: Mapping[str, Any] | None = None,
    metadata: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "version": ORKA_HARNESS_VERSION,
        "type": frame_type,
        "runtimeSessionID": turn.runtime_session_id,
        "turnID": turn.turn_id,
        "correlationID": turn.correlation_id,
        "seq": seq,
        "createdAt": _now_iso(),
        "severity": severity,
        "summary": summary,
        "content": dict(content or {}),
        "contentText": content_text,
        "completed": dict(completed) if completed is not None else None,
        "failed": dict(failed) if failed is not None else None,
        "error": dict(error) if error is not None else None,
        "metadata": dict(metadata or turn.metadata),
    }


def frame_started(turn: TurnState, seq: int) -> dict[str, Any]:
    return _frame_base(turn, seq, "TurnStarted", summary="turn started")


def frame_runtime_output(turn: TurnState, seq: int, text: str, content: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return _frame_base(turn, seq, "RuntimeOutput", summary="runtime output", content=content, content_text=text)


def frame_completed(turn: TurnState, seq: int, result_text: str) -> dict[str, Any]:
    return _frame_base(
        turn,
        seq,
        "TurnCompleted",
        summary="turn completed",
        completed={"result": result_text, "finalEventSeq": seq},
    )


def frame_failed(turn: TurnState, seq: int, reason: str, message: str, retryable: bool = False) -> dict[str, Any]:
    failed = {"reason": reason, "message": message, "retryable": retryable}
    return _frame_base(
        turn,
        seq,
        "TurnFailed",
        severity="error",
        summary="turn failed",
        failed=failed,
        error={"code": reason, "message": message, "retryable": retryable},
    )


def frame_cancelled(turn: TurnState, seq: int, reason: str = "cancelled") -> dict[str, Any]:
    return _frame_base(turn, seq, "TurnCancelled", summary=f"turn {reason}")


async def _run_turn(
    get_runtime: Callable[[RunRequest], Awaitable[Any]],
    turns: dict[str, TurnState],
    terminal_order: list[str],
    state: TurnState,
    run_request: RunRequest,
    *,
    max_terminal_turns: int,
) -> None:
    try:
        runtime = await get_runtime(run_request)
        if run_request.deadline is None:
            with _scoped_process_env(run_request.env):
                result = await runtime.run(run_request)
        else:
            seconds = (run_request.deadline - datetime.now(UTC)).total_seconds()
            if seconds <= 0:
                raise TimeoutError("turn deadline has already expired")
            async with asyncio.timeout(seconds):
                with _scoped_process_env(run_request.env):
                    result = await runtime.run(run_request)
    except asyncio.CancelledError:
        await _append_terminal_if_missing(
            state,
            terminal_order,
            turns,
            max_terminal_turns,
            "TurnCancelled",
            summary="turn cancelled",
        )
        return
    except TimeoutError as exc:
        await _append_terminal_if_missing(
            state,
            terminal_order,
            turns,
            max_terminal_turns,
            "TurnFailed",
            summary="turn deadline exceeded",
            failed={"reason": "DeadlineExceeded", "message": str(exc) or "turn deadline exceeded", "retryable": False},
            error={"code": "DeadlineExceeded", "message": str(exc) or "turn deadline exceeded", "retryable": False},
        )
        return
    except AgentRunError as exc:
        code = exc.code or exc.__class__.__name__
        await _append_terminal_if_missing(
            state,
            terminal_order,
            turns,
            max_terminal_turns,
            "TurnFailed",
            summary="turn failed",
            failed={"reason": code, "message": str(exc), "retryable": False},
            error={"code": code, "message": str(exc), "retryable": False},
        )
        return
    except Exception as exc:  # noqa: BLE001 - protocol envelope must be deterministic.
        code = exc.__class__.__name__
        await _append_terminal_if_missing(
            state,
            terminal_order,
            turns,
            max_terminal_turns,
            "TurnFailed",
            summary="turn failed",
            failed={"reason": code, "message": str(exc), "retryable": False},
            error={"code": code, "message": str(exc), "retryable": False},
        )
        return

    if result.text:
        await state.append(
            "RuntimeOutput",
            summary="runtime output",
            content={"message": result.text, "usage": _usage_payload(result)},
            content_text=result.text,
        )
    await _append_terminal_if_missing(
        state,
        terminal_order,
        turns,
        max_terminal_turns,
        "TurnCompleted",
        summary="turn completed",
        completed={"result": result.text},
    )


def create_orka_app(
    spec: AgentSpec,
    factory: RuntimeFactory,
    auth_token: str | None = None,
    *,
    max_terminal_turns: int | None = None,
    max_runtime_sessions: int | None = None,
) -> FastAPI:
    """Create an observed-mode Orka harness app for one AgentKit runtime."""
    if not auth_token:
        raise ValueError("Orka mode requires a bearer auth token")
    retention_limit = _max_terminal_turns(max_terminal_turns)
    runtime_session_limit = _max_runtime_sessions(max_runtime_sessions)
    turns: dict[str, TurnState] = {}
    terminal_order: list[str] = []
    active_runtimes: dict[str, ActiveRuntime] = {}
    runtime_order: list[str] = []

    async def get_runtime(run_request: RunRequest) -> Any:
        runtime_session_id = run_request.session_id or ""
        active = active_runtimes.get(runtime_session_id)
        if active is not None:
            if runtime_session_id in runtime_order:
                runtime_order.remove(runtime_session_id)
            runtime_order.append(runtime_session_id)
            return active.session
        with _scoped_process_env(run_request.env):
            context = factory.build_runtime(spec)
            session = await context.__aenter__()
        active_runtimes[runtime_session_id] = ActiveRuntime(context=context, session=session)
        runtime_order.append(runtime_session_id)
        while len(runtime_order) > runtime_session_limit:
            evict_id = runtime_order.pop(0)
            evicted = active_runtimes.pop(evict_id, None)
            if evicted is not None:
                await evicted.context.__aexit__(None, None, None)
        return session

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.turns = turns
        app.state.active_runtimes = active_runtimes
        try:
            yield
        finally:
            for state in turns.values():
                if state.task is not None and not state.task.done():
                    state.task.cancel()
            for active in reversed(list(active_runtimes.values())):
                await active.context.__aexit__(None, None, None)
            active_runtimes.clear()
            runtime_order.clear()

    app = FastAPI(title="agentkit-serve-orka", lifespan=lifespan)
    auth = Depends(make_auth_dependency(auth_token))

    @app.get("/v1/health")
    async def health() -> dict[str, Any]:
        return health_response(spec)

    @app.get("/v1/capabilities")
    async def capabilities() -> dict[str, Any]:
        return capabilities_response(spec)

    @app.post("/v1/turns", dependencies=[auth], status_code=202)
    async def create_turn(request: Request):
        try:
            data = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="Request body must be JSON") from exc
        if not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object")

        turn_id = _turn_id_from_payload(data)
        if turn_id in turns:
            raise HTTPException(status_code=409, detail=f"turn {turn_id!r} already exists")
        if any(state.terminal_event is None for state in turns.values()):
            raise HTTPException(status_code=429, detail="maxConcurrentTurns limit reached")

        run_request = _request_to_run_request(data, turn_id=turn_id, spec=spec)
        runtime_session_id = run_request.session_id or ""
        correlation_id = run_request.correlation_id or ""
        state = TurnState(
            runtime_session_id=runtime_session_id,
            turn_id=turn_id,
            correlation_id=correlation_id,
            metadata=run_request.metadata,
        )
        turns[turn_id] = state
        await state.append("TurnStarted", summary="turn started")
        state.task = asyncio.create_task(
            _run_turn(
                get_runtime,
                turns,
                terminal_order,
                state,
                run_request,
                max_terminal_turns=retention_limit,
            )
        )
        state.task.add_done_callback(
            lambda task, state=state: _ensure_terminal_on_task_done(
                task,
                state,
                terminal_order,
                turns,
                retention_limit,
            )
        )
        return start_turn_response(state)

    @app.get("/v1/turns/{turn_id}/events", dependencies=[auth])
    async def turn_events(
        turn_id: str,
        request: Request,
        after_seq: int = Query(default=0, ge=0, alias="afterSeq"),
    ):
        state = turns.get(turn_id)
        if state is None:
            raise HTTPException(status_code=404, detail="turn not found")

        async def stream():
            last_seq = after_seq
            while True:
                events = await state.events_after(last_seq)
                if not events:
                    return
                for event in events:
                    last_seq = event.seq
                    yield _sse_frame(event)
                    if event.terminal or await request.is_disconnected():
                        return

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/v1/turns/{turn_id}/cancel", dependencies=[auth], status_code=202)
    async def cancel_turn(turn_id: str, request: Request):
        try:
            data = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="Request body must be JSON") from exc
        if not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object")
        if data.get("version") != ORKA_HARNESS_VERSION:
            raise HTTPException(status_code=400, detail=f"version must be {ORKA_HARNESS_VERSION!r}")
        body_turn_id = _turn_id_from_payload(data)
        if body_turn_id != turn_id:
            raise HTTPException(status_code=400, detail="cancel turnID must match route turnID")
        runtime_session_id = _required_string(data, "runtimeSessionID")
        correlation_id = _required_string(data, "correlationID")
        for field_name in ("namespace", "taskName", "sessionName"):
            _required_string(data, field_name)

        state = turns.get(turn_id)
        if state is None:
            raise HTTPException(status_code=404, detail="turn not found")
        if runtime_session_id != state.runtime_session_id:
            raise HTTPException(status_code=400, detail="cancel runtimeSessionID must match turn runtimeSessionID")
        if correlation_id != state.correlation_id:
            raise HTTPException(status_code=400, detail="cancel correlationID must match turn correlationID")
        if state.terminal_event is None and state.task is not None and not state.task.done():
            state.task.cancel()
        elif state.terminal_event is None:
            await _append_terminal_if_missing(
                state,
                terminal_order,
                turns,
                retention_limit,
                "TurnCancelled" if state.task is None or state.task.cancelled() else "TurnFailed",
                summary="turn cancelled" if state.task is None or state.task.cancelled() else "turn failed",
                failed=None
                if state.task is None or state.task.cancelled()
                else {"reason": "RuntimeTaskEndedWithoutTerminal", "message": "runtime task ended without a terminal frame", "retryable": False},
                error=None
                if state.task is None or state.task.cancelled()
                else {"code": "RuntimeTaskEndedWithoutTerminal", "message": "runtime task ended without a terminal frame", "retryable": False},
            )
        return cancel_turn_response(state, data)

    @app.get("/v1/turns/{turn_id}/output", dependencies=[auth])
    async def turn_output(turn_id: str, ref: str):  # noqa: ARG001 - reserved optional endpoint.
        if turn_id not in turns:
            raise HTTPException(status_code=404, detail="turn not found")
        raise HTTPException(status_code=404, detail="output ref not found")

    return app
