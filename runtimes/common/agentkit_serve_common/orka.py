"""Orka harness protocol adapter over the AgentKit RuntimeSession seam.

This module exposes the observed-mode ``orka.harness.v1`` HTTP+SSE contract while
reusing the same loaded ``AgentSpec`` and runtime adapter used by the standalone
OpenAI and Foundry protocol skins. AgentKit does not enforce Orka policy here: it
accepts a turn, runs the local AgentKit-owned runtime/tools, and reports lifecycle
frames honestly for Orka to govern upstream.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Awaitable, Callable, Mapping
from urllib.parse import quote

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from .config import AgentSpec
from .conversation import RunRequest
from .runtime import (
    AgentRunError,
    BrokeredRuntimeSession,
    BrokeredToolCall,
    BrokeredToolDefinition,
    BrokeredToolResult,
    OfflineEchoRuntimeFactory,
    RunResult,
    RuntimeFactory,
    ToolBroker,
)
from .server import make_auth_dependency

ORKA_HARNESS_VERSION = "orka.harness.v1"
HTTP_TRANSPORT = "http+sse"
PROVIDER_KIND_KUBERNETES_SERVICE = "kubernetes-service"
TOOL_MODE_OBSERVED = "observed"
TOOL_MODE_BROKERED = "brokered"
BROKERED_CLASS_READ = "read"
BROKERED_CLASS_WRITE = "write"
BROKERED_CLASS_COORDINATION = "coordination"
_ENABLE_BROKERED_READ_ENV = "AGENTKIT_ORKA_ENABLE_BROKERED_READ"
_ENABLE_BROKERED_WRITE_ENV = "AGENTKIT_ORKA_ENABLE_BROKERED_WRITE"
_ENABLE_BROKERED_COORDINATION_ENV = "AGENTKIT_ORKA_ENABLE_BROKERED_COORDINATION"
_MAX_TOOL_SCHEMA_BYTES = 65536
_TERMINAL_TYPES = frozenset({"TurnCompleted", "TurnFailed", "TurnCancelled"})
# Orka's canonical Go client scans one SSE line with a 1 MiB token ceiling. The
# advertised payload ceiling deliberately leaves roughly half the line for the
# native frame envelope and JSON escaping. Runtime text counts encoded UTF-8
# bytes; brokered values count their compact JSON UTF-8 representation.
_MAX_OUTPUT_BYTES = 512 * 1024
_ORKA_CLIENT_MAX_SSE_TOKEN_BYTES = 1 << 20
_SSE_DATA_PREFIX = "data: "
_OUTPUT_LIMIT_CODE = "MaxOutputBytesExceeded"
_DEFAULT_MAX_TERMINAL_TURNS = 256
_DEFAULT_MAX_RUNTIME_SESSIONS = 64
_MAX_TERMINAL_TURNS_ENV = "AGENTKIT_ORKA_MAX_TERMINAL_TURNS"
_MAX_RUNTIME_SESSIONS_ENV = "AGENTKIT_ORKA_MAX_RUNTIME_SESSIONS"
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_JSON_VALUE_ABSENT = object()
# Go net/url.PathEscape leaves these reserved bytes unescaped in a path segment.
_ORKA_PATH_SEGMENT_SAFE = "$&+-.:=@_~"
# Mirrors Go strings.TrimSpace/unicode.IsSpace (which excludes U+001C-U+001F).
_ORKA_TRIM_SPACE_CHARS = (
    "\t\n\v\f\r \x85\xa0\u1680"
    "\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a"
    "\u2028\u2029\u202f\u205f\u3000"
)


@dataclass
class ActiveRuntime:
    context: Any
    session: Any
    env: dict[str, str]


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
    content: Any = _JSON_VALUE_ABSENT
    content_text: str = ""
    completed: Mapping[str, Any] | None = None
    failed: Mapping[str, Any] | None = None
    error: Mapping[str, Any] | None = None
    tool_name: str = ""
    tool_call_id: str = ""
    approval_id: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict)

    @property
    def terminal(self) -> bool:
        return self.type in _TERMINAL_TYPES

    def as_frame(self) -> dict[str, Any]:
        frame = {
            "version": ORKA_HARNESS_VERSION,
            "type": self.type,
            "runtimeSessionID": self.runtime_session_id,
            "turnID": self.turn_id,
            "correlationID": self.correlation_id,
            "seq": self.seq,
            "createdAt": self.created_at,
            "severity": self.severity,
            "summary": self.summary,
            "contentText": self.content_text,
            "toolName": self.tool_name,
            "toolCallID": self.tool_call_id,
            "approvalID": self.approval_id,
            "completed": dict(self.completed) if self.completed is not None else None,
            "failed": dict(self.failed) if self.failed is not None else None,
            "error": dict(self.error) if self.error is not None else None,
            "metadata": dict(self.metadata),
        }
        if self.content is not _JSON_VALUE_ABSENT:
            frame["content"] = self.content
        return frame


class _SSEFrameTooLargeError(ValueError):
    pass


class _SSEFrameEncodingError(ValueError):
    pass


def _frame_json(event: TurnEvent) -> str:
    return json.dumps(event.as_frame(), ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _compact_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _json_snapshot(value: Any) -> Any:
    return json.loads(_compact_json_bytes(value))


def _sse_data_line_bytes(event: TurnEvent) -> int:
    return len(_SSE_DATA_PREFIX.encode()) + len(_frame_json(event).encode())


def _ensure_sse_frame_fits(event: TurnEvent) -> None:
    try:
        line_bytes = _sse_data_line_bytes(event)
    except UnicodeEncodeError as exc:
        raise _SSEFrameEncodingError(f"{event.type} contains text that is not valid UTF-8") from exc
    if line_bytes >= _ORKA_CLIENT_MAX_SSE_TOKEN_BYTES:
        raise _SSEFrameTooLargeError(
            f"{event.type} SSE data line is {line_bytes} UTF-8 bytes; "
            f"Orka client limit is {_ORKA_CLIENT_MAX_SSE_TOKEN_BYTES - 1}"
        )


def _ensure_terminal_frame_fits(state: "TurnState") -> None:
    # Reserve ample room for deterministic failure detail even when identity
    # fields are unusually long.
    message = "x" * 1024
    _ensure_sse_frame_fits(
        TurnEvent(
            seq=9_223_372_036_854_775_807,
            type="TurnFailed",
            runtime_session_id=state.runtime_session_id,
            turn_id=state.turn_id,
            correlation_id=state.correlation_id,
            severity="error",
            summary="turn output rejected",
            failed={"reason": _OUTPUT_LIMIT_CODE, "message": message, "retryable": False},
            error={"code": _OUTPUT_LIMIT_CODE, "message": message, "retryable": False},
            metadata={},
        )
    )


@dataclass
class PendingBrokeredTool:
    call: BrokeredToolCall
    future: asyncio.Future[BrokeredToolResult]


class TurnState:
    """Buffered turn event state with replay support for SSE clients."""

    def __init__(
        self,
        *,
        runtime_session_id: str,
        turn_id: str,
        correlation_id: str,
        namespace: str = "",
        task_name: str = "",
        session_name: str = "",
        deadline: datetime | None = None,
        metadata: Mapping[str, str] | None = None,
        task: asyncio.Task[None] | None = None,
    ) -> None:
        self.runtime_session_id = runtime_session_id
        self.turn_id = turn_id
        self.correlation_id = correlation_id
        self.namespace = namespace
        self.task_name = task_name
        self.session_name = session_name
        self.deadline = deadline
        self.metadata = dict(metadata or {})
        self.task = task
        self.events: list[TurnEvent] = []
        self.condition = asyncio.Condition()
        self.terminal_event: TurnEvent | None = None
        self.pending_tools: dict[str, PendingBrokeredTool] = {}
        # Preserve idempotent /continue replay without retaining a second copy of
        # each potentially large accepted tool result.
        self.accepted_tool_result_digests: dict[str, str] = {}
        self.rejected_tool_result_digests: dict[str, tuple[str, str, str]] = {}

    def _append_locked(
        self,
        event_type: str,
        *,
        severity: str = "info",
        summary: str = "",
        content: Any = _JSON_VALUE_ABSENT,
        content_text: str = "",
        completed: Mapping[str, Any] | None = None,
        failed: Mapping[str, Any] | None = None,
        error: Mapping[str, Any] | None = None,
        tool_name: str = "",
        tool_call_id: str = "",
        approval_id: str = "",
        metadata: Mapping[str, str] | None = None,
    ) -> tuple[TurnEvent, bool]:
        if self.terminal_event is not None:
            if event_type in _TERMINAL_TYPES:
                return self.terminal_event, False
            raise AgentRunError("turn is already terminal", status=409, code="TurnTerminal")
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
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            approval_id=approval_id,
            metadata=self.metadata if metadata is None else metadata,
        )
        try:
            _ensure_sse_frame_fits(event)
        except (_SSEFrameTooLargeError, _SSEFrameEncodingError):
            if event_type != "TurnFailed":
                raise
            message = "terminal failure details could not be emitted safely"
            event = TurnEvent(
                seq=seq,
                type="TurnFailed",
                runtime_session_id=self.runtime_session_id,
                turn_id=self.turn_id,
                correlation_id=self.correlation_id,
                severity="error",
                summary="turn failed",
                failed={"reason": "TerminalFrameRejected", "message": message, "retryable": False},
                error={"code": "TerminalFrameRejected", "message": message, "retryable": False},
                metadata={},
            )
            _ensure_sse_frame_fits(event)
        self.events.append(event)
        if event.terminal:
            self.terminal_event = event
        self.condition.notify_all()
        return event, True

    async def append(
        self,
        event_type: str,
        *,
        severity: str = "info",
        summary: str = "",
        content: Any = _JSON_VALUE_ABSENT,
        content_text: str = "",
        completed: Mapping[str, Any] | None = None,
        failed: Mapping[str, Any] | None = None,
        error: Mapping[str, Any] | None = None,
        tool_name: str = "",
        tool_call_id: str = "",
        approval_id: str = "",
        metadata: Mapping[str, str] | None = None,
    ) -> tuple[TurnEvent, bool]:
        async with self.condition:
            return self._append_locked(
                event_type,
                severity=severity,
                summary=summary,
                content=content,
                content_text=content_text,
                completed=completed,
                failed=failed,
                error=error,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                approval_id=approval_id,
                metadata=metadata,
            )

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


def _truthy_env(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def _brokered_read_enabled(value: bool | None = None) -> bool:
    return bool(value) if value is not None else _truthy_env(_ENABLE_BROKERED_READ_ENV)


def _brokered_write_enabled(value: bool | None = None) -> bool:
    return bool(value) if value is not None else _truthy_env(_ENABLE_BROKERED_WRITE_ENV)


def _brokered_coordination_enabled(value: bool | None = None) -> bool:
    return bool(value) if value is not None else _truthy_env(_ENABLE_BROKERED_COORDINATION_ENV)


def _factory_supports_brokered_class(factory: RuntimeFactory, brokered_class: str) -> bool:
    if isinstance(factory, OfflineEchoRuntimeFactory):
        return True
    supports = getattr(factory, f"supports_brokered_{brokered_class}", None)
    if callable(supports):
        return bool(supports())
    return False


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
        metadata={},
    )
    if created:
        _record_terminal_turn(state.turn_id, terminal_order, turns, max_terminal_turns)


def _ensure_terminal_on_task_done(
    task: asyncio.Task[None],
    state: TurnState,
    terminal_order: list[str],
    turns: dict[str, TurnState],
    max_terminal_turns: int,
    background_tasks: set[asyncio.Task[None]],
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
    terminal_task = asyncio.create_task(
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
    background_tasks.add(terminal_task)
    terminal_task.add_done_callback(background_tasks.discard)


def _sse_frame(event: TurnEvent) -> str:
    data = _frame_json(event)
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
    value = data.get("turnID")
    if not isinstance(value, str):
        raise HTTPException(status_code=400, detail="turnID is required")
    trimmed = value.strip(_ORKA_TRIM_SPACE_CHARS)
    if not trimmed:
        raise HTTPException(status_code=400, detail="turnID is required")
    if value != trimmed or value in {".", ".."} or "/" in value or "\\" in value:
        raise HTTPException(
            status_code=400,
            detail="turnID must be a single safe path segment without surrounding whitespace",
        )
    return value


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


def _brokered_tools_from_input(input_value: Mapping[str, Any], *, allowed_brokered_classes: set[str]) -> list[BrokeredToolDefinition]:
    raw_tools = input_value.get("tools", [])
    if raw_tools in (None, ""):
        return []
    if not isinstance(raw_tools, list):
        raise HTTPException(status_code=400, detail="input.tools must be an array")
    tools: list[BrokeredToolDefinition] = []
    seen: set[str] = set()
    for idx, item in enumerate(raw_tools):
        if not isinstance(item, dict):
            raise HTTPException(status_code=400, detail=f"input.tools[{idx}] must be an object")
        name = _clean(item.get("name"))
        if name is None:
            raise HTTPException(status_code=400, detail=f"input.tools[{idx}].name is required")
        if name in seen:
            raise HTTPException(status_code=400, detail=f"input.tools[{idx}].name {name!r} is duplicated")
        seen.add(name)
        description = item.get("description", "")
        if not isinstance(description, str):
            raise HTTPException(status_code=400, detail=f"input.tools[{idx}].description must be a string")
        brokered_class = _clean(item.get("brokeredClass")) or BROKERED_CLASS_READ
        if brokered_class not in {BROKERED_CLASS_READ, BROKERED_CLASS_WRITE, BROKERED_CLASS_COORDINATION}:
            raise HTTPException(status_code=400, detail=f"unsupported brokered tool class {brokered_class!r}")
        if brokered_class not in allowed_brokered_classes:
            raise HTTPException(status_code=400, detail=f"brokered {brokered_class} tools are not enabled")
        parameters = item.get("parameters", {})
        if parameters is None:
            parameters = {}
        if not isinstance(parameters, dict):
            raise HTTPException(status_code=400, detail=f"input.tools[{idx}].parameters must be an object")
        try:
            encoded = json.dumps(parameters, separators=(",", ":"), sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"input.tools[{idx}].parameters must be JSON serializable") from exc
        if len(encoded.encode("utf-8")) > _MAX_TOOL_SCHEMA_BYTES:
            raise HTTPException(status_code=400, detail=f"input.tools[{idx}].parameters is too large")
        tools.append(
            BrokeredToolDefinition(
                name=name,
                description=description,
                brokered_class=brokered_class,
                parameters=json.loads(encoded),
            )
        )
    return tools


def _validate_auth_identity(data: Mapping[str, Any]) -> None:
    raw = data.get("authIdentity")
    if not isinstance(raw, dict):
        raise HTTPException(status_code=400, detail="authIdentity is required")
    if _clean(raw.get("subject")) is None and _clean(raw.get("username")) is None:
        raise HTTPException(status_code=400, detail="authIdentity.subject or authIdentity.username is required")


def _request_to_run_request(data: dict[str, Any], *, turn_id: str, spec: AgentSpec, allow_brokered: bool = False) -> RunRequest:
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
    allowed_modes = {None, "", TOOL_MODE_OBSERVED}
    if allow_brokered:
        allowed_modes.add(TOOL_MODE_BROKERED)
    if tool_mode not in allowed_modes:
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

    env = _env_from_input(input_value, allowed_names=_allowed_turn_env_names(spec))
    missing_required = []
    for entry in spec.env:
        if not entry.required:
            continue
        if entry.name in env:
            if env[entry.name] == "":
                missing_required.append(entry.name)
            continue
        if not os.environ.get(entry.name):
            missing_required.append(entry.name)
    if missing_required:
        names = ", ".join(missing_required)
        raise HTTPException(status_code=400, detail=f"required env var(s) missing from input.env or process env: {names}")

    return RunRequest(
        prompt=prompt,
        history=(),
        session_id=runtime_session_id,
        env=env,
        deadline=deadline,
        turn_id=turn_id,
        correlation_id=correlation_id,
        metadata=metadata,
    )


def _usage_payload(result: RunResult) -> dict[str, int]:
    usage = result.usage or {}
    return {key: int(usage.get(key, 0) or 0) for key in sorted(usage)}


def _utf8_bytes(value: str) -> int:
    return len(value.encode())


def _output_limit_message(output_kind: str, actual_bytes: int) -> str:
    return f"{output_kind} is {actual_bytes} UTF-8 bytes; maxOutputBytes is {_MAX_OUTPUT_BYTES}"


def _brokered_result_digest(result: BrokeredToolResult) -> str:
    value: dict[str, Any] = {
        "approved": result.approved,
        "outputPresent": result.output_present,
    }
    if result.output_present:
        value["output"] = result.output
    if result.error is not None:
        value["error"] = dict(result.error)
    return hashlib.sha256(_compact_json_bytes(value)).hexdigest()


async def _append_output_failure(
    state: TurnState,
    terminal_order: list[str],
    turns: dict[str, TurnState],
    max_terminal_turns: int,
    *,
    message: str,
    code: str = _OUTPUT_LIMIT_CODE,
) -> None:
    await _append_terminal_if_missing(
        state,
        terminal_order,
        turns,
        max_terminal_turns,
        "TurnFailed",
        summary="turn output rejected",
        failed={"reason": code, "message": message, "retryable": False},
        error={"code": code, "message": message, "retryable": False},
    )


def _append_output_failure_locked(state: TurnState, message: str, code: str) -> bool:
    _, created = state._append_locked(
        "TurnFailed",
        severity="error",
        summary="turn output rejected",
        failed={"reason": code, "message": message, "retryable": False},
        error={"code": code, "message": message, "retryable": False},
        metadata={},
    )
    return created


def health_response(spec: AgentSpec) -> dict[str, Any]:
    return {
        "version": ORKA_HARNESS_VERSION,
        "status": "ok",
        "ready": True,
        "checkedAt": _now_iso(),
        "metadata": {"agentName": spec.metadata.name},
    }


def capabilities_response(spec: AgentSpec, *, brokered_classes: set[str] | None = None) -> dict[str, Any]:
    response = {
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
        "maxOutputBytes": _MAX_OUTPUT_BYTES,
        "metadata": {
            "agentName": spec.metadata.name,
            "model": spec.model.name,
            "agentkitProvider": spec.model.provider,
        },
    }
    brokered_classes = brokered_classes or set()
    if brokered_classes:
        response["toolExecutionModes"] = [TOOL_MODE_OBSERVED, TOOL_MODE_BROKERED]
        response["brokeredToolClasses"] = [
            klass
            for klass in (BROKERED_CLASS_READ, BROKERED_CLASS_WRITE, BROKERED_CLASS_COORDINATION)
            if klass in brokered_classes
        ]
        response["supportsContinuation"] = True
    return response


def start_turn_response(turn: TurnState) -> dict[str, Any]:
    return {
        "version": ORKA_HARNESS_VERSION,
        "accepted": True,
        "runtimeSessionID": turn.runtime_session_id,
        "turnID": turn.turn_id,
        "correlationID": turn.correlation_id,
        "eventStreamPath": f"/v1/turns/{quote(turn.turn_id, safe=_ORKA_PATH_SEGMENT_SAFE)}/events",
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


def continue_turn_response(turn: TurnState) -> dict[str, Any]:
    return {
        "version": ORKA_HARNESS_VERSION,
        "accepted": True,
        "runtimeSessionID": turn.runtime_session_id,
        "turnID": turn.turn_id,
        "correlationID": turn.correlation_id,
        "message": "continue accepted",
    }


class OrkaToolBroker:
    def __init__(self, state: TurnState, tools: list[BrokeredToolDefinition]) -> None:
        self.state = state
        self.tools = {tool.name: tool for tool in tools}

    async def request_tool(self, call: BrokeredToolCall) -> BrokeredToolResult:
        if not call.tool_call_id.strip():
            raise AgentRunError("brokered tool call id is required", status=400, code="InvalidToolCallID")
        if call.tool_call_id != call.tool_call_id.strip():
            raise AgentRunError("brokered tool call id must not contain leading or trailing whitespace", status=400, code="InvalidToolCallID")
        tool = self.tools.get(call.name)
        if tool is None:
            raise AgentRunError(f"unknown brokered tool {call.name!r}", status=400, code="UnknownBrokeredTool")
        if tool.brokered_class != call.brokered_class:
            raise AgentRunError(f"brokered class mismatch for tool {call.name!r}", status=400, code="BrokeredClassMismatch")
        if not isinstance(call.arguments, Mapping):
            raise AgentRunError("brokered tool arguments must be an object", status=400, code="InvalidToolArguments")
        try:
            encoded_arguments = json.dumps(dict(call.arguments), separators=(",", ":"), sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise AgentRunError("brokered tool arguments must be JSON serializable", status=400, code="InvalidToolArguments") from exc
        arguments = json.loads(encoded_arguments)
        async with self.state.condition:
            if self.state.terminal_event is not None:
                raise AgentRunError("turn is already terminal", status=409, code="TurnTerminal")
            if call.tool_call_id in self.state.pending_tools:
                raise AgentRunError(f"duplicate brokered tool call id {call.tool_call_id!r}", status=400, code="DuplicateToolCallID")
            future: asyncio.Future[BrokeredToolResult] = asyncio.get_running_loop().create_future()
            pending = PendingBrokeredTool(call=call, future=future)
            self.state.pending_tools[call.tool_call_id] = pending
        try:
            await self.state.append(
                "ToolCallRequested",
                summary="brokered tool requested",
                content=arguments,
                tool_name=call.name,
                tool_call_id=call.tool_call_id,
                metadata={},
            )
            if self.state.deadline is None:
                result = await future
            else:
                seconds = (self.state.deadline - datetime.now(UTC)).total_seconds()
                if seconds <= 0:
                    raise TimeoutError("turn deadline exceeded while waiting for brokered tool result")
                async with asyncio.timeout(seconds):
                    result = await future
            await self.state.append(
                "ToolResultReceived",
                summary="tool result received",
                content=_json_snapshot(result.output) if result.output_present else _JSON_VALUE_ABSENT,
                error=_json_snapshot(dict(result.error)) if result.error is not None else None,
                tool_name=call.name,
                tool_call_id=call.tool_call_id,
                metadata={},
            )
            return result
        finally:
            async with self.state.condition:
                if self.state.pending_tools.get(call.tool_call_id) is pending:
                    self.state.pending_tools.pop(call.tool_call_id, None)
                if not future.done():
                    future.cancel()
                self.state.condition.notify_all()


async def _run_turn(
    get_runtime: Callable[[RunRequest], Awaitable[Any]],
    turns: dict[str, TurnState],
    terminal_order: list[str],
    state: TurnState,
    run_request: RunRequest,
    *,
    max_terminal_turns: int,
    brokered_tools: list[BrokeredToolDefinition] | None = None,
) -> None:
    async def _run_with_runtime() -> RunResult:
        runtime = await get_runtime(run_request)
        with _scoped_process_env(run_request.env):
            if brokered_tools is not None:
                if not isinstance(runtime, BrokeredRuntimeSession):
                    raise AgentRunError("runtime does not support brokered Orka tools", status=400, code="BrokeredUnsupported")
                return await runtime.run_brokered(run_request, brokered_tools, OrkaToolBroker(state, brokered_tools))
            return await runtime.run(run_request)

    try:
        if run_request.deadline is None:
            result = await _run_with_runtime()
        else:
            seconds = (run_request.deadline - datetime.now(UTC)).total_seconds()
            if seconds <= 0:
                raise TimeoutError("turn deadline has already expired")
            async with asyncio.timeout(seconds):
                result = await _run_with_runtime()
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

    try:
        output_bytes = _utf8_bytes(result.text)
    except UnicodeEncodeError:
        await _append_output_failure(
            state,
            terminal_order,
            turns,
            max_terminal_turns,
            message="runtime output is not valid UTF-8",
            code="InvalidOutputEncoding",
        )
        return
    if output_bytes > _MAX_OUTPUT_BYTES:
        await _append_output_failure(
            state,
            terminal_order,
            turns,
            max_terminal_turns,
            message=_output_limit_message("runtime output", output_bytes),
        )
        return
    try:
        if result.text:
            await state.append(
                "RuntimeOutput",
                summary="runtime output",
                # contentText is the native text channel; keep only structured
                # usage in content so one SSE frame does not duplicate the text.
                content={"usage": _usage_payload(result)},
                content_text=result.text,
                metadata={},
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
    except _SSEFrameTooLargeError as exc:
        await _append_output_failure(
            state,
            terminal_order,
            turns,
            max_terminal_turns,
            message=str(exc),
            code="HarnessFrameTooLarge",
        )


def create_orka_app(
    spec: AgentSpec,
    factory: RuntimeFactory,
    auth_token: str | None = None,
    *,
    max_terminal_turns: int | None = None,
    max_runtime_sessions: int | None = None,
    enable_brokered_read: bool | None = None,
    enable_brokered_write: bool | None = None,
    enable_brokered_coordination: bool | None = None,
) -> FastAPI:
    """Create an Orka harness app for one AgentKit runtime."""
    if not auth_token:
        raise ValueError("Orka mode requires a bearer auth token")
    retention_limit = _max_terminal_turns(max_terminal_turns)
    runtime_session_limit = _max_runtime_sessions(max_runtime_sessions)
    brokered_classes: set[str] = set()
    if _brokered_read_enabled(enable_brokered_read):
        brokered_classes.add(BROKERED_CLASS_READ)
    if _brokered_write_enabled(enable_brokered_write):
        brokered_classes.add(BROKERED_CLASS_WRITE)
    if _brokered_coordination_enabled(enable_brokered_coordination):
        brokered_classes.add(BROKERED_CLASS_COORDINATION)
    for brokered_class in brokered_classes:
        if not _factory_supports_brokered_class(factory, brokered_class):
            raise ValueError(f"Orka brokered {brokered_class} requires a runtime factory that supports brokered tools")
    turns: dict[str, TurnState] = {}
    terminal_order: list[str] = []
    active_runtimes: dict[str, ActiveRuntime] = {}
    runtime_order: list[str] = []
    background_tasks: set[asyncio.Task[None]] = set()

    async def get_runtime(run_request: RunRequest) -> Any:
        runtime_session_id = run_request.session_id or ""
        active = active_runtimes.get(runtime_session_id)
        if active is not None:
            if active.env == dict(run_request.env):
                if runtime_session_id in runtime_order:
                    runtime_order.remove(runtime_session_id)
                runtime_order.append(runtime_session_id)
                return active.session
            active_runtimes.pop(runtime_session_id, None)
            if runtime_session_id in runtime_order:
                runtime_order.remove(runtime_session_id)
            await asyncio.shield(active.context.__aexit__(None, None, None))
        # Runtime factories read the process environment today. Keep this scoped
        # section on the event loop thread so cancellation cannot leave a worker
        # thread running with turn credentials in process-global os.environ.
        with _scoped_process_env(run_request.env):
            context = factory.build_runtime(spec)
            session = await context.__aenter__()
        active_runtimes[runtime_session_id] = ActiveRuntime(context=context, session=session, env=dict(run_request.env))
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
            active_turn_tasks = [state.task for state in turns.values() if state.task is not None and not state.task.done()]
            for task in active_turn_tasks:
                task.cancel()
            if active_turn_tasks:
                await asyncio.gather(*active_turn_tasks, return_exceptions=True)
            while background_tasks:
                terminal_tasks = list(background_tasks)
                await asyncio.gather(*terminal_tasks, return_exceptions=True)
                background_tasks.difference_update(terminal_tasks)
            background_tasks.clear()
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
        return capabilities_response(spec, brokered_classes=brokered_classes)

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
            raise HTTPException(status_code=409, detail="turn already exists")
        if any(state.terminal_event is None for state in turns.values()):
            raise HTTPException(status_code=429, detail="maxConcurrentTurns limit reached")

        tool_mode = _clean(data.get("toolExecutionMode")) or TOOL_MODE_OBSERVED
        run_request = _request_to_run_request(data, turn_id=turn_id, spec=spec, allow_brokered=bool(brokered_classes))
        brokered_tools: list[BrokeredToolDefinition] | None = None
        if tool_mode == TOOL_MODE_BROKERED:
            input_value = data.get("input")
            if not isinstance(input_value, dict):
                raise HTTPException(status_code=400, detail="input must be an object")
            brokered_tools = _brokered_tools_from_input(input_value, allowed_brokered_classes=brokered_classes)
        runtime_session_id = run_request.session_id or ""
        correlation_id = run_request.correlation_id or ""
        state = TurnState(
            runtime_session_id=runtime_session_id,
            turn_id=turn_id,
            correlation_id=correlation_id,
            namespace=_required_string(data, "namespace"),
            task_name=_required_string(data, "taskName"),
            session_name=_required_string(data, "sessionName"),
            deadline=run_request.deadline,
            metadata=run_request.metadata,
        )
        try:
            await state.append("TurnStarted", summary="turn started")
            _ensure_terminal_frame_fits(state)
        except _SSEFrameTooLargeError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except _SSEFrameEncodingError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        turns[turn_id] = state
        state.task = asyncio.create_task(
            _run_turn(
                get_runtime,
                turns,
                terminal_order,
                state,
                run_request,
                max_terminal_turns=retention_limit,
                brokered_tools=brokered_tools,
            )
        )
        state.task.add_done_callback(
            lambda task, state=state: _ensure_terminal_on_task_done(
                task,
                state,
                terminal_order,
                turns,
                retention_limit,
                background_tasks,
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

    @app.post("/v1/turns/{turn_id}/continue", dependencies=[auth], status_code=202)
    async def continue_turn(turn_id: str, request: Request):
        if not brokered_classes:
            raise HTTPException(status_code=404, detail="brokered continuation is not enabled")
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
            raise HTTPException(status_code=400, detail="continue turnID must match route turnID")
        runtime_session_id = _required_string(data, "runtimeSessionID")
        correlation_id = _required_string(data, "correlationID")
        namespace = _required_string(data, "namespace")
        task_name = _required_string(data, "taskName")
        session_name = _required_string(data, "sessionName")
        state = turns.get(turn_id)
        if state is None:
            raise HTTPException(status_code=404, detail="turn not found")
        if runtime_session_id != state.runtime_session_id:
            raise HTTPException(status_code=400, detail="continue runtimeSessionID must match turn runtimeSessionID")
        if correlation_id != state.correlation_id:
            raise HTTPException(status_code=400, detail="continue correlationID must match turn correlationID")
        if namespace != state.namespace or task_name != state.task_name or session_name != state.session_name:
            raise HTTPException(status_code=400, detail="continue namespace/taskName/sessionName must match turn")
        raw_results = data.get("toolResults")
        if not isinstance(raw_results, list) or not raw_results:
            raise HTTPException(status_code=400, detail="toolResults must be a non-empty array")
        results: list[tuple[BrokeredToolResult, int, str]] = []
        for idx, raw_result in enumerate(raw_results):
            if not isinstance(raw_result, dict):
                raise HTTPException(status_code=400, detail=f"toolResults[{idx}] must be an object")
            if raw_result.get("version") != ORKA_HARNESS_VERSION:
                raise HTTPException(status_code=400, detail=f"toolResults[{idx}].version must be {ORKA_HARNESS_VERSION!r}")
            result_runtime_session_id = _required_string(raw_result, "runtimeSessionID")
            result_turn_id = _required_string(raw_result, "turnID")
            tool_call_id = _required_string(raw_result, "toolCallID")
            idempotency_key = _required_string(raw_result, "idempotencyKey")
            expected_idempotency_key = f"{runtime_session_id}:{turn_id}:{tool_call_id}"
            if result_runtime_session_id != runtime_session_id or result_turn_id != turn_id:
                raise HTTPException(status_code=400, detail=f"toolResults[{idx}] identity must match continue request")
            if idempotency_key != expected_idempotency_key:
                raise HTTPException(status_code=400, detail=f"toolResults[{idx}].idempotencyKey does not match tool call")
            approved = raw_result.get("approved", False)
            if not isinstance(approved, bool):
                raise HTTPException(status_code=400, detail=f"toolResults[{idx}].approved must be a boolean")
            output_present = "output" in raw_result
            output_value = raw_result.get("output")
            error_value = raw_result.get("error")
            if not approved and output_present:
                raise HTTPException(status_code=400, detail=f"toolResults[{idx}].output is not allowed when approved is false")
            if not output_present and error_value is None:
                if approved:
                    raise HTTPException(status_code=400, detail=f"toolResults[{idx}] output or error is required")
                error_value = {"code": "ToolCallDenied", "message": "tool call was not approved", "retryable": False}
            if error_value is not None and not isinstance(error_value, dict):
                raise HTTPException(status_code=400, detail=f"toolResults[{idx}].error must be an object")
            try:
                output_bytes = len(_compact_json_bytes(output_value)) if output_present else 0
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=400, detail=f"toolResults[{idx}].output must be valid JSON") from exc
            result_kwargs: dict[str, Any] = {
                "tool_call_id": tool_call_id,
                "approved": approved,
                "error": dict(error_value) if error_value is not None else None,
            }
            if output_present:
                result_kwargs["output"] = output_value
            result = BrokeredToolResult(**result_kwargs)
            try:
                digest = _brokered_result_digest(result)
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=400, detail=f"toolResults[{idx}] must be valid JSON") from exc
            results.append((result, output_bytes, digest))

        seen_results: set[str] = set()
        for result, _, _ in results:
            if result.tool_call_id in seen_results:
                raise HTTPException(status_code=400, detail=f"duplicate toolCallID {result.tool_call_id!r}")
            seen_results.add(result.tool_call_id)

        rejection_message: str | None = None
        rejection_code = _OUTPUT_LIMIT_CODE
        cancel_runtime_task = False
        async with state.condition:
            if state.terminal_event is not None:
                rejected_outcomes: set[tuple[str, str]] = set()
                for result, _, digest in results:
                    accepted_digest = state.accepted_tool_result_digests.get(result.tool_call_id)
                    if accepted_digest is not None:
                        if accepted_digest != digest:
                            raise HTTPException(
                                status_code=409,
                                detail=f"conflicting tool result for toolCallID {result.tool_call_id!r}",
                            )
                        continue
                    rejected = state.rejected_tool_result_digests.get(result.tool_call_id)
                    if rejected is None:
                        raise HTTPException(status_code=409, detail="turn is already terminal")
                    rejected_digest, message, code = rejected
                    if rejected_digest != digest:
                        raise HTTPException(
                            status_code=409,
                            detail=f"conflicting tool result for toolCallID {result.tool_call_id!r}",
                        )
                    rejected_outcomes.add((message, code))
                if not rejected_outcomes:
                    return continue_turn_response(state)
                if len(rejected_outcomes) != 1:
                    raise HTTPException(status_code=409, detail="tool results do not match one rejected continuation")
                rejection_message, rejection_code = rejected_outcomes.pop()

            pending_results: list[tuple[PendingBrokeredTool, BrokeredToolResult, str]] = []
            if state.terminal_event is None:
                fresh_results: list[tuple[PendingBrokeredTool, BrokeredToolResult, int, str]] = []
                # Validate every identity and accepted-result digest before any
                # oversized member can terminalize the whole continuation batch.
                for result, output_bytes, digest in results:
                    accepted_digest = state.accepted_tool_result_digests.get(result.tool_call_id)
                    if accepted_digest is not None:
                        if accepted_digest != digest:
                            raise HTTPException(
                                status_code=409,
                                detail=f"conflicting tool result for toolCallID {result.tool_call_id!r}",
                            )
                        continue
                    pending = state.pending_tools.get(result.tool_call_id)
                    if pending is None:
                        raise HTTPException(status_code=400, detail=f"unknown toolCallID {result.tool_call_id!r}")
                    fresh_results.append((pending, result, output_bytes, digest))

                rejection_candidates: list[tuple[str, str, str]] = []
                for pending, result, output_bytes, digest in fresh_results:
                    if output_bytes > _MAX_OUTPUT_BYTES:
                        rejection_candidates.append(
                            (
                                result.tool_call_id,
                                _output_limit_message("brokered tool output", output_bytes),
                                _OUTPUT_LIMIT_CODE,
                            )
                        )
                        continue
                    candidate = TurnEvent(
                        # Preflight with the widest native int64 sequence so an
                        # in-flight event cannot add a digit after HTTP 202.
                        seq=9_223_372_036_854_775_807,
                        type="ToolResultReceived",
                        runtime_session_id=state.runtime_session_id,
                        turn_id=state.turn_id,
                        correlation_id=state.correlation_id,
                        summary="tool result received",
                        content=result.output if result.output_present else _JSON_VALUE_ABSENT,
                        error=dict(result.error) if result.error is not None else None,
                        tool_name=pending.call.name,
                        tool_call_id=result.tool_call_id,
                        metadata={},
                    )
                    try:
                        _ensure_sse_frame_fits(candidate)
                    except _SSEFrameTooLargeError as exc:
                        rejection_candidates.append((result.tool_call_id, str(exc), "HarnessFrameTooLarge"))
                        continue
                    pending_results.append((pending, result, digest))
                if rejection_candidates:
                    _, rejection_message, rejection_code = min(rejection_candidates)

            if rejection_message is None and state.terminal_event is None:
                for pending, result, digest in pending_results:
                    state.accepted_tool_result_digests[result.tool_call_id] = digest
                    if not pending.future.done():
                        pending.future.set_result(result)
                state.condition.notify_all()
            elif state.terminal_event is None:
                for result, _, digest in results:
                    if result.tool_call_id not in state.accepted_tool_result_digests:
                        state.rejected_tool_result_digests[result.tool_call_id] = (
                            digest,
                            rejection_message,
                            rejection_code,
                        )
                created = _append_output_failure_locked(state, rejection_message, rejection_code)
                if created:
                    _record_terminal_turn(state.turn_id, terminal_order, turns, retention_limit)
                pending = list(state.pending_tools.values())
                state.pending_tools.clear()
                for item in pending:
                    if not item.future.done():
                        item.future.cancel()
                state.condition.notify_all()
                cancel_runtime_task = state.task is not None and not state.task.done()

        if rejection_message is not None:
            if cancel_runtime_task and state.task is not None and state.task is not asyncio.current_task():
                state.task.cancel()
            raise HTTPException(status_code=413, detail=rejection_message)
        return continue_turn_response(state)


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
        namespace = _required_string(data, "namespace")
        task_name = _required_string(data, "taskName")
        session_name = _required_string(data, "sessionName")

        state = turns.get(turn_id)
        if state is None:
            raise HTTPException(status_code=404, detail="turn not found")
        if runtime_session_id != state.runtime_session_id:
            raise HTTPException(status_code=400, detail="cancel runtimeSessionID must match turn runtimeSessionID")
        if correlation_id != state.correlation_id:
            raise HTTPException(status_code=400, detail="cancel correlationID must match turn correlationID")
        if namespace != state.namespace or task_name != state.task_name or session_name != state.session_name:
            raise HTTPException(status_code=400, detail="cancel namespace/taskName/sessionName must match turn")
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
