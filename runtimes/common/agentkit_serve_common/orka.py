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
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Mapping

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from .config import AgentSpec
from .conversation import FORWARDED_ROLES, ConversationTurn, RunRequest, text_of
from .runtime import AgentRunError, RunResult, RuntimeFactory
from .server import make_auth_dependency

ORKA_HARNESS_VERSION = "orka.harness.v1"
_TERMINAL_TYPES = frozenset({"TurnCompleted", "TurnFailed", "TurnCancelled"})
_DEFAULT_MAX_TERMINAL_TURNS = 256
_MAX_TERMINAL_TURNS_ENV = "AGENTKIT_ORKA_MAX_TERMINAL_TURNS"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class TurnEvent:
    seq: int
    type: str
    turn_id: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_now_iso)

    @property
    def terminal(self) -> bool:
        return self.type in _TERMINAL_TYPES

    def as_frame(self) -> dict[str, Any]:
        return {
            "version": ORKA_HARNESS_VERSION,
            "turnID": self.turn_id,
            "seq": self.seq,
            "type": self.type,
            "timestamp": self.timestamp,
            "payload": dict(self.payload),
        }


class TurnState:
    """Buffered turn event state with replay support for SSE clients."""

    def __init__(self, turn_id: str, task: asyncio.Task[None] | None = None) -> None:
        self.turn_id = turn_id
        self.task = task
        self.events: list[TurnEvent] = []
        self.condition = asyncio.Condition()
        self.terminal_event: TurnEvent | None = None

    async def append(self, event_type: str, payload: Mapping[str, Any] | None = None) -> TurnEvent:
        async with self.condition:
            if event_type in _TERMINAL_TYPES and self.terminal_event is not None:
                return self.terminal_event
            event = TurnEvent(
                seq=len(self.events) + 1,
                type=event_type,
                turn_id=self.turn_id,
                payload=payload or {},
            )
            self.events.append(event)
            if event.terminal:
                self.terminal_event = event
            self.condition.notify_all()
            return event

    async def events_after(self, seq: int) -> list[TurnEvent]:
        async with self.condition:
            while True:
                events = [event for event in self.events if event.seq > seq]
                if events or self.terminal_event is not None:
                    return events
                await self.condition.wait()


def _max_terminal_turns(value: int | None = None) -> int:
    if value is not None:
        if value < 1:
            raise ValueError("max_terminal_turns must be at least 1")
        return value
    raw = os.environ.get(_MAX_TERMINAL_TURNS_ENV)
    if not raw:
        return _DEFAULT_MAX_TERMINAL_TURNS
    try:
        parsed = int(raw)
    except ValueError:
        return _DEFAULT_MAX_TERMINAL_TURNS
    return parsed if parsed > 0 else _DEFAULT_MAX_TERMINAL_TURNS


def _evict_terminal_turns(turns: dict[str, TurnState], max_terminal_turns: int) -> None:
    terminal_ids = [turn_id for turn_id, state in turns.items() if state.terminal_event is not None]
    overflow = len(terminal_ids) - max_terminal_turns
    if overflow <= 0:
        return
    for turn_id in terminal_ids[:overflow]:
        turns.pop(turn_id, None)


def _sse_frame(event: TurnEvent) -> str:
    data = json.dumps(event.as_frame(), separators=(",", ":"), sort_keys=True)
    return f"id: {event.seq}\nevent: {event.type}\ndata: {data}\n\n"


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


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


def _parse_deadline(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
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


def _prompt_from_payload(data: dict[str, Any]) -> str:
    for key in ("prompt", "input", "message"):
        if key not in data:
            continue
        value = data[key]
        if isinstance(value, str):
            return value
        if isinstance(value, dict) and isinstance(value.get("prompt"), str):
            return value["prompt"]
        return json.dumps(value, separators=(",", ":"), sort_keys=True)
    raise HTTPException(status_code=400, detail="turn request must include prompt, input, or message")


def _history_from_payload(data: dict[str, Any]) -> tuple[ConversationTurn, ...]:
    raw = data.get("history", data.get("messages", []))
    if raw in (None, ""):
        return ()
    if not isinstance(raw, list):
        raise HTTPException(status_code=400, detail="history/messages must be an array")
    history: list[ConversationTurn] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise HTTPException(status_code=400, detail=f"history[{idx}] must be an object")
        role = str(item.get("role") or "")
        if role not in FORWARDED_ROLES:
            continue
        if "text" in item:
            text = str(item.get("text") or "")
        else:
            text = text_of(item.get("content"))
        if text:
            history.append(ConversationTurn(role=role, text=text))
    return tuple(history)


def _request_to_run_request(data: dict[str, Any], *, turn_id: str) -> RunRequest:
    version = data.get("version") or data.get("contractVersion")
    if version != ORKA_HARNESS_VERSION:
        raise HTTPException(status_code=400, detail=f"version must be {ORKA_HARNESS_VERSION!r}")

    session_id = _clean(data.get("sessionID", data.get("sessionId", data.get("session_id"))))
    correlation_id = _clean(
        data.get("correlationID", data.get("correlationId", data.get("correlation_id")))
    )
    return RunRequest(
        prompt=_prompt_from_payload(data),
        history=_history_from_payload(data),
        session_id=session_id,
        env=_mapping_of_strings(data.get("env"), field_name="env"),
        deadline=_parse_deadline(data.get("deadline")),
        turn_id=turn_id,
        correlation_id=correlation_id,
        metadata=_mapping_of_strings(data.get("metadata"), field_name="metadata"),
    )


def _usage_payload(result: RunResult) -> dict[str, int]:
    usage = result.usage or {}
    return {key: int(usage.get(key, 0) or 0) for key in sorted(usage)}


async def _run_turn(
    runtime: Any,
    turns: dict[str, TurnState],
    state: TurnState,
    run_request: RunRequest,
    *,
    max_terminal_turns: int,
) -> None:
    terminal_type = "TurnCompleted"
    terminal_payload: Mapping[str, Any]
    try:
        if run_request.deadline is None:
            result = await runtime.run(run_request)
        else:
            seconds = (run_request.deadline - datetime.now(UTC)).total_seconds()
            if seconds <= 0:
                raise TimeoutError("turn deadline has already expired")
            async with asyncio.timeout(seconds):
                result = await runtime.run(run_request)
        terminal_payload = {
            "result": {"text": result.text, "usage": _usage_payload(result)},
            "finishReason": "stop",
        }
    except asyncio.CancelledError:
        terminal_type = "TurnCancelled"
        terminal_payload = {"reason": "cancelled"}
    except TimeoutError as exc:
        terminal_type = "TurnFailed"
        terminal_payload = {"code": "DeadlineExceeded", "message": str(exc) or "turn deadline exceeded"}
    except AgentRunError as exc:
        terminal_type = "TurnFailed"
        terminal_payload = {"code": exc.code or exc.__class__.__name__, "message": str(exc), "status": exc.status}
    except Exception as exc:  # noqa: BLE001 - protocol envelope must be deterministic.
        terminal_type = "TurnFailed"
        terminal_payload = {"code": exc.__class__.__name__, "message": str(exc), "status": 502}

    await state.append(terminal_type, terminal_payload)
    _evict_terminal_turns(turns, max_terminal_turns)


def create_orka_app(
    spec: AgentSpec,
    factory: RuntimeFactory,
    auth_token: str | None = None,
    *,
    max_terminal_turns: int | None = None,
) -> FastAPI:
    """Create an observed-mode Orka harness app for one AgentKit runtime."""
    if not auth_token:
        raise ValueError("Orka mode requires a bearer auth token")
    retention_limit = _max_terminal_turns(max_terminal_turns)
    runtime = factory.build_runtime(spec)
    turns: dict[str, TurnState] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with runtime:
            app.state.runtime = runtime
            app.state.turns = turns
            yield
            for state in turns.values():
                if state.task is not None and not state.task.done():
                    state.task.cancel()

    app = FastAPI(title="agentkit-serve-orka", lifespan=lifespan)
    auth = Depends(make_auth_dependency(auth_token))

    @app.get("/v1/health")
    async def health() -> dict[str, Any]:
        return {"version": ORKA_HARNESS_VERSION, "status": "ok", "ready": True}

    @app.get("/v1/capabilities")
    async def capabilities() -> dict[str, Any]:
        return {
            "version": ORKA_HARNESS_VERSION,
            "runtime": {
                "name": "agentkit-serve",
                "version": "0.0.0",
                "agentName": spec.metadata.name,
            },
            "provider": {"kind": spec.model.provider, "model": spec.model.name},
            "toolExecutionModes": ["observed"],
            "supportsCancel": True,
            "supportsRuntimeSessions": True,
            "supportsReplay": True,
        }

    @app.post("/v1/turns", dependencies=[auth], status_code=202)
    async def create_turn(request: Request):
        try:
            data = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="Request body must be JSON") from exc
        if not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object")

        turn_id = _clean(data.get("turnID", data.get("turnId", data.get("turn_id")))) or f"turn_{uuid.uuid4().hex}"
        if turn_id in turns:
            raise HTTPException(status_code=409, detail=f"turn {turn_id!r} already exists")

        run_request = _request_to_run_request(data, turn_id=turn_id)
        state = TurnState(turn_id)
        turns[turn_id] = state
        await state.append(
            "TurnStarted",
            {
                "sessionID": run_request.session_id,
                "correlationID": run_request.correlation_id,
                "metadata": dict(run_request.metadata),
            },
        )
        state.task = asyncio.create_task(
            _run_turn(
                request.app.state.runtime,
                turns,
                state,
                run_request,
                max_terminal_turns=retention_limit,
            )
        )
        return {
            "version": ORKA_HARNESS_VERSION,
            "turnID": turn_id,
            "status": "accepted",
            "events": f"/v1/turns/{turn_id}/events",
        }

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
    async def cancel_turn(turn_id: str):
        state = turns.get(turn_id)
        if state is None:
            raise HTTPException(status_code=404, detail="turn not found")
        if state.terminal_event is None and state.task is not None and not state.task.done():
            state.task.cancel()
        elif state.terminal_event is None:
            await state.append("TurnCancelled", {"reason": "cancelled"})
            _evict_terminal_turns(turns, retention_limit)
        return {"version": ORKA_HARNESS_VERSION, "turnID": turn_id, "status": "accepted"}

    @app.get("/v1/turns/{turn_id}/output", dependencies=[auth])
    async def turn_output(turn_id: str, ref: str):  # noqa: ARG001 - reserved optional endpoint.
        if turn_id not in turns:
            raise HTTPException(status_code=404, detail="turn not found")
        raise HTTPException(status_code=404, detail="output ref not found")

    return app
