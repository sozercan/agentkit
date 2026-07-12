"""Foundry Hosted Agent protocol adapters over the AgentKit RuntimeSession seam.

The adapter intentionally stays provider-light: it exposes the container HTTP
contract expected by Foundry Hosted Agents (``/readiness``, ``/invocations`` and
``/responses``) while reusing the same ``RuntimeFactory`` / ``RunRequest`` seam
as the native OpenAI facade.

When ``agent.yaml`` contains static ``brokeredTools`` declarations, the hosted
``/responses`` route enters a deterministic brokered function-call loop. The
container emits Responses ``function_call`` output items but never executes the
Orka-governed tool locally. A later continuation must provide a matching
``function_call_output`` for the same ``previous_response_id`` and pending
``call_id``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import time
import uuid
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Mapping

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, Response

from .brokered import brokered_tool_definitions
from .config import AgentSpec, _unsafe_brokered_key, _unsafe_brokered_text
from .foundry_model_loop import BrokeredChatModelLoop, ModelLoopFinal, ModelLoopToolRequest
from .conversation import FORWARDED_ROLES, ConversationTurn, RunRequest
from .runtime import AgentRunError, BrokeredToolDefinition, RunResult, RuntimeFactory
from .server import make_auth_dependency

logger = logging.getLogger(__name__)

try:  # Prefer the official hosted Responses SDK ID format/state-compatible prefix.
    from azure.ai.agentserver.responses._id_generator import IdGenerator as _AzureResponsesIdGenerator
except Exception:  # pragma: no cover - the SDK is optional outside conformance installs.
    _AzureResponsesIdGenerator = None

_DEFAULT_STATE_TTL_SECONDS = 15 * 60
_DEFAULT_MAX_PENDING_RESPONSES = 128
_DEFAULT_MAX_ARGUMENT_BYTES = 8192
_DEFAULT_MAX_OUTPUT_BYTES = 64 * 1024
_MAX_SYNTHETIC_ARRAY_ITEMS = 32
_MAX_SYNTHETIC_STRING_LENGTH = 4096
_STATE_TTL_ENV = "AGENTKIT_FOUNDRY_RESPONSE_STATE_TTL_SECONDS"
_MAX_PENDING_ENV = "AGENTKIT_FOUNDRY_RESPONSE_STATE_MAX_PENDING"
_MAX_ARGUMENT_BYTES_ENV = "AGENTKIT_FOUNDRY_BROKERED_MAX_ARGUMENT_BYTES"
_MAX_OUTPUT_BYTES_ENV = "AGENTKIT_FOUNDRY_BROKERED_MAX_OUTPUT_BYTES"
_CONTINUATION_PROOF_ENV = "AGENTKIT_FOUNDRY_BROKERED_CONTINUATION_PROOF"
_CONTINUATION_PROOF_HEADER = "x-agentkit-brokered-continuation-proof"
_MODEL_LOOP_ENV = "AGENTKIT_FOUNDRY_BROKERED_MODEL_LOOP"
_STATE_FILE_ENV = "AGENTKIT_FOUNDRY_RESPONSE_STATE_FILE"


def _new_response_id(previous_response_id: str | None = None) -> str:
    if _AzureResponsesIdGenerator is not None:
        try:
            return _AzureResponsesIdGenerator.new_response_id(previous_response_id or "")
        except TypeError:  # Older/newer SDKs may expose this as a zero-argument helper.
            return _AzureResponsesIdGenerator.new_response_id()
    return f"caresp_{uuid.uuid4().hex}{uuid.uuid4().hex[:18]}"


def _new_message_id(response_id: str) -> str:
    if _AzureResponsesIdGenerator is not None:
        message_id = getattr(_AzureResponsesIdGenerator, "new_message_item_id", None)
        if callable(message_id):
            return message_id(response_id)
    return f"msg_{uuid.uuid4().hex}"


def _new_function_call_id(response_id: str) -> str:
    if _AzureResponsesIdGenerator is not None:
        function_call_id = getattr(_AzureResponsesIdGenerator, "new_function_call_item_id", None)
        if callable(function_call_id):
            return function_call_id(response_id)
    return f"fc_{uuid.uuid4().hex}"


def _usage(result: RunResult) -> dict[str, int]:
    usage = result.usage or {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
    }


def _responses_usage(result: RunResult | None = None, usage: Mapping[str, int] | None = None) -> dict[str, int]:
    raw = dict(usage or (result.usage if result is not None else {}) or {})
    input_tokens = int(raw.get("input_tokens", raw.get("prompt_tokens", 0)) or 0)
    output_tokens = int(raw.get("output_tokens", raw.get("completion_tokens", 0)) or 0)
    total_tokens = int(raw.get("total_tokens", input_tokens + output_tokens) or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _combine_usage(*usages: Mapping[str, int] | None) -> dict[str, int]:
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    for usage in usages:
        if not usage:
            continue
        prompt_count = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
        completion_count = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
        prompt_tokens += prompt_count
        completion_tokens += completion_count
        total_tokens += int(usage.get("total_tokens", prompt_count + completion_count) or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _error(message: str, status: int = 400, code: str | None = None) -> JSONResponse:
    return JSONResponse(
        {"error": {"message": message, "code": code}},
        status_code=status,
    )


def _message_to_prompt(message: Any) -> str:
    if isinstance(message, str):
        return message
    return json.dumps(message, separators=(",", ":"), sort_keys=True)


def _session_id_from_request(request: Request) -> str | None:
    # Foundry hosted agents may pass the session as a query parameter to the
    # container and expose it as x-agent-session-id externally. The AgentKit
    # header keeps local standalone validation provider-neutral.
    for name in ("agent_session_id", "session_id"):
        value = request.query_params.get(name)
        if value and value.strip():
            return value.strip()
    for name in ("x-agent-session-id", "x-agentkit-session-id"):
        value = request.headers.get(name)
        if value and value.strip():
            return value.strip()
    value = os.environ.get("FOUNDRY_AGENT_SESSION_ID")
    if value and value.strip():
        return value.strip()
    return None


def _responses_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("input_text") or block.get("output_text")
                if text is not None:
                    parts.append(str(text))
        return "".join(parts)
    return str(content)


def _responses_input_to_run_request(value: Any, *, session_id: str | None) -> RunRequest:
    """Extract a RunRequest from common non-streaming Responses API input shapes."""
    if isinstance(value, str):
        return RunRequest(prompt=value, session_id=session_id)

    if isinstance(value, list) and all(isinstance(item, dict) and "role" in item for item in value):
        history: list[ConversationTurn] = []
        for item in value:
            role = str(item.get("role") or "")
            text = _responses_content_to_text(item.get("content"))
            if role in FORWARDED_ROLES and text:
                history.append(ConversationTurn(role=role, text=text))
        if not history:
            return RunRequest(prompt="", session_id=session_id)
        last = history[-1]
        if last.role != "user":
            raise ValueError("Responses input list final message must have role 'user'")
        return RunRequest(prompt=last.text, history=tuple(history[:-1]), session_id=session_id)

    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = _responses_content_to_text(item.get("content"))
                if text:
                    parts.append(text)
                    continue
            parts.append(str(item))
        if parts:
            return RunRequest(prompt="\n".join(parts), session_id=session_id)

    return RunRequest(prompt=json.dumps(value, separators=(",", ":"), sort_keys=True), session_id=session_id)


def _responses_payload(spec: AgentSpec, result: RunResult, *, previous_response_id: str | None = None) -> dict[str, Any]:
    response_id = _new_response_id(previous_response_id)
    message_id = _new_message_id(response_id)
    payload: dict[str, Any] = {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": spec.model.name,
        "output": [
            {
                "id": message_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": result.text,
                        "annotations": [],
                    }
                ],
                "response_id": response_id,
            }
        ],
        "usage": _responses_usage(result),
    }
    if previous_response_id:
        payload["previous_response_id"] = previous_response_id
    return payload


@dataclass(frozen=True)
class _PendingCall:
    call_id: str
    item_id: str
    tool: BrokeredToolDefinition
    arguments: dict[str, Any]


@dataclass
class _HostedResponseState:
    response_id: str
    session_id: str | None
    pending_calls: dict[str, _PendingCall]
    expires_at: float
    status: str = "pending"
    accepted_outputs: dict[str, str] = field(default_factory=dict)
    final_payload: dict[str, Any] | None = None
    model_messages: list[dict[str, Any]] | None = None
    initial_usage: dict[str, int] = field(default_factory=dict)


class _StateExpired(KeyError):
    pass


class _StateStoreFull(Exception):
    pass


def _tool_to_state_payload(tool: BrokeredToolDefinition) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "brokeredClass": tool.brokered_class,
        "parameters": dict(tool.parameters),
        "schemaDigest": tool.schema_digest,
    }


def _tool_from_state_payload(data: Mapping[str, Any]) -> BrokeredToolDefinition:
    brokered_class = data.get("brokeredClass")
    if brokered_class not in {"read", "write", "coordination"}:
        raise ValueError("stored brokered tool class is invalid")
    parameters = data.get("parameters", {})
    if not isinstance(parameters, Mapping):
        raise ValueError("stored brokered tool parameters must be an object")
    schema_digest = data.get("schemaDigest")
    return BrokeredToolDefinition(
        name=str(data.get("name") or ""),
        description=str(data.get("description") or ""),
        brokered_class=brokered_class,  # type: ignore[arg-type]
        parameters=dict(parameters),
        schema_digest=str(schema_digest) if schema_digest else None,
    )


def _pending_call_to_state_payload(call: _PendingCall) -> dict[str, Any]:
    return {
        "callID": call.call_id,
        "itemID": call.item_id,
        "tool": _tool_to_state_payload(call.tool),
        "arguments": dict(call.arguments),
    }


def _pending_call_from_state_payload(data: Mapping[str, Any]) -> _PendingCall:
    tool = data.get("tool")
    if not isinstance(tool, Mapping):
        raise ValueError("stored pending call tool must be an object")
    arguments = data.get("arguments", {})
    if not isinstance(arguments, Mapping):
        raise ValueError("stored pending call arguments must be an object")
    return _PendingCall(
        call_id=str(data.get("callID") or ""),
        item_id=str(data.get("itemID") or ""),
        tool=_tool_from_state_payload(tool),
        arguments=dict(arguments),
    )


def _state_to_payload(state: _HostedResponseState) -> dict[str, Any]:
    return {
        "responseID": state.response_id,
        "sessionID": state.session_id,
        "pendingCalls": {call_id: _pending_call_to_state_payload(call) for call_id, call in state.pending_calls.items()},
        "expiresAt": state.expires_at,
        "status": state.status,
        "acceptedOutputs": dict(state.accepted_outputs),
        "finalPayload": state.final_payload,
        "modelMessages": state.model_messages,
        "initialUsage": dict(state.initial_usage),
    }


def _state_from_payload(data: Mapping[str, Any]) -> _HostedResponseState:
    pending_calls_raw = data.get("pendingCalls", {})
    if not isinstance(pending_calls_raw, Mapping):
        raise ValueError("stored pendingCalls must be an object")
    accepted_outputs = data.get("acceptedOutputs", {})
    if not isinstance(accepted_outputs, Mapping):
        raise ValueError("stored acceptedOutputs must be an object")
    final_payload = data.get("finalPayload")
    model_messages = data.get("modelMessages")
    initial_usage = data.get("initialUsage", {})
    if final_payload is not None and not isinstance(final_payload, dict):
        raise ValueError("stored finalPayload must be an object")
    if model_messages is not None and not isinstance(model_messages, list):
        raise ValueError("stored modelMessages must be an array")
    if not isinstance(initial_usage, Mapping):
        raise ValueError("stored initialUsage must be an object")
    status = str(data.get("status") or "pending")
    accepted = {str(key): str(value) for key, value in accepted_outputs.items()}
    if final_payload is None and accepted:
        # A persisted accepted output without a final payload means the process
        # stopped between accepting the continuation and completing the resume.
        # Clear it on load so Orka can retry the same continuation instead of
        # being stuck behind duplicate_continuation_in_progress until TTL expiry.
        status = "pending"
        accepted = {}
    return _HostedResponseState(
        response_id=str(data.get("responseID") or ""),
        session_id=str(data["sessionID"]) if data.get("sessionID") is not None else None,
        pending_calls={str(call_id): _pending_call_from_state_payload(call) for call_id, call in pending_calls_raw.items() if isinstance(call, Mapping)},
        expires_at=float(data.get("expiresAt") or 0),
        status=status,
        accepted_outputs=accepted,
        final_payload=final_payload,
        model_messages=model_messages,
        initial_usage={str(key): int(value or 0) for key, value in initial_usage.items()},
    )


class _FoundryResponseStateStore:
    """Continuation store for hosted Responses brokered calls.

    Without a file path this is in-memory only. With a file path, the store
    persists pending/final response state using atomic JSON writes so a restarted
    single-replica/sticky deployment can resume known response IDs.
    """

    def __init__(self, ttl_seconds: float, max_entries: int, state_file: str | Path | None = None) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.state_file = Path(state_file) if state_file else None
        self._states: dict[str, _HostedResponseState] = {}
        self._load()

    @property
    def backend_name(self) -> str:
        return "file" if self.state_file else "memory"

    def add(self, state: _HostedResponseState) -> None:
        self.purge_expired()
        if len(self._states) >= self.max_entries and state.response_id not in self._states:
            self.evict_completed_to_capacity(reserve_slots=1)
        if len(self._states) >= self.max_entries and state.response_id not in self._states:
            raise _StateStoreFull("too many pending brokered responses")
        self._states[state.response_id] = state
        self._persist()

    def save(self, state: _HostedResponseState) -> None:
        if state.response_id in self._states:
            self._states[state.response_id] = state
            self._persist()

    def evict_completed_to_capacity(self, *, reserve_slots: int = 0) -> None:
        target = max(self.max_entries - reserve_slots, 0)
        if len(self._states) <= target:
            return
        completed = sorted(
            (entry for entry in self._states.values() if entry.status == "completed" and entry.final_payload is not None),
            key=lambda entry: entry.expires_at,
        )
        changed = False
        for entry in completed:
            self._states.pop(entry.response_id, None)
            changed = True
            if len(self._states) <= target:
                break
        if changed:
            self._persist()

    def get(self, response_id: str) -> _HostedResponseState:
        state = self._states.get(response_id)
        if state is None:
            raise KeyError(response_id)
        if state.expires_at <= time.time():
            self._states.pop(response_id, None)
            self._persist()
            raise _StateExpired(response_id)
        self.purge_expired()
        return state

    def purge_expired(self) -> None:
        now = time.time()
        expired = [response_id for response_id, state in self._states.items() if state.expires_at <= now]
        for response_id in expired:
            self._states.pop(response_id, None)
        if expired:
            self._persist()

    def _load(self) -> None:
        if self.state_file is None or not self.state_file.exists():
            return
        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
            states = data.get("states", {}) if isinstance(data, Mapping) else {}
            if not isinstance(states, Mapping):
                raise ValueError("Foundry response state file states must be an object")
            self._states = {str(response_id): _state_from_payload(state) for response_id, state in states.items() if isinstance(state, Mapping)}
            self.purge_expired()
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            logger.warning("ignoring invalid Foundry response state file %s: %s", self.state_file, exc)
            self._states = {}

    def _persist(self) -> None:
        if self.state_file is None:
            return
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {"states": {response_id: _state_to_payload(state) for response_id, state in self._states.items()}}
        tmp = self.state_file.with_name(f".{self.state_file.name}.tmp")
        data = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(tmp, flags, 0o600)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            try:
                tmp.unlink(missing_ok=True)
            finally:
                raise
        os.chmod(tmp, 0o600)
        tmp.replace(self.state_file)
        os.chmod(self.state_file, 0o600)


def _state_ttl_seconds(value: float | None = None) -> float:
    if value is not None:
        parsed = float(value)
        return max(parsed, 0.0) if math.isfinite(parsed) else float(_DEFAULT_STATE_TTL_SECONDS)
    raw = os.environ.get(_STATE_TTL_ENV)
    if not raw:
        return float(_DEFAULT_STATE_TTL_SECONDS)
    try:
        parsed = float(raw)
    except ValueError:
        return float(_DEFAULT_STATE_TTL_SECONDS)
    return max(parsed, 0.0) if math.isfinite(parsed) else float(_DEFAULT_STATE_TTL_SECONDS)


def _positive_int_setting(value: int | None, *, env_name: str, default: int) -> int:
    if value is not None:
        return max(int(value), 1)
    raw = os.environ.get(env_name)
    if not raw:
        return default
    try:
        parsed = int(raw)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _max_pending_responses(value: int | None = None) -> int:
    return _positive_int_setting(value, env_name=_MAX_PENDING_ENV, default=_DEFAULT_MAX_PENDING_RESPONSES)


def _max_argument_bytes(value: int | None = None) -> int:
    return _positive_int_setting(value, env_name=_MAX_ARGUMENT_BYTES_ENV, default=_DEFAULT_MAX_ARGUMENT_BYTES)


def _max_output_bytes(value: int | None = None) -> int:
    return _positive_int_setting(value, env_name=_MAX_OUTPUT_BYTES_ENV, default=_DEFAULT_MAX_OUTPUT_BYTES)


def _brokered_model_loop_enabled(value: bool | None = None) -> bool:
    if value is not None:
        return value
    return os.environ.get(_MODEL_LOOP_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _response_state_file(value: str | Path | None = None) -> str | Path | None:
    if value is not None:
        return value
    raw = os.environ.get(_STATE_FILE_ENV)
    return raw.strip() if raw and raw.strip() else None


def _function_call_outputs_from_input(input_value: Any) -> list[dict[str, Any]]:
    items: list[Any]
    if isinstance(input_value, dict):
        items = [input_value]
    elif isinstance(input_value, list):
        items = input_value
    else:
        return []
    outputs: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict) and item.get("type") == "function_call_output":
            outputs.append(item)
    return outputs


def _reject_nonfinite_json_values(value: Any, *, path: str = "value") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} must be finite")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_nonfinite_json_values(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            _reject_nonfinite_json_values(child, path=f"{path}[{idx}]")


def _parse_output_float(raw: str) -> float:
    try:
        decimal = Decimal(raw)
    except InvalidOperation as exc:
        raise ValueError("function_call_output.output must contain valid JSON numbers") from exc
    parsed = float(decimal)
    if not math.isfinite(parsed) or Decimal(str(parsed)) != decimal:
        raise ValueError("function_call_output.output contains a number that cannot be represented exactly")
    return parsed


def _reject_output_constant(raw: str) -> None:
    raise ValueError(f"function_call_output.output contains non-finite number {raw}")


def _json_object_from_output(output: Any) -> dict[str, Any]:
    if not isinstance(output, str):
        raise ValueError("function_call_output.output must be a JSON object string")
    try:
        parsed = json.loads(
            output,
            parse_float=_parse_output_float,
            parse_constant=_reject_output_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError("function_call_output.output must be a JSON object string") from exc
    if not isinstance(parsed, dict):
        raise ValueError("function_call_output.output must be a JSON object")
    approved = parsed.get("approved")
    if not isinstance(approved, bool):
        raise ValueError("function_call_output.output.approved must be a boolean")
    if approved:
        unexpected = set(parsed) - {"approved", "output"}
        if unexpected:
            raise ValueError("approved function_call_output.output contains unsupported fields")
        tool_output = parsed.get("output", {})
        if tool_output is not None and not isinstance(tool_output, dict):
            raise ValueError("approved function_call_output.output.output must be an object")
    else:
        unexpected = set(parsed) - {"approved", "error"}
        if unexpected:
            raise ValueError("denied function_call_output.output contains unsupported fields")
        error = parsed.get("error", {})
        if error is not None and not isinstance(error, dict):
            raise ValueError("denied function_call_output.output.error must be an object")
    _reject_nonfinite_json_values(parsed, path="function_call_output.output")
    return json.loads(json.dumps(parsed, allow_nan=False, separators=(",", ":"), sort_keys=True))


def _canonical_output_json(output: dict[str, Any]) -> str:
    _reject_nonfinite_json_values(output)
    return json.dumps(output, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _first_typed_value(schema: Mapping[str, Any], key: str, expected_type: type) -> Any:
    if key in schema and isinstance(schema[key], expected_type):
        return schema[key]
    return None


def _enum_value(schema: Mapping[str, Any], expected_type: type) -> Any:
    values = schema.get("enum")
    if isinstance(values, list):
        for value in values:
            if isinstance(value, expected_type):
                return value
    return None


def _integer_schema_bound(value: Any, *, lower: bool, exclusive: bool) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        if exclusive:
            return value + 1 if lower else value - 1
        return value
    if isinstance(value, float) and math.isfinite(value):
        if lower:
            return math.floor(value) + 1 if exclusive else math.ceil(value)
        return math.ceil(value) - 1 if exclusive else math.floor(value)
    return None


def _effective_numeric_bounds(schema: Mapping[str, Any]) -> tuple[Fraction | None, bool, Fraction | None, bool]:
    lower_value: Fraction | None = None
    lower_open = False
    for key, exclusive in (("minimum", False), ("exclusiveMinimum", True)):
        raw = schema.get(key)
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            continue
        if isinstance(raw, float) and not math.isfinite(raw):
            raise AgentRunError("brokered tool schema has a non-finite numeric bound", status=400, code="UnsupportedBrokeredSchema")
        try:
            candidate = Fraction(str(raw))
        except (ValueError, ZeroDivisionError) as exc:
            raise AgentRunError("brokered tool schema has an invalid numeric bound", status=400, code="UnsupportedBrokeredSchema") from exc
        if lower_value is None or candidate > lower_value:
            lower_value = candidate
            lower_open = exclusive
        elif candidate == lower_value and exclusive:
            lower_open = True

    upper_value: Fraction | None = None
    upper_open = False
    for key, exclusive in (("maximum", False), ("exclusiveMaximum", True)):
        raw = schema.get(key)
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            continue
        if isinstance(raw, float) and not math.isfinite(raw):
            raise AgentRunError("brokered tool schema has a non-finite numeric bound", status=400, code="UnsupportedBrokeredSchema")
        try:
            candidate = Fraction(str(raw))
        except (ValueError, ZeroDivisionError) as exc:
            raise AgentRunError("brokered tool schema has an invalid numeric bound", status=400, code="UnsupportedBrokeredSchema") from exc
        if upper_value is None or candidate < upper_value:
            upper_value = candidate
            upper_open = exclusive
        elif candidate == upper_value and exclusive:
            upper_open = True
    return lower_value, lower_open, upper_value, upper_open


def _fraction_floor(value: Fraction) -> int:
    return value.numerator // value.denominator


def _fraction_ceil(value: Fraction) -> int:
    return -((-value.numerator) // value.denominator)


def _fraction_json_candidate(value: Fraction, *, name: str, require_exact: bool) -> int | float:
    if value.denominator == 1:
        integer_text = str(value.numerator)
        try:
            compact_float = float(value.numerator)
        except OverflowError:
            compact_float = math.inf
        if math.isfinite(compact_float) and Fraction(str(compact_float)) == value and len(str(compact_float)) < len(integer_text):
            return compact_float
        return value.numerator
    try:
        candidate = float(value)
    except OverflowError as exc:
        raise AgentRunError(
            f"brokered tool schema for {name!r} has no representable numeric value in bounds",
            status=400,
            code="UnsupportedBrokeredSchema",
        ) from exc
    if not math.isfinite(candidate) or (require_exact and Fraction(str(candidate)) != value):
        raise AgentRunError(
            f"brokered tool schema for {name!r} has no representable numeric value in bounds",
            status=400,
            code="UnsupportedBrokeredSchema",
        )
    return candidate


def _fraction_in_numeric_bounds(
    candidate: Fraction,
    *,
    lower_value: Fraction | None,
    lower_open: bool,
    upper_value: Fraction | None,
    upper_open: bool,
) -> bool:
    if lower_value is not None and (candidate < lower_value or (candidate == lower_value and lower_open)):
        return False
    if upper_value is not None and (candidate > upper_value or (candidate == upper_value and upper_open)):
        return False
    return True


def _integer_candidate_from_bounds(
    *,
    lower_value: Fraction | None,
    lower_open: bool,
    upper_value: Fraction | None,
    upper_open: bool,
    step: int = 1,
) -> int:
    if lower_value is not None:
        quotient = lower_value / step
        multiplier = _fraction_ceil(quotient)
        if lower_open and quotient.denominator == 1:
            multiplier += 1
        return multiplier * step
    if upper_value is not None:
        quotient = upper_value / step
        multiplier = _fraction_floor(quotient)
        if upper_open and quotient.denominator == 1:
            multiplier -= 1
        return multiplier * step
    return 0


def _required_property_names(schema: Mapping[str, Any]) -> list[str]:
    names: list[str] = []
    required = schema.get("required")
    if isinstance(required, list):
        names.extend(name for name in required if isinstance(name, str))

    dependent_required = schema.get("dependentRequired")
    min_properties = schema.get("minProperties")
    properties = schema.get("properties")

    changed = True
    while changed:
        changed = False
        if isinstance(dependent_required, Mapping):
            present = set(names)
            for trigger, dependent_names in dependent_required.items():
                if trigger not in present or not isinstance(dependent_names, list):
                    continue
                for dependent_name in dependent_names:
                    if isinstance(dependent_name, str) and dependent_name not in present:
                        names.append(dependent_name)
                        present.add(dependent_name)
                        changed = True
        if isinstance(min_properties, int) and not isinstance(min_properties, bool) and isinstance(properties, Mapping):
            for name in properties:
                if len(names) >= min_properties:
                    break
                if isinstance(name, str) and name not in names:
                    names.append(name)
                    changed = True
                    break

    if isinstance(min_properties, int) and not isinstance(min_properties, bool) and len(names) < min_properties:
        raise AgentRunError(
            "brokered tool schema has minProperties that cannot be synthesized",
            status=400,
            code="UnsupportedBrokeredSchema",
        )
    return names


def _sample_argument_value(name: str, schema: Any, run_request: RunRequest) -> Any:
    if not isinstance(schema, Mapping):
        return run_request.prompt
    if "multipleOf" in schema:
        raise AgentRunError(
            f"brokered tool schema for {name!r} has unsupported numeric multipleOf",
            status=400,
            code="UnsupportedBrokeredSchema",
        )
    if "const" in schema:
        return schema["const"]
    if "default" in schema:
        return schema["default"]
    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and enum_values:
        return enum_values[0]

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        schema_type = next((item for item in schema_type if item != "null"), schema_type[0] if schema_type else None)

    if schema_type == "null":
        return None

    if schema_type == "boolean":
        for key in ("const", "default"):
            value = _first_typed_value(schema, key, bool)
            if value is not None:
                return value
        value = _enum_value(schema, bool)
        return value if value is not None else True

    if schema_type == "integer":
        for key in ("const", "default"):
            value = _first_typed_value(schema, key, int)
            if value is not None and not isinstance(value, bool):
                return value
        value = _enum_value(schema, int)
        if value is not None and not isinstance(value, bool):
            return value
        lower_candidates = [
            candidate
            for candidate in (
                _integer_schema_bound(schema.get("minimum"), lower=True, exclusive=False),
                _integer_schema_bound(schema.get("exclusiveMinimum"), lower=True, exclusive=True),
            )
            if candidate is not None
        ]
        upper_candidates = [
            candidate
            for candidate in (
                _integer_schema_bound(schema.get("maximum"), lower=False, exclusive=False),
                _integer_schema_bound(schema.get("exclusiveMaximum"), lower=False, exclusive=True),
            )
            if candidate is not None
        ]
        lower = max(lower_candidates, default=0)
        upper = min(upper_candidates) if upper_candidates else None
        if upper is not None and lower > upper:
            if lower_candidates:
                raise AgentRunError(
                    f"brokered tool schema for {name!r} has incompatible integer bounds",
                    status=400,
                    code="UnsupportedBrokeredSchema",
                )
            lower = upper
        multiple_of = schema.get("multipleOf")
        if multiple_of is not None:
            raise AgentRunError(
                f"brokered tool schema for {name!r} has unsupported integer multipleOf",
                status=400,
                code="UnsupportedBrokeredSchema",
            )
        return lower

    if schema_type == "number":
        for key in ("const", "default"):
            value = schema.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return value
        value = _enum_value(schema, (int, float))
        if value is not None and not isinstance(value, bool):
            return value
        lower_fraction, lower_open, upper_fraction, upper_open = _effective_numeric_bounds(schema)
        has_lower = lower_fraction is not None
        has_upper = upper_fraction is not None
        if lower_fraction is not None and upper_fraction is not None:
            if lower_fraction > upper_fraction or (lower_fraction == upper_fraction and (lower_open or upper_open)):
                raise AgentRunError(
                    f"brokered tool schema for {name!r} has incompatible numeric bounds",
                    status=400,
                    code="UnsupportedBrokeredSchema",
                )
            if lower_fraction == upper_fraction:
                return _fraction_json_candidate(lower_fraction, name=name, require_exact=True)

        multiple_of = schema.get("multipleOf")
        if multiple_of is not None:
            raise AgentRunError(
                f"brokered tool schema for {name!r} has unsupported numeric multipleOf",
                status=400,
                code="UnsupportedBrokeredSchema",
            )

        candidates: list[int | float] = []

        def add_candidate(candidate: int | float) -> None:
            if isinstance(candidate, float) and not math.isfinite(candidate):
                return
            if _fraction_in_numeric_bounds(
                Fraction(str(candidate)),
                lower_value=lower_fraction,
                lower_open=lower_open,
                upper_value=upper_fraction,
                upper_open=upper_open,
            ):
                candidates.append(candidate)

        add_candidate(0)

        for boundary, is_open in (
            (lower_fraction, lower_open),
            (upper_fraction, upper_open),
        ):
            if boundary is None or is_open:
                continue
            try:
                add_candidate(_fraction_json_candidate(boundary, name=name, require_exact=True))
            except AgentRunError:
                pass

        for boundary, is_open, direction in (
            (lower_fraction, lower_open, math.inf),
            (upper_fraction, upper_open, -math.inf),
        ):
            if boundary is None or not is_open:
                continue
            try:
                boundary_float = float(boundary)
            except OverflowError:
                continue
            if math.isfinite(boundary_float):
                add_candidate(math.nextafter(boundary_float, direction))

        integer_candidate = _integer_candidate_from_bounds(
            lower_value=lower_fraction,
            lower_open=lower_open,
            upper_value=upper_fraction,
            upper_open=upper_open,
            step=1,
        )
        add_candidate(integer_candidate)

        if lower_fraction is not None and upper_fraction is not None:
            midpoint = (lower_fraction + upper_fraction) / 2
            try:
                add_candidate(_fraction_json_candidate(midpoint, name=name, require_exact=False))
            except AgentRunError:
                pass

        if candidates:
            return min(
                candidates,
                key=lambda candidate: len(json.dumps(candidate, allow_nan=False, separators=(",", ":"))),
            )

        raise AgentRunError(
            f"brokered tool schema for {name!r} has no representable numeric value in bounds",
            status=400,
            code="UnsupportedBrokeredSchema",
        )

    if schema_type == "array":
        for key in ("const", "default"):
            value = _first_typed_value(schema, key, list)
            if value is not None:
                return value
        value = _enum_value(schema, list)
        if value is not None:
            return value
        min_items = schema.get("minItems", 0)
        if isinstance(min_items, int) and min_items > 0:
            if min_items > _MAX_SYNTHETIC_ARRAY_ITEMS:
                raise AgentRunError(
                    f"brokered tool schema for {name!r} has minItems too large for deterministic synthesis",
                    status=413,
                    code="brokered_arguments_too_large",
                )
            item_schema = schema.get("items", {})
            return [_sample_argument_value(name, item_schema, run_request) for _ in range(min_items)]
        return []

    if schema_type == "object":
        for key in ("const", "default"):
            value = _first_typed_value(schema, key, dict)
            if value is not None:
                return value
        nested: dict[str, Any] = {}
        properties = schema.get("properties") if isinstance(schema.get("properties"), Mapping) else {}
        for child_name in _required_property_names(schema):
            nested[child_name] = _sample_argument_value(child_name, properties.get(child_name, {}), run_request)
        return nested

    for key in ("const", "default"):
        value = _first_typed_value(schema, key, str)
        if value is not None:
            return value
    value = _enum_value(schema, str)
    if value is not None:
        return value
    if schema.get("pattern"):
        raise AgentRunError(
            f"brokered tool schema for {name!r} uses pattern without const/default/enum",
            status=400,
            code="UnsupportedBrokeredSchema",
        )
    sample = run_request.prompt if name in {"prompt", "site"} else name
    min_length = schema.get("minLength", 0)
    if isinstance(min_length, int) and min_length > _MAX_SYNTHETIC_STRING_LENGTH:
        raise AgentRunError(
            f"brokered tool schema for {name!r} has minLength too large for deterministic synthesis",
            status=413,
            code="brokered_arguments_too_large",
        )
    if isinstance(min_length, int) and len(sample) < min_length:
        sample = sample + ("x" * (min_length - len(sample)))
    max_length = schema.get("maxLength")
    if isinstance(max_length, int) and max_length >= 0 and len(sample) > max_length:
        if isinstance(min_length, int) and min_length > max_length:
            raise AgentRunError(
                f"brokered tool schema for {name!r} has incompatible minLength/maxLength",
                status=400,
                code="UnsupportedBrokeredSchema",
            )
        sample = sample[:max_length]
    return sample


def _schema_has_literal_value(schema: Any) -> bool:
    return isinstance(schema, Mapping) and ("const" in schema or "default" in schema or (isinstance(schema.get("enum"), list) and len(schema.get("enum")) == 1) or schema.get("type") == "null")


def _deterministic_tool_arguments(tool: BrokeredToolDefinition, run_request: RunRequest) -> dict[str, Any]:
    parameters = tool.parameters if isinstance(tool.parameters, Mapping) else {}
    for key in ("const", "default"):
        literal = parameters.get(key)
        if isinstance(literal, Mapping):
            return dict(literal)
    enum = parameters.get("enum")
    if isinstance(enum, list):
        if tool.brokered_class != "read" and len(enum) > 1:
            raise AgentRunError(
                f"brokered {tool.brokered_class} tool {tool.name!r} has multiple root enum payloads; deterministic mode refuses to choose side-effecting arguments",
                status=400,
                code="UnsupportedBrokeredSchema",
            )
        for item in enum:
            if isinstance(item, Mapping):
                return dict(item)
    properties = parameters.get("properties") if isinstance(parameters.get("properties"), Mapping) else {}
    required_names = _required_property_names(parameters)
    if tool.brokered_class != "read":
        for name in required_names:
            if not _schema_has_literal_value(properties.get(name, {})):
                raise AgentRunError(
                    f"brokered {tool.brokered_class} tool {tool.name!r} requires non-literal argument {name!r}; deterministic mode refuses to synthesize side-effecting arguments",
                    status=400,
                    code="UnsupportedBrokeredSchema",
                )
    arguments: dict[str, Any] = {}
    for name in required_names:
        arguments[name] = _sample_argument_value(name, properties.get(name, {}), run_request)
    if tool.name == "conformance_read" and "probe" in properties and "probe" not in arguments:
        arguments["probe"] = True
    if not arguments and "prompt" in properties:
        if tool.brokered_class != "read" and not _schema_has_literal_value(properties["prompt"]):
            raise AgentRunError(
                f"brokered {tool.brokered_class} tool {tool.name!r} has a non-literal optional prompt; deterministic mode refuses to synthesize side-effecting arguments",
                status=400,
                code="UnsupportedBrokeredSchema",
            )
        arguments["prompt"] = _sample_argument_value("prompt", properties["prompt"], run_request)
    return arguments


def _prompt_mentions_tool(prompt: str, tool_name: str) -> bool:
    pattern = r"(?<![A-Za-z0-9_-])" + re.escape(tool_name) + r"(?![A-Za-z0-9_-])"
    return re.search(pattern, prompt, flags=re.IGNORECASE) is not None


def _select_brokered_tool(tools: list[BrokeredToolDefinition], run_request: RunRequest) -> BrokeredToolDefinition | None:
    matches = [tool for tool in tools if _prompt_mentions_tool(run_request.prompt, tool.name)]
    if len(matches) == 1:
        return matches[0]
    if len(tools) == 1 and tools[0].name == "conformance_read":
        return tools[0]
    return None




def _validate_model_brokered_arguments(value: Any, *, path: str = "arguments") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise AgentRunError("model brokered tool argument keys must be strings", status=400, code="InvalidToolArguments")
            if _unsafe_brokered_key(key) is not None:
                raise AgentRunError(
                    f"model brokered tool argument {path}.{key} is not safe",
                    status=400,
                    code="UnsafeBrokeredArguments",
                )
            _validate_model_brokered_arguments(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            _validate_model_brokered_arguments(child, path=f"{path}[{idx}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise AgentRunError(
            f"model brokered tool argument {path} must be finite",
            status=400,
            code="InvalidToolArguments",
        )
    elif isinstance(value, str) and _unsafe_brokered_text(value):
        raise AgentRunError(
            f"model brokered tool argument {path} contains unsafe text",
            status=400,
            code="UnsafeBrokeredArguments",
        )




def _schema_types(schema: Mapping[str, Any]) -> list[str]:
    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        return [schema_type]
    if isinstance(schema_type, list):
        return [item for item in schema_type if isinstance(item, str)]
    return []


def _decimal_json_number(value: int | float) -> Decimal:
    try:
        decimal = Decimal(str(value))
    except InvalidOperation as exc:
        raise AgentRunError("model brokered tool argument is not a finite JSON number", status=400, code="InvalidToolArguments") from exc
    if not decimal.is_finite():
        raise AgentRunError("model brokered tool argument is not a finite JSON number", status=400, code="InvalidToolArguments")
    return decimal


def _json_type_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return _decimal_json_number(left) == _decimal_json_number(right)
    if isinstance(left, str) or isinstance(right, str):
        return isinstance(left, str) and isinstance(right, str) and left == right
    if isinstance(left, list) or isinstance(right, list):
        return (
            isinstance(left, list)
            and isinstance(right, list)
            and len(left) == len(right)
            and all(_json_type_equal(a, b) for a, b in zip(left, right))
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_json_type_equal(left[key], right[key]) for key in left)
        )
    return left == right


def _value_matches_type(value: Any, schema_type: str) -> bool:
    if schema_type == "null":
        return value is None
    if schema_type == "boolean":
        return isinstance(value, bool)
    if schema_type == "integer":
        return (isinstance(value, int) and not isinstance(value, bool)) or (
            isinstance(value, float) and math.isfinite(value) and value.is_integer()
        )
    if schema_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "array":
        return isinstance(value, list)
    if schema_type == "object":
        return isinstance(value, Mapping)
    return True


def _validate_model_argument_against_schema(value: Any, schema: Any, *, path: str) -> None:
    if not isinstance(schema, Mapping):
        return
    if "const" in schema and not _json_type_equal(value, schema["const"]):
        raise AgentRunError(f"model brokered tool argument {path} does not match const", status=400, code="InvalidToolArguments")
    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and not any(_json_type_equal(value, enum_value) for enum_value in enum_values):
        raise AgentRunError(f"model brokered tool argument {path} is not in enum", status=400, code="InvalidToolArguments")
    types = _schema_types(schema)
    if types and not any(_value_matches_type(value, schema_type) for schema_type in types):
        raise AgentRunError(f"model brokered tool argument {path} has wrong type", status=400, code="InvalidToolArguments")

    if isinstance(value, str):
        min_length = schema.get("minLength")
        if isinstance(min_length, int) and len(value) < min_length:
            raise AgentRunError(f"model brokered tool argument {path} is too short", status=400, code="InvalidToolArguments")
        max_length = schema.get("maxLength")
        if isinstance(max_length, int) and len(value) > max_length:
            raise AgentRunError(f"model brokered tool argument {path} is too long", status=400, code="InvalidToolArguments")
        pattern = schema.get("pattern")
        if isinstance(pattern, str):
            try:
                matches = re.search(pattern, value) is not None
            except re.error as exc:
                raise AgentRunError(
                    f"brokered tool schema pattern for {path} is not supported by the model-loop validator",
                    status=400,
                    code="UnsupportedBrokeredSchema",
                ) from exc
            if not matches:
                raise AgentRunError(f"model brokered tool argument {path} does not match pattern", status=400, code="InvalidToolArguments")

    if isinstance(value, int) and not isinstance(value, bool) or isinstance(value, float):
        number = _decimal_json_number(value)
        for key, compare in (("minimum", lambda a, b: a >= b), ("exclusiveMinimum", lambda a, b: a > b), ("maximum", lambda a, b: a <= b), ("exclusiveMaximum", lambda a, b: a < b)):
            bound = schema.get(key)
            if isinstance(bound, (int, float)) and not isinstance(bound, bool) and not compare(number, _decimal_json_number(bound)):
                raise AgentRunError(f"model brokered tool argument {path} violates {key}", status=400, code="InvalidToolArguments")
        multiple_of = schema.get("multipleOf")
        if isinstance(multiple_of, (int, float)) and not isinstance(multiple_of, bool) and multiple_of > 0:
            divisor = _decimal_json_number(multiple_of)
            if divisor == 0 or number % divisor != 0:
                raise AgentRunError(f"model brokered tool argument {path} violates multipleOf", status=400, code="InvalidToolArguments")

    if isinstance(value, list):
        min_items = schema.get("minItems")
        if isinstance(min_items, int) and len(value) < min_items:
            raise AgentRunError(f"model brokered tool argument {path} has too few items", status=400, code="InvalidToolArguments")
        max_items = schema.get("maxItems")
        if isinstance(max_items, int) and len(value) > max_items:
            raise AgentRunError(f"model brokered tool argument {path} has too many items", status=400, code="InvalidToolArguments")
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for idx, item in enumerate(value):
                _validate_model_argument_against_schema(item, item_schema, path=f"{path}[{idx}]")

    if isinstance(value, Mapping):
        required = schema.get("required")
        if isinstance(required, list):
            for name in required:
                if isinstance(name, str) and name not in value:
                    raise AgentRunError(f"model brokered tool argument {path}.{name} is required", status=400, code="InvalidToolArguments")
        dependent_required = schema.get("dependentRequired")
        if isinstance(dependent_required, Mapping):
            for trigger, dependent_names in dependent_required.items():
                if trigger not in value or not isinstance(dependent_names, list):
                    continue
                for dependent_name in dependent_names:
                    if isinstance(dependent_name, str) and dependent_name not in value:
                        raise AgentRunError(f"model brokered tool argument {path}.{dependent_name} is required", status=400, code="InvalidToolArguments")
        min_properties = schema.get("minProperties")
        if isinstance(min_properties, int) and len(value) < min_properties:
            raise AgentRunError(f"model brokered tool argument {path} has too few properties", status=400, code="InvalidToolArguments")
        max_properties = schema.get("maxProperties")
        if isinstance(max_properties, int) and len(value) > max_properties:
            raise AgentRunError(f"model brokered tool argument {path} has too many properties", status=400, code="InvalidToolArguments")
        properties = schema.get("properties") if isinstance(schema.get("properties"), Mapping) else {}
        additional = schema.get("additionalProperties")
        if additional is False:
            for key in value:
                if key not in properties:
                    raise AgentRunError(f"model brokered tool argument {path}.{key} is not declared", status=400, code="InvalidToolArguments")
        elif isinstance(additional, Mapping):
            for key, child in value.items():
                if key not in properties:
                    _validate_model_argument_against_schema(child, additional, path=f"{path}.{key}")
        for key, child_schema in properties.items():
            if isinstance(key, str) and key in value:
                _validate_model_argument_against_schema(value[key], child_schema, path=f"{path}.{key}")


def _validate_model_arguments_for_tool(arguments: Mapping[str, Any], tool: BrokeredToolDefinition) -> None:
    _validate_model_argument_against_schema(dict(arguments), tool.parameters, path="arguments")


def _function_call_response_payload(
    spec: AgentSpec,
    *,
    response_id: str,
    call: _PendingCall,
    previous_response_id: str | None = None,
    usage: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": spec.model.name,
        "output": [
            {
                "id": call.item_id,
                "type": "function_call",
                "call_id": call.call_id,
                "name": call.tool.name,
                "arguments": _canonical_output_json(call.arguments),
                "status": "completed",
                "response_id": response_id,
            }
        ],
        "usage": _responses_usage(usage=usage),
    }
    if previous_response_id:
        payload["previous_response_id"] = previous_response_id
    return payload


def _final_text_from_tool_output(call: _PendingCall, output: dict[str, Any]) -> str:
    if not output.get("approved"):
        error = output.get("error") if isinstance(output.get("error"), dict) else {}
        code = str(error.get("code") or "brokered_tool_denied")
        message = str(error.get("message") or "brokered tool was not performed")
        return f"Brokered tool {call.tool.name} was not performed: {code}: {message}"
    tool_output = output.get("output") if isinstance(output.get("output"), dict) else {}
    return f"Brokered tool {call.tool.name} completed with output: {_canonical_output_json(tool_output)}"


async def _handle_brokered_continuation(
    *,
    spec: AgentSpec,
    store: _FoundryResponseStateStore,
    previous_response_id: str | None,
    input_value: Any,
    continuation_proof: str | None,
    request: Request,
    max_output_bytes: int,
    model_loop: BrokeredChatModelLoop | None = None,
) -> JSONResponse:
    if not continuation_proof:
        return _error(
            "brokered continuation proof is not configured; refusing function_call_output",
            status=503,
            code="brokered_continuation_auth_required",
        )
    provided_proof = request.headers.get(_CONTINUATION_PROOF_HEADER)
    if provided_proof != continuation_proof:
        return _error(
            "function_call_output is restricted to the Orka broker continuation path",
            status=403,
            code="brokered_continuation_forbidden",
        )
    outputs = _function_call_outputs_from_input(input_value)
    if not outputs:
        return _error(
            "Responses continuation requires a function_call_output input item",
            status=400,
            code="missing_function_call_output",
        )
    if previous_response_id is None or not str(previous_response_id).strip():
        return _error(
            "function_call_output requires previous_response_id",
            status=400,
            code="missing_previous_response_id",
        )
    if len(outputs) != 1:
        return _error(
            "multiple function_call_output items are not supported by this deterministic brokered adapter",
            status=400,
            code="multiple_tool_outputs_unsupported",
        )
    try:
        state = store.get(str(previous_response_id))
    except _StateExpired:
        return _error("previous_response_id state has expired", status=410, code="response_state_expired")
    except KeyError:
        return _error("unknown previous_response_id", status=404, code="unknown_previous_response_id")

    item = outputs[0]
    call_id = item.get("call_id")
    if not isinstance(call_id, str) or not call_id:
        return _error("function_call_output.call_id is required", status=400, code="missing_call_id")
    call = state.pending_calls.get(call_id)
    if call is None:
        return _error("unknown function_call_output call_id", status=400, code="unknown_call_id")
    try:
        parsed_output = _json_object_from_output(item.get("output"))
    except ValueError as exc:
        return _error(str(exc), status=400, code="invalid_function_call_output")
    output_json = _canonical_output_json(parsed_output)
    output_size = len(output_json.encode("utf-8"))
    if output_size > max_output_bytes:
        return _error(
            "brokered function_call_output is too large",
            status=413,
            code="brokered_output_too_large",
        )

    existing_output = state.accepted_outputs.get(call_id)
    if existing_output is not None:
        if existing_output == output_json and state.final_payload is not None:
            return JSONResponse(state.final_payload)
        if existing_output == output_json:
            return _error(
                "matching function_call_output is already being processed",
                status=409,
                code="duplicate_continuation_in_progress",
            )
        return _error(
            "conflicting duplicate function_call_output for call_id",
            status=409,
            code="conflicting_duplicate_continuation",
        )
    if state.status != "pending":
        return _error("previous response is not pending a tool result", status=409, code="response_not_pending")

    state.accepted_outputs[call_id] = output_json
    store.save(state)
    if state.model_messages is not None and model_loop is not None:
        state.status = "resuming"
        store.save(state)
        try:
            model_result = await model_loop.resume(state.model_messages, call_id=call_id, output=output_json)
        except AgentRunError as exc:
            state.accepted_outputs.pop(call_id, None)
            state.status = "pending"
            store.save(state)
            if exc.status >= 500:
                logger.warning("brokered model-loop resume failed: %s", exc)
                return _error("model resume failed", status=exc.status, code="ModelResumeError")
            return _error(str(exc), status=exc.status, code=exc.code)
        except asyncio.CancelledError:
            state.accepted_outputs.pop(call_id, None)
            state.status = "pending"
            store.save(state)
            raise
        except Exception as exc:  # noqa: BLE001 - reset continuation state before surfacing unexpected model failures.
            logger.exception("brokered model-loop resume failed unexpectedly")
            state.accepted_outputs.pop(call_id, None)
            state.status = "pending"
            store.save(state)
            return _error("model resume failed", status=502, code="ModelResumeError")
        if not isinstance(model_result, ModelLoopFinal):
            state.accepted_outputs.pop(call_id, None)
            state.status = "pending"
            store.save(state)
            return _error(
                "model requested another brokered tool after resume",
                status=400,
                code="tool_loop_limit_exceeded",
            )
        result = RunResult(text=model_result.text, usage=_combine_usage(state.initial_usage, model_result.usage))
    else:
        result = RunResult(text=_final_text_from_tool_output(call, parsed_output))
    final_payload = _responses_payload(spec, result, previous_response_id=state.response_id)
    state.status = "completed"
    state.final_payload = final_payload
    store.save(state)
    store.evict_completed_to_capacity()
    return JSONResponse(final_payload)


def create_foundry_app(
    spec: AgentSpec,
    factory: RuntimeFactory,
    auth_token: str | None = None,
    *,
    state_ttl_seconds: float | None = None,
    brokered_continuation_proof: str | None = None,
    max_pending_responses: int | None = None,
    max_brokered_argument_bytes: int | None = None,
    max_brokered_output_bytes: int | None = None,
    brokered_model_loop_enabled: bool | None = None,
    brokered_model_http_client: Any | None = None,
    response_state_file: str | Path | None = None,
) -> FastAPI:
    """Create a Foundry-compatible wrapper app for one AgentKit runtime."""
    brokered_tools = brokered_tool_definitions(spec)
    runtime = None if brokered_tools else factory.build_runtime(spec)
    continuation_proof = brokered_continuation_proof or os.environ.get(_CONTINUATION_PROOF_ENV) or None
    max_argument_bytes = _max_argument_bytes(max_brokered_argument_bytes)
    max_output_bytes = _max_output_bytes(max_brokered_output_bytes)
    response_states = _FoundryResponseStateStore(
        ttl_seconds=_state_ttl_seconds(state_ttl_seconds),
        max_entries=_max_pending_responses(max_pending_responses),
        state_file=_response_state_file(response_state_file) if brokered_tools else None,
    )
    model_loop = (
        BrokeredChatModelLoop(
            spec,
            brokered_tools,
            http_client=brokered_model_http_client,
            max_output_bytes=max_output_bytes,
        )
        if brokered_tools and _brokered_model_loop_enabled(brokered_model_loop_enabled)
        else None
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if runtime is None:
            yield
            return
        async with runtime:
            app.state.runtime = runtime
            yield

    app = FastAPI(lifespan=lifespan)
    auth = Depends(make_auth_dependency(auth_token))

    @app.get("/readiness")
    async def readiness():
        body: dict[str, Any] = {"ready": True}
        if brokered_tools:
            body["foundryResponses"] = {
                "brokeredTools": len(brokered_tools),
                "ownedToolsDisabled": len(spec.tools),
                "stateBackend": response_states.backend_name,
                "stateTtlSeconds": response_states.ttl_seconds,
                "stateMaxPending": response_states.max_entries,
                "continuationAuth": "configured" if continuation_proof else "missing",
                "runtime": "model-loop" if model_loop is not None else "deterministic",
                "scaling": "single-replica-or-sticky-routing-required",
            }
            if not continuation_proof:
                body["ready"] = False
                return JSONResponse(body, status_code=503)
        return body

    @app.post("/invocations", dependencies=[auth])
    async def invocations(request: Request):
        if brokered_tools:
            return _error(
                "Foundry /invocations is disabled when brokeredTools are configured; use /responses so Orka can broker tools",
                status=400,
                code="invocations_disabled_in_brokered_mode",
            )
        try:
            data = await request.json()
        except json.JSONDecodeError:
            return Response("Request body must be JSON", status_code=400)

        if not isinstance(data, dict):
            return Response("Request body must be a JSON object", status_code=400)
        if "message" not in data:
            return Response("Missing 'message' in request", status_code=400)
        prompt = _message_to_prompt(data["message"])

        try:
            result = await request.app.state.runtime.run(
                RunRequest(prompt=prompt, session_id=_session_id_from_request(request))
            )
        except AgentRunError as exc:
            return _error(str(exc), status=exc.status, code=exc.code)
        except Exception as exc:  # noqa: BLE001 - deterministic protocol envelope.
            return _error(str(exc), status=502, code=exc.__class__.__name__)

        return JSONResponse({"response": result.text, "usage": _usage(result)})

    @app.post("/responses", dependencies=[auth])
    async def responses(request: Request):
        try:
            data = await request.json()
        except json.JSONDecodeError:
            return _error("Request body must be JSON", status=400, code="invalid_json")

        if not isinstance(data, dict):
            return _error("Request body must be a JSON object", status=400, code="invalid_request")
        # Foundry/azd clients may include stream=true by default. The adapter is
        # intentionally non-streaming, so tolerate the flag and return a normal
        # completed response instead of failing readiness/e2e checks.
        if data.get("tools"):
            return _error(
                "request-supplied Responses tools are not allowed; hosted brokered mode uses static safe schemas",
                status=400,
                code="tools_unsupported",
            )
        tool_choice = data.get("tool_choice")
        if tool_choice not in (None, "", "none", "auto") or (brokered_tools and tool_choice == "none"):
            return _error(
                "request-supplied Responses tool_choice is not allowed; hosted brokered mode owns tool selection",
                status=400,
                code="tool_choice_unsupported",
            )
        if "input" not in data:
            return _error("Missing 'input' in request", status=400, code="missing_input")

        previous_response_id = data.get("previous_response_id")
        function_outputs = _function_call_outputs_from_input(data["input"])
        if brokered_tools and function_outputs:
            return await _handle_brokered_continuation(
                spec=spec,
                store=response_states,
                previous_response_id=previous_response_id if isinstance(previous_response_id, str) else None,
                input_value=data["input"],
                continuation_proof=continuation_proof,
                request=request,
                max_output_bytes=max_output_bytes,
                model_loop=model_loop,
            )
        if brokered_tools and isinstance(previous_response_id, str) and previous_response_id:
            try:
                previous_state = response_states.get(previous_response_id)
            except (KeyError, _StateExpired):
                previous_state = None
            if previous_state is not None and previous_state.status in {"pending", "resuming"}:
                return _error(
                    "previous_response_id is pending a brokered function_call_output",
                    status=409,
                    code="response_pending_function_call_output",
                )

        try:
            run_request = _responses_input_to_run_request(
                data["input"],
                session_id=_session_id_from_request(request),
            )
        except ValueError as exc:
            return _error(str(exc), status=400, code="invalid_input")

        if brokered_tools:
            if not continuation_proof:
                return _error(
                    "brokered continuation proof is not configured; refusing to start an uncontinuable brokered response",
                    status=503,
                    code="brokered_continuation_auth_required",
                )
            previous_response_id_for_output = previous_response_id if isinstance(previous_response_id, str) and previous_response_id else None
            if model_loop is not None:
                response_id = _new_response_id(previous_response_id_for_output)
                call_id = f"call_{response_id}_1"
                try:
                    model_result = await model_loop.start(run_request, call_id=call_id)
                except AgentRunError as exc:
                    return _error(str(exc), status=exc.status, code=exc.code)
                if isinstance(model_result, ModelLoopFinal):
                    return JSONResponse(_responses_payload(spec, RunResult(text=model_result.text, usage=model_result.usage), previous_response_id=previous_response_id_for_output))
                tool = {tool.name: tool for tool in brokered_tools}.get(model_result.name)
                if tool is None:
                    return _error("model requested unknown brokered tool", status=400, code="unknown_brokered_tool")
                try:
                    _validate_model_brokered_arguments(model_result.arguments)
                    _validate_model_arguments_for_tool(model_result.arguments, tool)
                except AgentRunError as exc:
                    return _error(str(exc), status=exc.status, code=exc.code)
                if len(_canonical_output_json(model_result.arguments).encode("utf-8")) > max_argument_bytes:
                    return _error(
                        "brokered function_call arguments are too large for pending state",
                        status=413,
                        code="brokered_arguments_too_large",
                    )
                call = _PendingCall(
                    call_id=call_id,
                    item_id=_new_function_call_id(response_id),
                    tool=tool,
                    arguments=model_result.arguments,
                )
                state = _HostedResponseState(
                    response_id=response_id,
                    session_id=run_request.session_id,
                    pending_calls={call_id: call},
                    expires_at=time.time() + response_states.ttl_seconds,
                    model_messages=model_result.messages,
                    initial_usage=dict(model_result.usage),
                )
                try:
                    response_states.add(state)
                except _StateStoreFull:
                    return _error(
                        "too many pending brokered responses",
                        status=429,
                        code="brokered_response_state_full",
                    )
                return JSONResponse(
                    _function_call_response_payload(
                        spec,
                        response_id=response_id,
                        call=call,
                        previous_response_id=previous_response_id_for_output,
                        usage=model_result.usage,
                    )
                )
            tool = _select_brokered_tool(brokered_tools, run_request)
            if tool is None:
                return _error(
                    "prompt must name exactly one configured brokered tool in deterministic brokered mode",
                    status=400,
                    code="brokered_tool_selection_required",
                )
            response_id = _new_response_id(previous_response_id_for_output)
            call_id = f"call_{response_id}_1"
            try:
                arguments = _deterministic_tool_arguments(tool, run_request)
                _validate_model_brokered_arguments(arguments)
                _validate_model_arguments_for_tool(arguments, tool)
            except AgentRunError as exc:
                return _error(str(exc), status=exc.status, code=exc.code)
            if len(_canonical_output_json(arguments).encode("utf-8")) > max_argument_bytes:
                return _error(
                    "brokered function_call arguments are too large for pending state",
                    status=413,
                    code="brokered_arguments_too_large",
                )
            call = _PendingCall(
                call_id=call_id,
                item_id=_new_function_call_id(response_id),
                tool=tool,
                arguments=arguments,
            )
            state = _HostedResponseState(
                response_id=response_id,
                session_id=run_request.session_id,
                pending_calls={call_id: call},
                expires_at=time.time() + response_states.ttl_seconds,
            )
            try:
                response_states.add(state)
            except _StateStoreFull:
                return _error(
                    "too many pending brokered responses",
                    status=429,
                    code="brokered_response_state_full",
                )
            return JSONResponse(
                _function_call_response_payload(
                    spec,
                    response_id=response_id,
                    call=call,
                    previous_response_id=previous_response_id_for_output,
                )
            )

        try:
            result = await request.app.state.runtime.run(run_request)
        except AgentRunError as exc:
            return _error(str(exc), status=exc.status, code=exc.code)
        except Exception as exc:  # noqa: BLE001 - deterministic protocol envelope.
            return _error(str(exc), status=502, code=exc.__class__.__name__)

        return JSONResponse(_responses_payload(spec, result))

    return app
