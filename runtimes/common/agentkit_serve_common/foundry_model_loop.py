"""Lower-level OpenAI-compatible brokered tool loop for Foundry Responses mode.

This is the Phase A4 fallback path: when high-level frameworks cannot suspend and
resume externally brokered tool calls, AgentKit can drive a minimal model loop
itself. The loop exposes only static safe brokered schemas to the model, converts
one model tool request into a hosted Responses function_call, and later resumes
with Orka's function_call_output to obtain the final assistant message.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Sequence

import httpx

from .adapter_support import AgentBuildError, resolve_api_key, resolve_workload_identity_token
from .config import AgentSpec
from .conversation import FORWARDED_ROLES, RunRequest
from .runtime import AgentRunError, BrokeredToolDefinition

_MAX_ARGUMENT_DEPTH = 128


@dataclass(frozen=True)
class ModelLoopFinal:
    text: str
    usage: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelLoopToolRequest:
    name: str
    arguments: dict[str, Any]
    messages: list[dict[str, Any]]
    usage: dict[str, int] = field(default_factory=dict)


class BrokeredChatModelLoop:
    """Explicit one-tool brokered model loop over OpenAI Chat Completions."""

    def __init__(
        self,
        spec: AgentSpec,
        tools: Sequence[BrokeredToolDefinition],
        *,
        http_client: httpx.AsyncClient | None = None,
        max_argument_bytes: int = 8192,
        max_output_bytes: int = 64 * 1024,
    ) -> None:
        self.spec = spec
        self.tools = list(tools)
        self.http_client = http_client
        self.max_argument_bytes = max_argument_bytes
        self.max_output_bytes = max_output_bytes
        self.tools_by_name = {tool.name: tool for tool in self.tools}

    async def start(self, request: RunRequest, *, call_id: str) -> ModelLoopFinal | ModelLoopToolRequest:
        messages = self._initial_messages(request)
        data = await self._chat(messages, tools=self._tool_payloads())
        message = _choice_message(data)
        usage = _usage(data)
        tool_calls = message.get("tool_calls")
        if not tool_calls:
            return ModelLoopFinal(text=_message_text(message), usage=usage)
        if not isinstance(tool_calls, list) or len(tool_calls) != 1:
            raise AgentRunError(
                "model requested multiple brokered tools; deterministic brokered mode supports one call per turn",
                status=400,
                code="multiple_tool_calls_unsupported",
            )
        call = tool_calls[0]
        if not isinstance(call, Mapping) or call.get("type") != "function":
            raise AgentRunError("model returned an unsupported tool call", status=400, code="unsupported_tool_call")
        function = call.get("function")
        if not isinstance(function, Mapping):
            raise AgentRunError("model tool call is missing function payload", status=400, code="invalid_tool_call")
        name = function.get("name")
        if not isinstance(name, str) or name not in self.tools_by_name:
            raise AgentRunError(f"model requested unknown brokered tool {name!r}", status=400, code="unknown_brokered_tool")
        raw_arguments = function.get("arguments", "{}")
        if isinstance(raw_arguments, str):
            try:
                if len(raw_arguments) > self.max_argument_bytes or len(raw_arguments.encode("utf-8")) > self.max_argument_bytes:
                    raise AgentRunError(
                        "model brokered tool arguments are too large",
                        status=413,
                        code="brokered_arguments_too_large",
                    )
            except UnicodeEncodeError as exc:
                raise AgentRunError("model tool arguments must contain valid Unicode", status=400, code="InvalidToolArguments") from exc
        arguments = _parse_arguments(raw_arguments)
        _validate_json_unicode(arguments)
        argument_text = json.dumps(arguments, separators=(",", ":"), sort_keys=True)
        assistant_message = {
            "role": "assistant",
            "content": message.get("content"),
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": argument_text},
                }
            ],
        }
        return ModelLoopToolRequest(name=name, arguments=arguments, messages=[*messages, assistant_message], usage=usage)

    async def resume(self, messages: Sequence[Mapping[str, Any]], *, call_id: str, output: str) -> ModelLoopFinal:
        if len(output.encode("utf-8")) > self.max_output_bytes:
            raise AgentRunError("brokered tool output is too large for model resume", status=413, code="brokered_output_too_large")
        resumed = [dict(message) for message in messages]
        resumed.append({"role": "tool", "tool_call_id": call_id, "content": output})
        data = await self._chat(resumed, tools=[])
        message = _choice_message(data)
        if message.get("tool_calls"):
            raise AgentRunError("model requested another brokered tool after resume", status=400, code="tool_loop_limit_exceeded")
        return ModelLoopFinal(text=_message_text(message), usage=_usage(data))

    def _initial_messages(self, request: RunRequest) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if self.spec.instructions:
            messages.append({"role": "system", "content": self.spec.instructions})
        for turn in request.history:
            if turn.role in FORWARDED_ROLES and turn.text:
                messages.append({"role": turn.role, "content": turn.text})
        messages.append({"role": "user", "content": request.prompt})
        return messages

    def _tool_payloads(self) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for tool in self.tools:
            description = f"Brokered class: {tool.brokered_class}. {tool.description}".strip()
            payloads.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": description,
                        "parameters": dict(tool.parameters),
                    },
                }
            )
        return payloads

    async def _chat(self, messages: Sequence[Mapping[str, Any]], *, tools: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.spec.model.name, "messages": list(messages)}
        if tools:
            payload["tools"] = list(tools)
            payload["tool_choice"] = "auto"
        client = self.http_client
        close_client = False
        if client is None:
            headers: dict[str, str] = {}
            try:
                auth = self.spec.model.auth
                if auth is not None and auth.type == "workload-identity-token":
                    token = os.environ.get("AGENTKIT_MODEL_WORKLOAD_IDENTITY_TOKEN")
                    if not token:
                        token = await asyncio.to_thread(resolve_workload_identity_token, auth.audience or "")
                    headers["Authorization"] = f"Bearer {token}"
                else:
                    api_key = resolve_api_key(self.spec)
                    headers["Authorization"] = f"Bearer {api_key}"
            except AgentBuildError as exc:
                raise AgentRunError(str(exc), status=400, code="ModelAuthMissing") from exc
            client = httpx.AsyncClient(headers=headers, timeout=60)
            close_client = True
        try:
            response = await client.post(_chat_completions_url(self.spec.model.base_url), json=payload)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            raise AgentRunError(str(exc), status=exc.response.status_code, code="ModelHTTPError") from exc
        except Exception as exc:  # noqa: BLE001 - normalize transport/model failures.
            raise AgentRunError(str(exc), status=502, code=exc.__class__.__name__) from exc
        finally:
            if close_client:
                await client.aclose()
        if not isinstance(data, dict):
            raise AgentRunError("model response must be a JSON object", status=502, code="InvalidModelResponse")
        return data


def _chat_completions_url(base_url: str) -> str:
    root = base_url.rstrip("/")
    if root.endswith("/chat/completions"):
        return root
    return f"{root}/chat/completions"


def _choice_message(data: Mapping[str, Any]) -> Mapping[str, Any]:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise AgentRunError("model response did not include choices", status=502, code="InvalidModelResponse")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise AgentRunError("model response choice must be an object", status=502, code="InvalidModelResponse")
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise AgentRunError("model response choice did not include a message", status=502, code="InvalidModelResponse")
    return message


def _message_text(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    refusal = message.get("refusal")
    if content is None and isinstance(refusal, str):
        return refusal
    raise AgentRunError(
        "model response final assistant content must be a string",
        status=502,
        code="InvalidModelResponse",
    )


def _parse_arguments(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, str):
        raise AgentRunError("model tool arguments must be a JSON object string", status=400, code="InvalidToolArguments")
    try:
        parsed = json.loads(
            raw or "{}",
            parse_float=_parse_json_float,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_argument_keys,
        )
    except AgentRunError:
        raise
    except (json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise AgentRunError("model tool arguments must be valid JSON", status=400, code="InvalidToolArguments") from exc
    if not isinstance(parsed, dict):
        raise AgentRunError("model tool arguments must be a JSON object", status=400, code="InvalidToolArguments")
    return parsed


def _reject_duplicate_argument_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise AgentRunError(f"model tool arguments contain duplicate key {key!r}", status=400, code="InvalidToolArguments")
        out[key] = value
    return out


def _validate_json_unicode(value: Any, *, path: str = "arguments") -> None:
    pending: list[tuple[Any, str, int]] = [(value, path, 0)]
    while pending:
        current, current_path, depth = pending.pop()
        if depth > _MAX_ARGUMENT_DEPTH:
            raise AgentRunError("model tool arguments are nested too deeply", status=400, code="InvalidToolArguments")
        if isinstance(current, str):
            try:
                current.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise AgentRunError(f"model tool arguments contain invalid Unicode at {current_path}", status=400, code="InvalidToolArguments") from exc
        elif isinstance(current, Mapping):
            for key, child in current.items():
                pending.append((child, f"{current_path}[{key!r}]", depth + 1))
                pending.append((key, f"{current_path}.<key>", depth + 1))
        elif isinstance(current, list):
            for index, child in enumerate(current):
                pending.append((child, f"{current_path}[{index}]", depth + 1))


def _parse_json_float(raw: str) -> float:
    try:
        decimal = Decimal(raw)
    except InvalidOperation as exc:
        raise AgentRunError("model tool arguments must contain valid JSON numbers", status=400, code="InvalidToolArguments") from exc
    parsed = float(decimal)
    if not Decimal(str(parsed)) == decimal:
        raise AgentRunError(
            "model tool arguments contain a number that cannot be represented exactly",
            status=400,
            code="InvalidToolArguments",
        )
    return parsed


def _reject_json_constant(raw: str) -> None:
    raise AgentRunError(f"model tool arguments contain non-finite number {raw}", status=400, code="InvalidToolArguments")


def _usage_token_count(value: Any) -> int:
    if value is None or isinstance(value, bool):
        if value is None:
            return 0
        raise AgentRunError(
            "model response usage must contain non-negative integer token counts",
            status=502,
            code="InvalidModelResponse",
        )
    if isinstance(value, float) and (not math.isfinite(value) or not value.is_integer()):
        raise AgentRunError(
            "model response usage must contain non-negative integer token counts",
            status=502,
            code="InvalidModelResponse",
        )
    try:
        count = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AgentRunError(
            "model response usage must contain non-negative integer token counts",
            status=502,
            code="InvalidModelResponse",
        ) from exc
    if count < 0:
        raise AgentRunError(
            "model response usage must contain non-negative integer token counts",
            status=502,
            code="InvalidModelResponse",
        )
    return count


def _usage(data: Mapping[str, Any]) -> dict[str, int]:
    usage = data.get("usage") if isinstance(data.get("usage"), Mapping) else {}
    prompt_count = _usage_token_count(usage.get("prompt_tokens", usage.get("input_tokens", 0)))
    completion_count = _usage_token_count(usage.get("completion_tokens", usage.get("output_tokens", 0)))
    total_count = _usage_token_count(usage.get("total_tokens", prompt_count + completion_count))
    return {"prompt_tokens": prompt_count, "completion_tokens": completion_count, "total_tokens": total_count}


__all__ = ["BrokeredChatModelLoop", "ModelLoopFinal", "ModelLoopToolRequest"]
