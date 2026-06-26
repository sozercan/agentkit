"""Foundry Hosted Agent protocol adapters over the AgentKit RuntimeSession seam.

The adapter intentionally stays provider-light: it exposes the container HTTP
contract expected by Foundry Hosted Agents (``/readiness``, ``/invocations`` and a
minimal non-streaming ``/responses``) while reusing the same ``RuntimeFactory`` /
``RunRequest`` seam as the native OpenAI facade.
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from .config import AgentSpec
from .conversation import FORWARDED_ROLES, ConversationTurn, RunRequest
from .runtime import AgentRunError, RuntimeFactory, RunResult


def _usage(result: RunResult) -> dict[str, int]:
    usage = result.usage or {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
    }


def _responses_usage(result: RunResult) -> dict[str, int]:
    usage = result.usage or {}
    input_tokens = int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
    output_tokens = int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0)
    total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens) or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
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


def _responses_payload(spec: AgentSpec, result: RunResult) -> dict[str, Any]:
    response_id = f"resp_{uuid.uuid4().hex}"
    message_id = f"msg_{uuid.uuid4().hex}"
    return {
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
            }
        ],
        "usage": _responses_usage(result),
    }


def create_foundry_app(spec: AgentSpec, factory: RuntimeFactory) -> FastAPI:
    """Create a Foundry-compatible wrapper app for one AgentKit runtime."""
    runtime = factory.build_runtime(spec)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with runtime:
            app.state.runtime = runtime
            yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/readiness")
    async def readiness():
        return {"ready": True}

    @app.post("/invocations")
    async def invocations(request: Request):
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

    @app.post("/responses")
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
                "request-supplied Responses tools are not allowed; this agent owns its tools",
                status=400,
                code="tools_unsupported",
            )
        if data.get("tool_choice") not in (None, "", "none", "auto"):
            return _error(
                "request-supplied Responses tool_choice is not allowed; this agent owns its tools",
                status=400,
                code="tool_choice_unsupported",
            )
        if "input" not in data:
            return _error("Missing 'input' in request", status=400, code="missing_input")

        try:
            run_request = _responses_input_to_run_request(
                data["input"],
                session_id=_session_id_from_request(request),
            )
        except ValueError as exc:
            return _error(str(exc), status=400, code="invalid_input")

        try:
            result = await request.app.state.runtime.run(run_request)
        except AgentRunError as exc:
            return _error(str(exc), status=exc.status, code=exc.code)
        except Exception as exc:  # noqa: BLE001 - deterministic protocol envelope.
            return _error(str(exc), status=502, code=exc.__class__.__name__)

        return JSONResponse(_responses_payload(spec, result))

    return app
