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
from .conversation import RunRequest
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


def _responses_input_to_prompt(value: Any) -> str:
    """Extract a prompt from the common non-streaming Responses API input shapes."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            content = item.get("content")
            if isinstance(content, str):
                parts.append(content)
                continue
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, str):
                        parts.append(block)
                    elif isinstance(block, dict):
                        text = block.get("text") or block.get("input_text")
                        if text is not None:
                            parts.append(str(text))
        if parts:
            return "\n".join(parts)
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


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
            result = await request.app.state.runtime.run(RunRequest(prompt=prompt))
        except AgentRunError as exc:
            return _error(str(exc), status=exc.status, code=exc.code)
        except Exception as exc:  # noqa: BLE001 - deterministic protocol envelope.
            return _error(str(exc), status=502, code=exc.__class__.__name__)

        return JSONResponse({"response": result.text, "usage": result.usage})

    @app.post("/responses")
    async def responses(request: Request):
        try:
            data = await request.json()
        except json.JSONDecodeError:
            return _error("Request body must be JSON", status=400, code="invalid_json")

        if not isinstance(data, dict):
            return _error("Request body must be a JSON object", status=400, code="invalid_request")
        if data.get("stream"):
            return _error(
                "streaming Responses are not supported by this AgentKit Foundry adapter",
                status=400,
                code="stream_unsupported",
            )
        if "input" not in data:
            return _error("Missing 'input' in request", status=400, code="missing_input")

        prompt = _responses_input_to_prompt(data["input"])
        try:
            result = await request.app.state.runtime.run(RunRequest(prompt=prompt))
        except AgentRunError as exc:
            return _error(str(exc), status=exc.status, code=exc.code)
        except Exception as exc:  # noqa: BLE001 - deterministic protocol envelope.
            return _error(str(exc), status=502, code=exc.__class__.__name__)

        return JSONResponse(_responses_payload(spec, result))

    return app
