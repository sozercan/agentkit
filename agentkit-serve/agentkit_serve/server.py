"""FastAPI app: a NON-STREAMING OpenAI Chat-Completions facade over a pydantic-ai agent.

Endpoints (see ``docs/agent-abi.md`` §4):

* ``POST /v1/chat/completions`` — runs the agent once and returns a single
  ``chat.completion`` object. Rejects ``stream: true`` and request-supplied
  ``tools`` / ``tool_choice`` (the agent owns its tools).
* ``GET  /v1/models``           — SDK-compatibility listing of the one model.
* ``GET  /healthz``             — liveness (always open, even under auth).

The agent's MCP subprocesses are started ONCE in the lifespan (``async with
agent:``) and reused across requests, then torn down on shutdown.

Secret hygiene: this module never logs the request body, the API key, or any tool
env value.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from .agent_factory import build_agent
from .config import AgentSpec

# pydantic-ai surfaces upstream HTTP failures as ModelHTTPError (has .status_code).
try:  # pragma: no cover - import shape guard across pydantic-ai versions
    from pydantic_ai.exceptions import ModelHTTPError
except Exception:  # pragma: no cover
    ModelHTTPError = None  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# Request models (lenient: ignore unknown OpenAI fields like name/logit_bias).
# --------------------------------------------------------------------------- #
class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str
    # content is a string, a list of content parts, or null (tool-call turns).
    content: Any = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str | None = None
    messages: list[ChatMessage] = []
    stream: bool | None = False
    tools: list[Any] | None = None
    tool_choice: Any = None


# tool_choice values that mean "no specific tool requested" — treated as absent so
# generic OpenAI SDK clients that always attach one are not rejected outright.
_EMPTY_TOOL_CHOICE = (None, "", "none", "auto")


def _text_of(content: Any) -> str:
    """Flatten OpenAI message content into plain text.

    Accepts a string, a list of content parts (``{"type":"text","text":...}``),
    or ``None``. Non-text parts (images, etc.) are ignored in v0.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[str] = []
        for part in content:
            if isinstance(part, str):
                out.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                out.append(str(part.get("text", "")))
        return "".join(out)
    return str(content)


def _split_conversation(messages: list[ChatMessage]) -> tuple[list, str]:
    """Map an OpenAI message list to (pydantic-ai message_history, final prompt).

    Contract: the conversation must end with a ``user`` message; its text is the
    prompt for this turn. Earlier messages become history. ``tool`` messages are
    dropped — the agent owns its tools, so client-supplied tool results are not
    meaningful in v0.
    """
    # Imported lazily so config-only consumers don't pull the messages module.
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        SystemPromptPart,
        TextPart,
        UserPromptPart,
    )

    if not messages:
        raise HTTPException(status_code=400, detail="messages must be a non-empty array")

    last = messages[-1]
    if last.role != "user":
        raise HTTPException(
            status_code=400,
            detail="the final message must have role 'user'",
        )
    prompt = _text_of(last.content)

    history: list = []
    for msg in messages[:-1]:
        text = _text_of(msg.content)
        if text == "":
            continue
        if msg.role == "user":
            history.append(ModelRequest(parts=[UserPromptPart(content=text)]))
        elif msg.role == "system":
            history.append(ModelRequest(parts=[SystemPromptPart(content=text)]))
        elif msg.role == "assistant":
            history.append(ModelResponse(parts=[TextPart(content=text)]))
        # 'tool' and any other role: skipped by design.
    return history, prompt


def _usage_block(result: Any) -> dict[str, int]:
    """Best-effort OpenAI usage block from the pydantic-ai run result (zeros if unknown)."""
    try:
        usage = result.usage
        # In current pydantic-ai ``usage`` is a property returning a RunUsage; in
        # older builds it was a method. Prefer the property value; only call it if
        # we got a bare callable WITHOUT the token attributes (the real method).
        if not hasattr(usage, "input_tokens") and callable(usage):
            usage = usage()
        prompt_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    except Exception:
        prompt_tokens = completion_tokens = 0
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _completion_response(model_name: str, output: str, result: Any) -> dict:
    """Assemble a single OpenAI ``chat.completion`` object."""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": output},
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
        "usage": _usage_block(result),
    }


def _error_response(status: int, message: str, err_type: str, code: str | None = None) -> JSONResponse:
    """OpenAI-shaped error envelope."""
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": err_type, "code": code}},
    )


def make_auth_dependency(auth_token: str | None):
    """Build a FastAPI dependency enforcing ``Authorization: Bearer <token>``.

    When ``auth_token`` is falsy, auth is disabled and the dependency is a no-op.
    Applied to ``/v1/*`` only; ``/healthz`` stays open.
    """

    async def _require_auth(authorization: str | None = Header(default=None)) -> None:
        if not auth_token:
            return
        expected = f"Bearer {auth_token}"
        # Constant-ish comparison; tokens are short-lived deploy secrets.
        if authorization != expected:
            raise HTTPException(
                status_code=401,
                detail="missing or invalid bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    return _require_auth


def create_app(spec: AgentSpec, auth_token: str | None = None) -> FastAPI:
    """Construct the FastAPI app for one validated :class:`AgentSpec`."""
    agent = build_agent(spec)
    model_name = spec.model.name

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Enter the agent context ONCE: starts the stdio MCP subprocesses and keeps
        # them warm for the life of the server. Torn down on shutdown.
        async with agent:
            app.state.agent = agent
            yield

    app = FastAPI(title="agentkit-serve", lifespan=lifespan)
    auth = Depends(make_auth_dependency(auth_token))

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models", dependencies=[auth])
    async def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "agentkit",
                }
            ],
        }

    @app.post("/v1/chat/completions", dependencies=[auth])
    async def chat_completions(req: ChatCompletionRequest, request: Request):
        # --- reject unsupported request features (ABI §4) -------------------
        if req.stream:
            return _error_response(
                400,
                "streaming is not supported; this agent serves non-streaming "
                "chat completions only",
                "invalid_request_error",
                "stream_unsupported",
            )
        if req.tools:
            return _error_response(
                400,
                "request-supplied tools are not allowed; this agent owns its tools",
                "invalid_request_error",
                "tools_unsupported",
            )
        if req.tool_choice not in _EMPTY_TOOL_CHOICE:
            return _error_response(
                400,
                "request-supplied tool_choice is not allowed; this agent owns its tools",
                "invalid_request_error",
                "tool_choice_unsupported",
            )

        # --- map conversation & run the agent ------------------------------
        history, prompt = _split_conversation(req.messages)
        run_agent = request.app.state.agent
        try:
            result = await run_agent.run(prompt, message_history=history)
        except HTTPException:
            raise
        except Exception as exc:  # map runtime/model errors to OpenAI envelope
            status = 502
            if ModelHTTPError is not None and isinstance(exc, ModelHTTPError):
                status = getattr(exc, "status_code", 502) or 502
            return _error_response(
                status,
                f"agent run failed: {exc}",
                "agent_error",
                exc.__class__.__name__,
            )

        output = result.output if isinstance(result.output, str) else str(result.output)
        return _completion_response(model_name, output, result)

    # Map HTTPExceptions raised in helpers to the OpenAI error envelope too.
    @app.exception_handler(HTTPException)
    async def _http_exc_handler(request: Request, exc: HTTPException):
        return _error_response(
            exc.status_code,
            str(exc.detail),
            "invalid_request_error" if exc.status_code < 500 else "internal_error",
        )

    return app
