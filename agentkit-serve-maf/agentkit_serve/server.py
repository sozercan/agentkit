"""FastAPI app: a NON-STREAMING OpenAI Chat-Completions facade over a MAF agent.

Endpoints (see ``docs/agent-abi.md`` §4):

* ``POST /v1/chat/completions`` — runs the agent once and returns a single
  ``chat.completion`` object. Rejects ``stream: true`` and request-supplied
  ``tools`` / ``tool_choice`` (the agent owns its tools).
* ``GET  /v1/models``           — SDK-compatibility listing of the one model.
* ``GET  /healthz``             — liveness (always open, even under auth).

The agent's MCP subprocesses are started ONCE in the lifespan (``async with
agent:``) and reused across requests, then torn down on shutdown.

THIS MODULE IS FRAMEWORK-AGNOSTIC. It imports nothing from ``agent_framework``
(or any model SDK): the run is driven through ``agent_factory.run_agent``, which
returns a neutral :class:`~agentkit_serve.agent_factory.RunResult` and raises a
neutral :class:`~agentkit_serve.agent_factory.AgentRunError`. That is what makes
this file shareable across runtimes (packaging option B) and keeps the lock-in
import boundary (plan §12) trivially greppable — the boundary lives only in
``agent_factory.py``.

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

from .agent_factory import AgentRunError, RunResult, build_agent, run_agent
from .config import AgentSpec


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


def _split_conversation(messages: list[ChatMessage]) -> tuple[list[tuple[str, str]], str]:
    """Map an OpenAI message list to (neutral history, final prompt).

    Contract: the conversation must end with a ``user`` message; its text is the
    prompt for this turn. Earlier messages become history as neutral
    ``(role, text)`` tuples (framework-agnostic — the agent_factory maps them onto
    the runtime's message type). ``tool`` messages are dropped — the agent owns its
    tools, so client-supplied tool results are not meaningful in v0.
    """
    if not messages:
        raise HTTPException(status_code=400, detail="messages must be a non-empty array")

    last = messages[-1]
    if last.role != "user":
        raise HTTPException(
            status_code=400,
            detail="the final message must have role 'user'",
        )
    prompt = _text_of(last.content)

    history: list[tuple[str, str]] = []
    for msg in messages[:-1]:
        text = _text_of(msg.content)
        if text == "":
            continue
        if msg.role in ("user", "system", "assistant"):
            history.append((msg.role, text))
        # 'tool' and any other role: skipped by design.
    return history, prompt


def _completion_response(model_name: str, result: RunResult) -> dict:
    """Assemble a single OpenAI ``chat.completion`` object from a neutral result."""
    usage = result.usage or {}
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.text},
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        },
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
        agent = request.app.state.agent
        try:
            result = await run_agent(agent, prompt, history)
        except HTTPException:
            raise
        except AgentRunError as exc:  # neutral error: framework/model failure
            return _error_response(
                exc.status,
                str(exc),
                "agent_error",
                exc.__class__.__name__,
            )

        return _completion_response(model_name, result)

    # Map HTTPExceptions raised in helpers to the OpenAI error envelope too.
    @app.exception_handler(HTTPException)
    async def _http_exc_handler(request: Request, exc: HTTPException):
        return _error_response(
            exc.status_code,
            str(exc.detail),
            "invalid_request_error" if exc.status_code < 500 else "internal_error",
        )

    return app
