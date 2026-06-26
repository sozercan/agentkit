"""Foundry Hosted Agent invocations smoke wrapper for AgentKit.

The wrapper exposes the Foundry Hosted Agents invocations protocol on port 8088,
then routes each JSON ``{"message": ...}`` request into the baked AgentKit
runtime. A deterministic in-process OpenAI-compatible mock model keeps the smoke
test independent of external model credentials while still exercising AgentKit's
OpenAI-compatible model client path.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from azure.ai.agentserver.invocations import InvocationAgentServerHost
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from agentkit_serve.agent_factory import build_runtime
from agentkit_serve_common.config import load, validate_required_env
from agentkit_serve_common.conversation import RunRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agentkit-foundry-wrapper")

_SENTINEL = "DONE_FOUNDRY_AGENTKIT_123"


def _start_mock_openai() -> None:
    """Start a deterministic OpenAI-compatible mock model on localhost:9000."""

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/v1/models":
                return self._send(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": "mock-gpt",
                                "object": "model",
                                "created": 0,
                                "owned_by": "agentkit-foundry-wrapper",
                            }
                        ],
                    },
                )
            return self._send(404, {"error": {"message": "not found"}})

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("content-length", "0") or "0")
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw)
            except json.JSONDecodeError:
                body = {}

            if self.path == "/v1/chat/completions":
                user_text = ""
                for msg in body.get("messages", []):
                    if msg.get("role") == "user":
                        user_text = str(msg.get("content", ""))
                return self._send(
                    200,
                    {
                        "id": "chatcmpl-agentkit-foundry-mock",
                        "object": "chat.completion",
                        "created": 1770000000,
                        "model": body.get("model", "mock-gpt"),
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": f"AgentKit Foundry smoke OK: {user_text} {_SENTINEL}",
                                },
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 5,
                            "completion_tokens": 7,
                            "total_tokens": 12,
                        },
                    },
                )
            return self._send(404, {"error": {"message": "not found"}})

        def log_message(self, fmt: str, *args: Any) -> None:
            logger.info("mock-openai " + fmt, *args)

    server = ThreadingHTTPServer(("127.0.0.1", 9000), Handler)
    thread = threading.Thread(target=server.serve_forever, name="mock-openai", daemon=True)
    thread.start()
    logger.info("mock OpenAI-compatible endpoint listening on 127.0.0.1:9000")


os.environ.setdefault("MODEL_API_KEY", "not-needed")
_start_mock_openai()

spec = load("/agent/agent.yaml")
validate_required_env(spec)
runtime = build_runtime(spec)
app = InvocationAgentServerHost()
_original_lifespan = app.router.lifespan_context


@asynccontextmanager
async def _lifespan_with_agent(starlette_app):
    # Match agentkit_serve_common.server.create_app: enter the runtime session
    # once for the server lifetime so MCP tool subprocesses are started and kept
    # warm before any invocation is handled.
    async with _original_lifespan(starlette_app):
        async with runtime:
            starlette_app.state.runtime = runtime
            yield


app.router.lifespan_context = _lifespan_with_agent


@app.invoke_handler
async def handle_invoke(request: Request):
    try:
        data = await request.json()
    except json.JSONDecodeError:
        return Response("Request body must be JSON", status_code=400)

    message = data.get("message")
    if message is None:
        return Response("Missing 'message' in request", status_code=400)
    if not isinstance(message, str):
        message = json.dumps(message, separators=(",", ":"), sort_keys=True)

    try:
        result = await request.app.state.runtime.run(RunRequest(prompt=message))
    except Exception as exc:  # noqa: BLE001 - expose smoke-test failure clearly.
        logger.exception("AgentKit run failed")
        return JSONResponse({"error": str(exc)}, status_code=502)

    return JSONResponse({"response": result.text, "usage": result.usage})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8088")))
