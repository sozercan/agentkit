"""Foundry Hosted Agent protocol smoke wrapper for AgentKit.

The wrapper exposes the Foundry Hosted Agents readiness, invocations, and minimal
non-streaming responses protocols on port 8088, then routes requests into the baked
AgentKit runtime. A deterministic in-process OpenAI-compatible mock model keeps the smoke
test independent of external model credentials while still exercising AgentKit's
OpenAI-compatible model client path.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import uvicorn

from agentkit_serve import agent_factory
from agentkit_serve_common.config import load, validate_required_env
from agentkit_serve_common.foundry import create_foundry_app

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
app = create_foundry_app(spec, agent_factory)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8088")))
