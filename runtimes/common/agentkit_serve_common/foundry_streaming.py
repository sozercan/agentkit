"""Early Responses acknowledgements for hosted AgentKit runtimes."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi.responses import JSONResponse, Response


class BrokeredResponseStream:
    def __init__(self, model: str) -> None:
        self.model = model
        self.session_id: str | None = None
        self.created: asyncio.Future[dict[str, Any]] = (
            asyncio.get_running_loop().create_future()
        )
        self.delivered = asyncio.Event()

    def prepare(self, response_id: str, *, created_at: int | None = None) -> None:
        if self.created.done():
            raise RuntimeError("hosted response was already acknowledged")
        payload: dict[str, Any] = {
            "id": response_id,
            "object": "response",
            "created_at": int(time.time()) if created_at is None else created_at,
            "status": "in_progress",
            "model": self.model,
            "output": [],
        }
        if self.session_id:
            payload["agent_session_id"] = self.session_id
        self.created.set_result(payload)

    async def accept(self, response_id: str) -> int:
        self.prepare(response_id)
        # Model work starts only after the response.created frame has been sent.
        await self.delivered.wait()
        return self.created.result()["created_at"]

    def terminal(self, result: JSONResponse | None) -> dict[str, Any]:
        created = self.created.result()
        payload = json.loads(result.body) if result is not None else {}
        if (
            result is not None
            and result.status_code < 400
            and payload.get("id") == created["id"]
        ):
            if self.session_id:
                payload["agent_session_id"] = self.session_id
            return {"type": "response.completed", "response": payload}
        error = (
            payload.get("error")
            if result is not None and result.status_code >= 400
            else None
        )
        if not isinstance(error, dict):
            error = {"code": "HostedResponseError", "message": "hosted response failed"}
        return {
            "type": "response.failed",
            "response": {**created, "status": "failed", "error": error},
        }


def _frame(event: dict[str, Any]) -> bytes:
    data = json.dumps(event, separators=(",", ":"), ensure_ascii=True)
    return f"event: {event['type']}\ndata: {data}\n\n".encode()


class _BrokeredStreamingResponse(Response):
    media_type = "text/event-stream"

    def __init__(
        self, stream: BrokeredResponseStream, operation: asyncio.Task[JSONResponse]
    ) -> None:
        super().__init__(
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        )
        self.raw_headers = [
            (key, value) for key, value in self.raw_headers if key != b"content-length"
        ]
        self.stream = stream
        self.operation = operation

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        async def write_response() -> None:
            await send(
                {
                    "type": "http.response.start",
                    "status": self.status_code,
                    "headers": self.raw_headers,
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": _frame(
                        {
                            "type": "response.created",
                            "response": self.stream.created.result(),
                        }
                    ),
                    "more_body": True,
                }
            )
            self.stream.delivered.set()
            try:
                result = await self.operation
            except Exception:  # noqa: BLE001 - never leak model exception text into SSE.
                result = None
            await send(
                {
                    "type": "http.response.body",
                    "body": _frame(self.stream.terminal(result)),
                    "more_body": False,
                }
            )

        async def wait_for_disconnect() -> None:
            while True:
                if (await receive())["type"] == "http.disconnect":
                    return

        writer = asyncio.create_task(write_response())
        disconnected = asyncio.create_task(wait_for_disconnect())
        try:
            done, _ = await asyncio.wait(
                {writer, disconnected}, return_when=asyncio.FIRST_COMPLETED
            )
            if writer in done:
                await writer
        finally:
            for task in (writer, disconnected, self.operation):
                if not task.done():
                    task.cancel()
            await asyncio.gather(
                writer, disconnected, self.operation, return_exceptions=True
            )


async def brokered_stream_response(
    model: str,
    operation: Callable[[BrokeredResponseStream], Awaitable[JSONResponse]],
) -> Response:
    stream = BrokeredResponseStream(model)
    task = asyncio.create_task(operation(stream))
    try:
        await asyncio.wait({task, stream.created}, return_when=asyncio.FIRST_COMPLETED)
        if not stream.created.done():
            # Rejected requests keep their HTTP error status. Deterministic tool
            # responses and completed replays need no model work or early wait.
            result = await task
            if result.status_code >= 400:
                return result
            payload = json.loads(result.body)
            stream.prepare(payload["id"], created_at=payload["created_at"])
        return _BrokeredStreamingResponse(stream, task)
    except BaseException:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        raise
