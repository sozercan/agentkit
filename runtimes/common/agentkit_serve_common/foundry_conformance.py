"""Tiny Azure Responses SDK conformance app for Foundry hosted brokered spikes.

This module is intentionally separate from the production Foundry adapter. It is
used to prove the hosted Responses lifecycle can carry a deterministic
function_call/function_call_output loop with SDK-assigned response IDs and state.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Sequence

from starlette.responses import JSONResponse
from starlette.routing import Route
from azure.ai.agentserver.core._request_id import REQUEST_ID_STATE_KEY

from azure.ai.agentserver.responses import (
    InMemoryResponseProvider,
    ResponseEventStream,
    ResponsesAgentServerHost,
    get_input_expanded,
)

_CONFORMANCE_CALL_ID = "call_conformance_1"
_CONFORMANCE_TOOL_NAME = "conformance_read"
_CONFORMANCE_ARGUMENTS = '{"probe":true}'


class _ForceNonStoredResponsesMiddleware:
    def __init__(self, app, *, max_body_bytes: int, session_id: str) -> None:  # noqa: ANN001
        self.app = app
        self.max_body_bytes = max(int(max_body_bytes), 1)
        self.session_id = session_id

    async def __call__(self, scope, receive, send) -> None:  # noqa: ANN001
        if scope.get("type") != "http" or scope.get("method") != "POST" or scope.get("path", "").rstrip("/") != "/responses":
            await self.app(scope, receive, send)
            return

        chunks: list[bytes] = []
        total_bytes = 0
        more_body = True
        while more_body:
            message = await receive()
            if message.get("type") != "http.request":
                await self.app(scope, receive, send)
                return
            chunk = message.get("body", b"")
            total_bytes += len(chunk)
            if total_bytes > self.max_body_bytes:
                await self._send_error(scope, send, status=413, code="request_body_too_large", message="request body too large")
                return
            chunks.append(chunk)
            more_body = bool(message.get("more_body"))
        body = b"".join(chunks)
        try:
            payload = json.loads(body)
            if isinstance(payload, dict):
                if payload.get("background") is True:
                    await self._send_error(
                        scope,
                        send,
                        status=400,
                        code="background_unsupported",
                        message="background responses are not supported by the conformance fixture",
                    )
                    return
                if "store" in payload and payload["store"] is not None and not isinstance(payload["store"], bool):
                    rewritten = body
                else:
                    payload["store"] = False
                    rewritten = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            else:
                rewritten = body
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError, TypeError, RecursionError):
            rewritten = body

        sent = False
        original_receive = receive

        async def replay_receive():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": rewritten, "more_body": False}
            return await original_receive()

        rewritten_scope = dict(scope)
        headers = [(name, value) for name, value in scope.get("headers", []) if name.lower() != b"content-length"]
        headers.append((b"content-length", str(len(rewritten)).encode("ascii")))
        rewritten_scope["headers"] = headers
        await self.app(rewritten_scope, replay_receive, send)

    async def _send_error(self, scope, send, *, status: int, code: str, message: str) -> None:  # noqa: ANN001
        state = scope.get("state") if isinstance(scope.get("state"), dict) else {}
        request_id = str(state.get(REQUEST_ID_STATE_KEY) or "")
        error: dict[str, Any] = {"code": code, "message": message, "type": "invalid_request_error"}
        if request_id:
            error["additionalInfo"] = {"request_id": request_id}
        body = json.dumps({"error": error}, separators=(",", ":")).encode("utf-8")
        response_headers = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("ascii")),
            (b"x-platform-error-source", b"user"),
        ]
        if self.session_id:
            response_headers.append((b"x-agent-session-id", self.session_id.encode("utf-8")))
        await send({"type": "http.response.start", "status": status, "headers": response_headers})
        await send({"type": "http.response.body", "body": body})



def _input_items(request: Any) -> list[dict[str, Any]]:
    return [dict(item) for item in get_input_expanded(request)]


def _function_call_outputs(request: Any) -> list[dict[str, Any]]:
    return [item for item in _input_items(request) if item.get("type") == "function_call_output"]


def _request_tools(request: Any) -> Any:
    tools = getattr(request, "tools", None)
    if tools is None and hasattr(request, "get"):
        tools = request.get("tools")
    return tools


def create_foundry_conformance_app(
    *,
    model: str = "agentkit-foundry-conformance",
    pending_ttl_seconds: float = 15 * 60,
    max_pending_responses: int = 128,
    max_request_body_bytes: int = 1024 * 1024,
) -> ResponsesAgentServerHost:
    """Create a minimal Responses SDK app for A0 Foundry function-call smokes."""

    pending: dict[str, tuple[set[str], float]] = {}
    store = InMemoryResponseProvider()
    app = ResponsesAgentServerHost(store=store, configure_observability=None)
    app.add_middleware(
        _ForceNonStoredResponsesMiddleware,
        max_body_bytes=max_request_body_bytes,
        session_id=os.environ.get("FOUNDRY_AGENT_SESSION_ID", "").strip(),
    )
    app.user_middleware.append(app.user_middleware.pop(0))
    app.state.conformance_store = store

    def purge_expired() -> None:
        now = time.time()
        for response_id in [response_id for response_id, (_calls, expires_at) in pending.items() if expires_at <= now]:
            pending.pop(response_id, None)

    async def readiness(_request):  # noqa: ANN001 - Starlette passes Request.
        return JSONResponse(
            {
                "ready": True,
                "protocols": {"responses": "2.0.0"},
                "implementation": "azure-ai-agentserver-responses",
                "pendingStateTtlSeconds": pending_ttl_seconds,
                "pendingStateMax": max_pending_responses,
                "requestBodyMaxBytes": max_request_body_bytes,
                "background": False,
            }
        )

    app.router.routes.insert(0, Route("/readiness", readiness, methods=["GET"]))

    @app.response_handler
    async def response_handler(request, context, cancellation_signal):  # noqa: ANN001 - SDK-defined handler types.
        stream = ResponseEventStream(response_id=context.response_id, model=model, request=request)
        yield stream.emit_created()
        yield stream.emit_in_progress()

        if _request_tools(request):
            yield stream.emit_failed(
                code="tools_unsupported",
                message="request-level tools are not allowed for hosted brokered conformance",
            )
            return

        purge_expired()
        outputs = _function_call_outputs(request)
        if outputs:
            previous_response_id = getattr(request, "previous_response_id", None)
            if not previous_response_id:
                yield stream.emit_failed(
                    code="missing_previous_response_id",
                    message="function_call_output requires previous_response_id",
                )
                return
            pending_state = pending.get(str(previous_response_id))
            if pending_state is None:
                yield stream.emit_failed(
                    code="unknown_previous_response_id",
                    message="unknown previous_response_id",
                )
                return
            pending_calls, _expires_at = pending_state
            if len(outputs) != 1:
                yield stream.emit_failed(
                    code="multiple_tool_outputs_unsupported",
                    message="multiple function_call_output items are not supported",
                )
                return
            call_id = outputs[0].get("call_id")
            if call_id not in pending_calls:
                yield stream.emit_failed(code="unknown_call_id", message="unknown function_call_output call_id")
                return
            pending.pop(str(previous_response_id), None)
            output = outputs[0].get("output", "")
            try:
                parsed = json.loads(output) if isinstance(output, str) else output
            except json.JSONDecodeError:
                parsed = output
            for event in stream.output_item_message(f"conformance complete: {json.dumps(parsed, sort_keys=True)}"):
                yield event
            yield stream.emit_completed()
            return

        if len(pending) >= max_pending_responses:
            yield stream.emit_failed(
                code="brokered_response_state_full",
                message="too many pending brokered conformance responses",
            )
            return
        pending[context.response_id] = ({_CONFORMANCE_CALL_ID}, time.time() + max(pending_ttl_seconds, 0.0))
        for event in stream.output_item_function_call(
            name=_CONFORMANCE_TOOL_NAME,
            call_id=_CONFORMANCE_CALL_ID,
            arguments=_CONFORMANCE_ARGUMENTS,
        ):
            yield event
        yield stream.emit_completed()

    return app


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="agentkit-foundry-conformance",
        description="Run the tiny Azure Responses SDK conformance app for Foundry brokered A0 smokes.",
    )
    parser.add_argument("--host", default=os.environ.get("AGENTKIT_BIND", "0.0.0.0"), help="host interface to bind")
    parser.add_argument("--port", type=int, default=int(os.environ.get("AGENTKIT_PORT", os.environ.get("PORT", "8088"))), help="port to bind")
    parser.add_argument("--model", default=os.environ.get("AGENTKIT_FOUNDRY_CONFORMANCE_MODEL", "agentkit-foundry-conformance"), help="model name to stamp in response envelopes")
    parser.add_argument("--dry-run", action="store_true", help="validate arguments and print the selected bind/model without serving")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    app = create_foundry_conformance_app(model=args.model)
    if args.dry_run:
        print(json.dumps({"host": args.host, "port": args.port, "model": args.model, "protocols": {"responses": "2.0.0"}}, sort_keys=True))
        return 0
    app.run(host=args.host, port=args.port)
    return 0


__all__ = ["create_foundry_conformance_app", "main"]


if __name__ == "__main__":  # pragma: no cover - exercised by console script/dry-run tests.
    raise SystemExit(main())
