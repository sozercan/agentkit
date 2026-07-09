"""Tiny Azure Responses SDK conformance app for Foundry hosted brokered spikes.

This module is intentionally separate from the production Foundry adapter. It is
used to prove the hosted Responses lifecycle can carry a deterministic
function_call/function_call_output loop with SDK-assigned response IDs and state.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Sequence

from starlette.responses import JSONResponse
from starlette.routing import Route

from azure.ai.agentserver.responses import (
    InMemoryResponseProvider,
    ResponseEventStream,
    ResponsesAgentServerHost,
    get_input_expanded,
)

_CONFORMANCE_CALL_ID = "call_conformance_1"
_CONFORMANCE_TOOL_NAME = "conformance_read"
_CONFORMANCE_ARGUMENTS = '{"probe":true}'


def _input_items(request: Any) -> list[dict[str, Any]]:
    return [dict(item) for item in get_input_expanded(request)]


def _function_call_outputs(request: Any) -> list[dict[str, Any]]:
    return [item for item in _input_items(request) if item.get("type") == "function_call_output"]


def _request_tools(request: Any) -> Any:
    tools = getattr(request, "tools", None)
    if tools is None and hasattr(request, "get"):
        tools = request.get("tools")
    return tools


def create_foundry_conformance_app(*, model: str = "agentkit-foundry-conformance") -> ResponsesAgentServerHost:
    """Create a minimal Responses SDK app for A0 Foundry function-call smokes."""

    store = InMemoryResponseProvider()
    app = ResponsesAgentServerHost(store=store, configure_observability=None)
    pending: dict[str, set[str]] = {}

    async def readiness(_request):  # noqa: ANN001 - Starlette passes Request.
        return JSONResponse(
            {
                "ready": True,
                "protocols": {"responses": "2.0.0"},
                "implementation": "azure-ai-agentserver-responses",
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

        outputs = _function_call_outputs(request)
        if outputs:
            previous_response_id = getattr(request, "previous_response_id", None)
            if not previous_response_id:
                yield stream.emit_failed(
                    code="missing_previous_response_id",
                    message="function_call_output requires previous_response_id",
                )
                return
            pending_calls = pending.get(str(previous_response_id))
            if pending_calls is None:
                yield stream.emit_failed(
                    code="unknown_previous_response_id",
                    message="unknown previous_response_id",
                )
                return
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

        pending[context.response_id] = {_CONFORMANCE_CALL_ID}
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
