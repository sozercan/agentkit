#!/usr/bin/env python3
"""Verify a Foundry brokered Responses conformance transcript.

The transcript is produced by deploy/foundry/scripts/foundry_brokered_conformance.sh
and is intentionally token-free: it contains request/response JSON only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

EXPECTED_FILES = (
    "01-initial-request.json",
    "02-initial-response.json",
    "03-continuation-request.json",
    "04-continuation-response.json",
)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing transcript file: {path.name}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path.name} is not valid JSON: {exc}") from exc


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _message_text(response: dict[str, Any]) -> str:
    output = response.get("output")
    _require(isinstance(output, list) and bool(output), "final response output must be a non-empty array")
    message = output[0]
    _require(isinstance(message, dict) and message.get("type") == "message", "final response output[0] must be a message")
    content = message.get("content")
    _require(isinstance(content, list) and bool(content), "final message content must be a non-empty array")
    text = content[0].get("text") if isinstance(content[0], dict) else None
    _require(isinstance(text, str) and bool(text), "final message must contain text")
    return text


def verify_transcript(
    transcript_dir: str | Path,
    *,
    expected_tool_name: str = "conformance_read",
    expected_arguments_json: str = '{"probe":true}',
    expected_call_id: str = "call_conformance_1",
    expected_call_id_prefix: str | None = None,
) -> dict[str, Any]:
    root = Path(transcript_dir)
    expected_arguments = json.loads(expected_arguments_json)
    initial_request = _load_json(root / "01-initial-request.json")
    initial_response = _load_json(root / "02-initial-response.json")
    continuation_request = _load_json(root / "03-continuation-request.json")
    continuation_response = _load_json(root / "04-continuation-response.json")

    _require(isinstance(initial_request, dict), "initial request must be a JSON object")
    _require("tools" not in initial_request, "initial request must not contain request-level tools")
    _require("input" in initial_request, "initial request must contain input")

    _require(isinstance(initial_response, dict), "initial response must be a JSON object")
    _require(initial_response.get("status") == "completed", "initial response status must be completed")
    initial_response_id = initial_response.get("id")
    _require(isinstance(initial_response_id, str) and initial_response_id.startswith("caresp_"), "initial response id must start with caresp_")
    _require(not initial_response_id.startswith("resp_"), "initial response id must not use old resp_ format")
    output = initial_response.get("output")
    _require(isinstance(output, list) and len(output) == 1, "initial response output must contain exactly one item")
    call = output[0]
    _require(isinstance(call, dict), "initial response output[0] must be an object")
    _require(call.get("type") == "function_call", "initial output item must be function_call")
    function_name = call.get("name")
    _require(function_name == expected_tool_name, f"function_call name must be {expected_tool_name}")
    call_id = call.get("call_id")
    _require(isinstance(call_id, str) and bool(call_id), "function_call call_id must be a non-empty string")
    if expected_call_id != "auto":
        _require(call_id == expected_call_id, f"function_call call_id must be {expected_call_id}")
    if expected_call_id_prefix:
        _require(call_id.startswith(expected_call_id_prefix), f"function_call call_id must start with {expected_call_id_prefix}")
    arguments = call.get("arguments")
    _require(isinstance(arguments, str), "function_call arguments must be a JSON string")
    parsed_arguments = json.loads(arguments)
    _require(parsed_arguments == expected_arguments, f"function_call arguments must be {expected_arguments}")

    _require(isinstance(continuation_request, dict), "continuation request must be a JSON object")
    _require(continuation_request.get("previous_response_id") == initial_response_id, "continuation previous_response_id must match initial id")
    continuation_input = continuation_request.get("input")
    _require(isinstance(continuation_input, list) and len(continuation_input) == 1, "continuation input must contain exactly one item")
    continuation_item = continuation_input[0]
    _require(isinstance(continuation_item, dict), "continuation item must be an object")
    _require(continuation_item.get("type") == "function_call_output", "continuation item must be function_call_output")
    _require(continuation_item.get("call_id") == call_id, "continuation call_id must match function_call call_id")
    continuation_output = continuation_item.get("output")
    _require(isinstance(continuation_output, str), "continuation output must be a JSON string")
    parsed_output = json.loads(continuation_output)
    _require(isinstance(parsed_output, dict), "continuation output JSON must be an object")

    _require(isinstance(continuation_response, dict), "continuation response must be a JSON object")
    _require(continuation_response.get("status") == "completed", "continuation response status must be completed")
    _require(continuation_response.get("previous_response_id") == initial_response_id, "continuation response previous_response_id must match initial id")
    continuation_response_id = continuation_response.get("id")
    _require(isinstance(continuation_response_id, str) and continuation_response_id.startswith("caresp_"), "continuation response id must start with caresp_")
    _require(continuation_response_id != initial_response_id, "continuation response id must differ from initial response id")
    final_text = _message_text(continuation_response)

    return {
        "initial_response_id": initial_response_id,
        "continuation_response_id": continuation_response_id,
        "function_call_name": function_name,
        "call_id": call_id,
        "arguments": parsed_arguments,
        "final_text": final_text,
        "transcript_files": list(EXPECTED_FILES),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify a Foundry brokered conformance transcript directory.")
    parser.add_argument("transcript_dir", help="directory containing 01/02/03/04 conformance transcript JSON files")
    parser.add_argument("--expected-tool-name", default="conformance_read", help="expected function_call name")
    parser.add_argument("--expected-arguments-json", default='{"probe":true}', help="expected function_call arguments JSON")
    parser.add_argument("--expected-call-id", default="call_conformance_1", help="expected call_id, or 'auto' to only require a non-empty id")
    parser.add_argument("--expected-call-id-prefix", default=None, help="optional required call_id prefix")
    parser.add_argument("--write-summary", action="store_true", help="write summary.json in the transcript directory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        summary = verify_transcript(
            args.transcript_dir,
            expected_tool_name=args.expected_tool_name,
            expected_arguments_json=args.expected_arguments_json,
            expected_call_id=args.expected_call_id,
            expected_call_id_prefix=args.expected_call_id_prefix,
        )
        if args.write_summary:
            Path(args.transcript_dir, "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
    except Exception as exc:  # noqa: BLE001 - CLI verifier should print concise evidence failures.
        print(f"verify_brokered_transcript: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
