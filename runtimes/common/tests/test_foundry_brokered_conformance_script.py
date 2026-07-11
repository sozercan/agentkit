from __future__ import annotations

import json
import os
import subprocess
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator


_REPO_ROOT = Path(__file__).resolve().parents[3]
_HELPER = _REPO_ROOT / "deploy" / "foundry" / "scripts" / "foundry_brokered_conformance.sh"


@contextmanager
def _mock_gateway(
    *,
    agent_session_id: str | None = None,
    echo_continuation_proof: bool = False,
    malformed_continuation_response: bool = False,
    serialize_continuation_body: bool = False,
) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    requests: list[dict[str, Any]] = []
    initial_response_id = "caresp_mock_initial"

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - stdlib HTTP handler API.
            content_length = int(self.headers.get("content-length", "0"))
            body = json.loads(self.rfile.read(content_length))
            requests.append({"headers": dict(self.headers), "body": body})

            if len(requests) == 1:
                response: dict[str, Any] = {
                    "id": initial_response_id,
                    "status": "completed",
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "call_conformance_1",
                            "name": "conformance_read",
                            "arguments": '{"probe":true}',
                            "status": "completed",
                        }
                    ],
                }
                if agent_session_id is not None:
                    response["agent_session_id"] = agent_session_id
            else:
                final_text = "success"
                if serialize_continuation_body:
                    final_text = "request failed: " + json.dumps(body, separators=(",", ":"))
                elif echo_continuation_proof:
                    final_text = str(body.get("brokered_continuation_proof", ""))
                response = {
                    "id": "caresp_mock_final",
                    "previous_response_id": initial_response_id,
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": final_text}],
                        }
                    ],
                }

            if len(requests) > 1 and malformed_continuation_response:
                encoded = ("not-json: " + json.dumps(body, separators=(",", ":"))).encode()
            else:
                encoded = json.dumps(response, separators=(",", ":")).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}/responses", requests
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _run_helper(endpoint: str, transcript_dir: Path, **environment: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    for name in (
        "AGENTKIT_CONFORMANCE_OUTPUT",
        "AGENTKIT_CONTINUATION_PROOF",
        "AGENTKIT_CONTINUATION_PROOF_BODY",
        "AGENTKIT_EXPECTED_ARGUMENTS",
        "AGENTKIT_EXPECTED_CALL_ID",
        "AGENTKIT_EXPECTED_CALL_ID_PREFIX",
        "AGENTKIT_EXPECTED_TOOL_NAME",
    ):
        env.pop(name, None)
    env.update(
        {
            "AGENT_RESPONSES_ENDPOINT": endpoint,
            "AGENT_RESPONSES_BEARER_TOKEN": "mock-token",
            **environment,
        }
    )
    return subprocess.run(
        [str(_HELPER), "conformance_read", str(transcript_dir)],
        cwd=_REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_brokered_conformance_helper_propagates_returned_agent_session_id(tmp_path: Path):
    transcript = tmp_path / "transcript"
    with _mock_gateway(agent_session_id="gateway-session") as (endpoint, requests):
        result = _run_helper(endpoint, transcript)

    assert result.returncode == 0, result.stderr
    assert len(requests) == 2
    assert requests[1]["body"]["agent_session_id"] == "gateway-session"
    archived = json.loads((transcript / "03-continuation-request.json").read_text(encoding="utf-8"))
    assert archived["agent_session_id"] == "gateway-session"


def test_brokered_conformance_helper_sends_body_proof_without_archiving_it(tmp_path: Path):
    transcript = tmp_path / "transcript"
    proof = "body-proof-must-stay-out-of-transcript"
    with _mock_gateway(agent_session_id="gateway-session") as (endpoint, requests):
        result = _run_helper(endpoint, transcript, AGENTKIT_CONTINUATION_PROOF_BODY=proof)

    assert result.returncode == 0, result.stderr
    assert len(requests) == 2
    assert requests[1]["body"]["agent_session_id"] == "gateway-session"
    assert requests[1]["body"]["brokered_continuation_proof"] == proof
    assert "x-agentkit-brokered-continuation-proof" not in requests[1]["headers"]
    archived_text = "\n".join(path.read_text(encoding="utf-8") for path in transcript.iterdir() if path.is_file())
    assert proof not in archived_text
    archived_request = json.loads((transcript / "03-continuation-request.json").read_text(encoding="utf-8"))
    assert "brokered_continuation_proof" not in archived_request


def test_brokered_conformance_helper_preserves_header_proof_option(tmp_path: Path):
    transcript = tmp_path / "transcript"
    proof = "header-proof-must-stay-out-of-transcript"
    with _mock_gateway() as (endpoint, requests):
        result = _run_helper(endpoint, transcript, AGENTKIT_CONTINUATION_PROOF=proof)

    assert result.returncode == 0, result.stderr
    assert len(requests) == 2
    assert requests[1]["headers"]["x-agentkit-brokered-continuation-proof"] == proof
    assert "brokered_continuation_proof" not in requests[1]["body"]
    archived_text = "\n".join(path.read_text(encoding="utf-8") for path in transcript.iterdir() if path.is_file())
    assert proof not in archived_text


def test_brokered_conformance_helper_refuses_to_archive_echoed_proof(tmp_path: Path):
    transcript = tmp_path / "transcript"
    proof = 'echoed-"proof"\\with\nnewline-must-not-be-archived'
    with _mock_gateway(echo_continuation_proof=True) as (endpoint, requests):
        result = _run_helper(endpoint, transcript, AGENTKIT_CONTINUATION_PROOF_BODY=proof)

    assert len(requests) == 2
    assert result.returncode != 0
    assert not (transcript / "04-continuation-response.json").exists()
    archived_text = "\n".join(path.read_text(encoding="utf-8") for path in transcript.iterdir() if path.is_file())
    assert proof not in archived_text


def test_brokered_conformance_helper_refuses_nested_json_proof_echo(tmp_path: Path):
    transcript = tmp_path / "transcript"
    proof = 'nested-"proof"\\with\nnewline-must-not-be-archived'
    with _mock_gateway(serialize_continuation_body=True) as (endpoint, requests):
        result = _run_helper(endpoint, transcript, AGENTKIT_CONTINUATION_PROOF_BODY=proof)

    assert len(requests) == 2
    assert result.returncode != 0
    assert not (transcript / "04-continuation-response.json").exists()
    archived_text = "\n".join(path.read_text(encoding="utf-8") for path in transcript.iterdir() if path.is_file())
    assert proof not in archived_text


def test_brokered_conformance_helper_refuses_encoded_proof_in_malformed_response(tmp_path: Path):
    transcript = tmp_path / "transcript"
    proof = 'malformed-"proof"\\with\nnewline-must-not-be-archived'
    with _mock_gateway(malformed_continuation_response=True) as (endpoint, requests):
        result = _run_helper(endpoint, transcript, AGENTKIT_CONTINUATION_PROOF_BODY=proof)

    assert len(requests) == 2
    assert result.returncode != 0
    assert not (transcript / "04-continuation-response.json").exists()
    archived_text = "\n".join(path.read_text(encoding="utf-8") for path in transcript.iterdir() if path.is_file())
    assert proof not in archived_text


def test_brokered_conformance_helper_removes_stale_final_artifacts_before_refused_rerun(tmp_path: Path):
    transcript = tmp_path / "transcript"
    transcript.mkdir()
    stale_response = transcript / "04-continuation-response.json"
    stale_summary = transcript / "summary.json"
    stale_response.write_text('{"stale":true}', encoding="utf-8")
    stale_summary.write_text('{"stale":true}', encoding="utf-8")
    proof = "rerun-proof-must-not-be-archived"

    with _mock_gateway(echo_continuation_proof=True) as (endpoint, requests):
        result = _run_helper(endpoint, transcript, AGENTKIT_CONTINUATION_PROOF_BODY=proof)

    assert len(requests) == 2
    assert result.returncode != 0
    assert not stale_response.exists()
    assert not stale_summary.exists()
