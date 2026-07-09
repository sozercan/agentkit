from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from fastapi.testclient import TestClient

from agentkit_serve_common.foundry_conformance import create_foundry_conformance_app


def _load_verifier():
    repo = Path(__file__).resolve().parents[3]
    path = repo / "deploy" / "foundry" / "scripts" / "verify_brokered_transcript.py"
    spec = importlib.util.spec_from_file_location("verify_brokered_transcript", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_transcript(tmp_path: Path) -> Path:
    app = create_foundry_conformance_app()
    with TestClient(app) as client:
        initial_request = {"input": "conformance_read"}
        initial_response = client.post("/responses", json=initial_request).json()
        continuation_request = {
            "previous_response_id": initial_response["id"],
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": initial_response["output"][0]["call_id"],
                    "output": '{"approved":true,"output":{"success":true}}',
                    "status": "completed",
                }
            ],
        }
        continuation_response = client.post("/responses", json=continuation_request).json()

    files = {
        "01-initial-request.json": initial_request,
        "02-initial-response.json": initial_response,
        "03-continuation-request.json": continuation_request,
        "04-continuation-response.json": continuation_response,
    }
    for name, payload in files.items():
        (tmp_path / name).write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return tmp_path


def test_verify_brokered_transcript_accepts_conformance_loop(tmp_path):
    verifier = _load_verifier()
    transcript = _write_transcript(tmp_path)

    summary = verifier.verify_transcript(transcript)

    assert summary["initial_response_id"].startswith("caresp_")
    assert summary["continuation_response_id"].startswith("caresp_")
    assert summary["call_id"] == "call_conformance_1"
    assert "success" in summary["final_text"]


def test_verify_brokered_transcript_rejects_old_response_ids(tmp_path):
    verifier = _load_verifier()
    transcript = _write_transcript(tmp_path)
    initial = json.loads((transcript / "02-initial-response.json").read_text(encoding="utf-8"))
    initial["id"] = "resp_old"
    (transcript / "02-initial-response.json").write_text(json.dumps(initial), encoding="utf-8")

    try:
        verifier.verify_transcript(transcript)
    except ValueError as exc:
        assert "caresp_" in str(exc)
    else:  # pragma: no cover - assertion path.
        raise AssertionError("expected old response id to fail")


def test_verify_brokered_transcript_cli_writes_summary(tmp_path, capsys):
    verifier = _load_verifier()
    transcript = _write_transcript(tmp_path)

    assert verifier.main([str(transcript), "--write-summary"]) == 0

    output = json.loads(capsys.readouterr().out)
    written = json.loads((transcript / "summary.json").read_text(encoding="utf-8"))
    assert output == written
    assert written["call_id"] == "call_conformance_1"


def test_verify_brokered_transcript_accepts_agentkit_generated_call_id(tmp_path):
    from agentkit_serve_common.config import AgentSpec
    from agentkit_serve_common.foundry import create_foundry_app

    class Factory:
        def build_runtime(self, spec):  # noqa: ANN001
            raise AssertionError("brokered Foundry mode must not build direct runtime")

    spec = AgentSpec.model_validate(
        {
            "abiVersion": "v0",
            "metadata": {"name": "agentkit-brokered"},
            "model": {"provider": "openai-compatible", "baseURL": "https://api.openai.com/v1", "name": "gpt-4o-mini"},
            "instructions": "Be helpful.",
            "tools": [],
            "brokeredTools": [
                {
                    "name": "conformance_read",
                    "description": "Read conformance data.",
                    "brokeredClass": "read",
                    "parameters": {"type": "object", "properties": {"probe": {"type": "boolean"}}},
                }
            ],
            "expose": {"openai": True, "port": 8080},
        }
    )
    app = create_foundry_app(spec, Factory(), brokered_continuation_proof="proof")
    initial_request = {"input": "conformance_read"}
    with TestClient(app) as client:
        initial_response = client.post("/responses", json=initial_request).json()
        continuation_request = {
            "previous_response_id": initial_response["id"],
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": initial_response["output"][0]["call_id"],
                    "output": '{"approved":true,"output":{"success":true}}',
                    "status": "completed",
                }
            ],
        }
        continuation_response = client.post(
            "/responses",
            headers={"x-agentkit-brokered-continuation-proof": "proof"},
            json=continuation_request,
        ).json()
    for name, payload in {
        "01-initial-request.json": initial_request,
        "02-initial-response.json": initial_response,
        "03-continuation-request.json": continuation_request,
        "04-continuation-response.json": continuation_response,
    }.items():
        (tmp_path / name).write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

    summary = _load_verifier().verify_transcript(
        tmp_path,
        expected_call_id="auto",
        expected_call_id_prefix="call_",
    )

    assert summary["call_id"].startswith("call_")
    assert summary["arguments"] == {"probe": True}
