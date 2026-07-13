from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

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


def test_foundry_brokered_conformance_script_uses_private_file_umask():
    script = Path(__file__).parents[3] / "deploy" / "foundry" / "scripts" / "foundry_brokered_conformance.sh"
    lines = script.read_text(encoding="utf-8").splitlines()

    assert "umask 077" in lines[:5]


def test_foundry_brokered_conformance_script_keeps_sensitive_headers_out_of_curl_argv():
    script = Path(__file__).parents[3] / "deploy" / "foundry" / "scripts" / "foundry_brokered_conformance.sh"
    text = script.read_text(encoding="utf-8")

    assert text.count("--config -") == 2
    assert '-H "Authorization: Bearer ${token}"' not in text
    assert '-H "x-agentkit-brokered-continuation-proof:' not in text
    assert '--expected-output-file "$expected_output_file"' in text
    assert '--expected-output-json "$conformance_output"' not in text
    assert '--expected-final-text-file "$expected_final_text_file"' in text


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


def test_verify_brokered_transcript_rejects_duplicate_keys_in_top_level_files(tmp_path):
    verifier = _load_verifier()
    transcript = _write_transcript(tmp_path)
    initial_path = transcript / "02-initial-response.json"
    initial = json.loads(initial_path.read_text(encoding="utf-8"))
    encoded = json.dumps(initial, separators=(",", ":"))
    initial_path.write_text(encoded[:-1] + ',"status":"failed"}', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        verifier.verify_transcript(transcript)


def test_verify_brokered_transcript_rejects_mismatched_function_call_response_id(tmp_path):
    verifier = _load_verifier()
    transcript = _write_transcript(tmp_path)
    initial_path = transcript / "02-initial-response.json"
    initial = json.loads(initial_path.read_text(encoding="utf-8"))
    initial["output"][0]["response_id"] = "caresp_other"
    initial_path.write_text(json.dumps(initial), encoding="utf-8")

    with pytest.raises(ValueError, match="function_call response_id must match"):
        verifier.verify_transcript(transcript)


def test_verify_brokered_transcript_rejects_mismatched_final_message_response_id(tmp_path):
    verifier = _load_verifier()
    transcript = _write_transcript(tmp_path)
    continuation_path = transcript / "04-continuation-response.json"
    continuation = json.loads(continuation_path.read_text(encoding="utf-8"))
    continuation["output"][0]["response_id"] = "caresp_other"
    continuation_path.write_text(json.dumps(continuation), encoding="utf-8")

    with pytest.raises(ValueError, match="final message response_id must match"):
        verifier.verify_transcript(transcript)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("role", "user", "final message role must be assistant"),
        ("status", "in_progress", "final message status must be completed"),
    ],
)
def test_verify_brokered_transcript_rejects_noncompleted_assistant_message(
    tmp_path, field: str, value: str, message: str
):
    verifier = _load_verifier()
    transcript = _write_transcript(tmp_path)
    continuation_path = transcript / "04-continuation-response.json"
    continuation = json.loads(continuation_path.read_text(encoding="utf-8"))
    continuation["output"][0][field] = value
    continuation_path.write_text(json.dumps(continuation), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        verifier.verify_transcript(transcript)


def test_verify_brokered_transcript_rejects_reused_continuation_response_id(tmp_path):
    verifier = _load_verifier()
    transcript = _write_transcript(tmp_path)
    initial = json.loads((transcript / "02-initial-response.json").read_text(encoding="utf-8"))
    continuation = json.loads((transcript / "04-continuation-response.json").read_text(encoding="utf-8"))
    continuation["id"] = initial["id"]
    (transcript / "04-continuation-response.json").write_text(json.dumps(continuation), encoding="utf-8")

    with pytest.raises(ValueError, match="must differ from initial response id"):
        verifier.verify_transcript(transcript)


def test_verify_brokered_transcript_rejects_extra_final_output_items(tmp_path):
    verifier = _load_verifier()
    transcript = _write_transcript(tmp_path)
    continuation = json.loads((transcript / "04-continuation-response.json").read_text(encoding="utf-8"))
    continuation["output"].append(
        {
            "type": "function_call",
            "id": "fc_unexpected",
            "call_id": "call_unexpected",
            "name": "unexpected",
            "arguments": "{}",
            "status": "completed",
        }
    )
    (transcript / "04-continuation-response.json").write_text(json.dumps(continuation), encoding="utf-8")

    with pytest.raises(ValueError, match="must contain exactly one item"):
        verifier.verify_transcript(transcript)


def test_verify_brokered_transcript_rejects_unexpected_final_text(tmp_path):
    verifier = _load_verifier()
    transcript = _write_transcript(tmp_path)

    with pytest.raises(ValueError, match="final message text did not match"):
        verifier.verify_transcript(transcript, expected_final_text="unrelated response")


def test_verify_brokered_transcript_rejects_unexpected_continuation_output(tmp_path):
    verifier = _load_verifier()
    transcript = _write_transcript(tmp_path)
    continuation = json.loads((transcript / "03-continuation-request.json").read_text(encoding="utf-8"))
    continuation["input"][0]["output"] = '{"approved":false,"error":{"code":"denied"}}'
    (transcript / "03-continuation-request.json").write_text(json.dumps(continuation), encoding="utf-8")

    with pytest.raises(ValueError, match="continuation output did not match expected JSON"):
        verifier.verify_transcript(transcript)


def test_verify_brokered_transcript_compares_output_json_types_strictly(tmp_path):
    verifier = _load_verifier()
    transcript = _write_transcript(tmp_path)
    continuation = json.loads((transcript / "03-continuation-request.json").read_text(encoding="utf-8"))
    continuation["input"][0]["output"] = '{"approved":1,"output":{"success":1}}'
    (transcript / "03-continuation-request.json").write_text(json.dumps(continuation), encoding="utf-8")

    with pytest.raises(ValueError, match="continuation output did not match expected JSON"):
        verifier.verify_transcript(transcript)


def test_verify_brokered_transcript_json_comparison_distinguishes_booleans_but_normalizes_numbers():
    verifier = _load_verifier()

    assert verifier._strict_json_equal({"limit": 1}, {"limit": 1.0})
    assert not verifier._strict_json_equal({"approved": True}, {"approved": 1})
    assert not verifier._strict_json_equal(
        verifier._parse_json_lossless('{"value":9007199254740992.0}'),
        verifier._parse_json_lossless('{"value":9007199254740993.0}'),
    )
    compatible = verifier._json_compatible(verifier._parse_json_lossless('{"count":9007199254740993,"ratio":0.5}'))
    assert compatible == {"count": 9007199254740993, "ratio": 0.5}
    json.dumps(compatible)


def test_verify_brokered_transcript_rejects_duplicate_json_keys_and_huge_summary_integers():
    verifier = _load_verifier()

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        verifier._parse_json_lossless('{"probe":false,"probe":true}')
    with pytest.raises(ValueError, match="too large to include safely"):
        verifier._json_compatible(verifier._parse_json_lossless("1e1000000"))


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
