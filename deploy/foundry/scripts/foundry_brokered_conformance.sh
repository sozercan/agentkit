#!/usr/bin/env bash
set -euo pipefail
umask 077

usage() {
  cat >&2 <<'EOF'
usage: foundry_brokered_conformance.sh [prompt] [transcript-dir]

Runs the Phase A0 hosted Responses brokered conformance loop against a deployed
Foundry hosted agent endpoint:
  1. POST an initial /responses request with no request-level tools.
  2. Assert the response contains the deterministic conformance_read function_call.
  3. POST a function_call_output continuation with previous_response_id.
  4. Assert the final response is a completed assistant message.

Required environment:
  AGENT_RESPONSES_ENDPOINT   Full deployed /responses endpoint URL.

Authentication, one of:
  AGENT_RESPONSES_BEARER_TOKEN  Pre-acquired bearer token for the endpoint.
  AZURE_SUBSCRIPTION_ID         Optional subscription to select before `az account get-access-token`.
                               If omitted, the current `az` account is used.

Optional:
  AGENTKIT_CONFORMANCE_OUTPUT   Defaults to {"approved":true,"output":{"success":true}}.
  AGENTKIT_EXPECTED_TOOL_NAME   Defaults to conformance_read.
  AGENTKIT_EXPECTED_ARGUMENTS   Defaults to {"probe":true}.
  AGENTKIT_EXPECTED_CALL_ID     Defaults to call_conformance_1; set to auto for generated IDs.
  AGENTKIT_EXPECTED_CALL_ID_PREFIX  Optional required call_id prefix, e.g. call_.
  AGENTKIT_EXPECTED_FINAL_TEXT  Optional exact final assistant text; defaults are derived for SDK/AgentKit fixtures.
  AGENTKIT_CONTINUATION_PROOF   Optional x-agentkit-brokered-continuation-proof header.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

write_curl_header() {
  local header="$1"
  if [[ "$header" == *$'\r'* || "$header" == *$'\n'* ]]; then
    echo "refusing curl header containing a newline" >&2
    return 1
  fi
  header="${header//\\/\\\\}"
  header="${header//\"/\\\"}"
  printf 'header = "%s"\n' "$header"
}

: "${AGENT_RESPONSES_ENDPOINT:?set AGENT_RESPONSES_ENDPOINT to the deployed /responses URL}"

prompt="${1:-conformance_read}"
transcript_dir="${2:-$(mktemp -d "${TMPDIR:-/tmp}/agentkit-foundry-brokered.XXXXXX")}"
mkdir -p "$transcript_dir"

if [[ -n "${AGENT_RESPONSES_BEARER_TOKEN:-}" ]]; then
  token="$AGENT_RESPONSES_BEARER_TOKEN"
else
  if [[ -n "${AZURE_SUBSCRIPTION_ID:-}" ]]; then
    az account set --subscription "$AZURE_SUBSCRIPTION_ID" >/dev/null
  else
    az account show >/dev/null
  fi
  token="$(az account get-access-token --resource https://ai.azure.com --query accessToken -o tsv)"
fi

conformance_output="${AGENTKIT_CONFORMANCE_OUTPUT:-{\"approved\":true,\"output\":{\"success\":true}}}"
expected_tool_name="${AGENTKIT_EXPECTED_TOOL_NAME:-conformance_read}"
expected_arguments="${AGENTKIT_EXPECTED_ARGUMENTS:-{\"probe\":true}}"
expected_call_id="${AGENTKIT_EXPECTED_CALL_ID:-call_conformance_1}"
expected_call_id_prefix="${AGENTKIT_EXPECTED_CALL_ID_PREFIX:-}"
expected_final_text="${AGENTKIT_EXPECTED_FINAL_TEXT:-}"
if [[ -z "$expected_final_text" ]]; then
  expected_final_text="$(
    EXPECTED_CALL_ID="$expected_call_id" EXPECTED_TOOL_NAME="$expected_tool_name" CONFORMANCE_OUTPUT="$conformance_output" python3 - <<'PY'
import json
import os

call_id = os.environ["EXPECTED_CALL_ID"]
tool_name = os.environ["EXPECTED_TOOL_NAME"]
payload = json.loads(os.environ["CONFORMANCE_OUTPUT"])
if call_id == "call_conformance_1":
    print(f"conformance complete: {json.dumps(payload, sort_keys=True)}")
elif payload.get("approved") is True:
    output = payload.get("output") if isinstance(payload.get("output"), dict) else {}
    print(f"Brokered tool {tool_name} completed with output: {json.dumps(output, separators=(',', ':'), sort_keys=True)}")
else:
    error = payload.get("error") if isinstance(payload.get("error"), dict) else {}
    code = str(error.get("code") or "brokered_tool_denied")
    message = str(error.get("message") or "brokered tool was not performed")
    print(f"Brokered tool {tool_name} was not performed: {code}: {message}")
PY
  )"
fi
initial_request="$transcript_dir/01-initial-request.json"
initial_response="$transcript_dir/02-initial-response.json"
continuation_request="$transcript_dir/03-continuation-request.json"
continuation_response="$transcript_dir/04-continuation-response.json"
summary_file="$transcript_dir/summary.json"
expected_output_file="$transcript_dir/.expected-output.json"
expected_final_text_file="$transcript_dir/.expected-final-text.txt"
trap 'rm -f -- "$expected_output_file" "$expected_final_text_file"' EXIT
printf '%s' "$conformance_output" >"$expected_output_file"
printf '%s' "$expected_final_text" >"$expected_final_text_file"

PROMPT="$prompt" python3 - <<'PY' >"$initial_request"
import json
import os
print(json.dumps({"input": os.environ["PROMPT"]}, separators=(",", ":")))
PY

initial_curl_config="$(write_curl_header "Authorization: Bearer ${token}")"
printf '%s\n' "$initial_curl_config" | curl -fsS \
  --config - \
  -H 'content-type: application/json' \
  "$AGENT_RESPONSES_ENDPOINT" \
  -d "@$initial_request" >"$initial_response"
unset initial_curl_config

read -r response_id call_id < <(EXPECTED_TOOL_NAME="$expected_tool_name" EXPECTED_ARGUMENTS="$expected_arguments" EXPECTED_CALL_ID="$expected_call_id" EXPECTED_CALL_ID_PREFIX="$expected_call_id_prefix" python3 - "$initial_response" <<'PY'
import json
import os
import sys
from pathlib import Path

body = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert body.get("status") == "completed", body
response_id = body.get("id")
assert isinstance(response_id, str) and response_id.startswith("caresp_"), body
assert not response_id.startswith("resp_"), body
output = body.get("output")
assert isinstance(output, list) and len(output) == 1, body
call = output[0]
assert call.get("type") == "function_call", call
expected_tool_name = os.environ["EXPECTED_TOOL_NAME"]
expected_arguments = json.loads(os.environ["EXPECTED_ARGUMENTS"])
expected_call_id = os.environ["EXPECTED_CALL_ID"]
expected_call_id_prefix = os.environ.get("EXPECTED_CALL_ID_PREFIX", "")
assert call.get("name") == expected_tool_name, call
call_id = call.get("call_id")
assert isinstance(call_id, str) and call_id, call
if expected_call_id != "auto":
    assert call_id == expected_call_id, call
if expected_call_id_prefix:
    assert call_id.startswith(expected_call_id_prefix), call
assert json.loads(call.get("arguments") or "{}") == expected_arguments, call
print(response_id, call_id)
PY
)

PREVIOUS_RESPONSE_ID="$response_id" CALL_ID="$call_id" CONFORMANCE_OUTPUT="$conformance_output" python3 - <<'PY' >"$continuation_request"
import json
import os
# Validate the configured output is JSON before placing it in the Responses item.
json.loads(os.environ["CONFORMANCE_OUTPUT"])
print(json.dumps({
    "previous_response_id": os.environ["PREVIOUS_RESPONSE_ID"],
    "input": [{
        "type": "function_call_output",
        "call_id": os.environ["CALL_ID"],
        "output": os.environ["CONFORMANCE_OUTPUT"],
        "status": "completed",
    }],
}, separators=(",", ":")))
PY

continuation_curl_config="$({
  write_curl_header "Authorization: Bearer ${token}"
  if [[ -n "${AGENTKIT_CONTINUATION_PROOF:-}" ]]; then
    write_curl_header "x-agentkit-brokered-continuation-proof: ${AGENTKIT_CONTINUATION_PROOF}"
  fi
})"
printf '%s\n' "$continuation_curl_config" | curl -fsS \
  --config - \
  -H 'content-type: application/json' \
  "$AGENT_RESPONSES_ENDPOINT" \
  -d "@$continuation_request" >"$continuation_response"
unset continuation_curl_config

verifier_args=(
  "$transcript_dir"
  --expected-tool-name "$expected_tool_name"
  --expected-arguments-json "$expected_arguments"
  --expected-output-file "$expected_output_file"
  --expected-final-text-file "$expected_final_text_file"
  --expected-call-id "$expected_call_id"
  --write-summary
)
if [[ -n "$expected_call_id_prefix" ]]; then
  verifier_args+=(--expected-call-id-prefix "$expected_call_id_prefix")
fi
python3 deploy/foundry/scripts/verify_brokered_transcript.py "${verifier_args[@]}" >"$summary_file.tmp"
rm -f "$summary_file.tmp"
rm -f "$expected_output_file" "$expected_final_text_file"
trap - EXIT

echo "Foundry brokered conformance passed. Sanitized transcript: ${transcript_dir}"
cat "$summary_file"
