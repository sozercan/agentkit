#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: local_brokered_conformance_container.sh [--fixture sdk|agentkit] [--platform linux/amd64] [--tag TAG] [--port PORT] [--transcript-dir DIR]

Builds and runs a local Foundry brokered Responses conformance container, then
validates the full function_call/function_call_output loop with
foundry_brokered_conformance.sh. This is local proof for packaged container paths
before pushing/deploying images to Foundry.

Fixtures:
  sdk       Minimal Azure Responses SDK conformance app (default).
  agentkit  Production AgentKit create_foundry_app brokered-only path.

Options:
  --fixture NAME         sdk or agentkit (default: sdk).
  --platform PLATFORM   Optional docker build/run platform, e.g. linux/amd64.
  --tag TAG             Image tag to build/run. Defaults per fixture.
  --port PORT           Local host port. Defaults per fixture.
  --transcript-dir DIR  Transcript directory. Defaults to a temp directory.
  -h, --help            Show this help.
EOF
}

fixture="sdk"
platform=""
tag=""
port=""
transcript_dir=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fixture)
      fixture="${2:?--fixture requires a value}"
      shift 2
      ;;
    --platform)
      platform="${2:?--platform requires a value}"
      shift 2
      ;;
    --tag)
      tag="${2:?--tag requires a value}"
      shift 2
      ;;
    --port)
      port="${2:?--port requires a value}"
      shift 2
      ;;
    --transcript-dir)
      transcript_dir="${2:?--transcript-dir requires a value}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

case "$fixture" in
  sdk)
    dockerfile="test/foundry-brokered-conformance/Dockerfile"
    tag="${tag:-agentkit-foundry-brokered-conformance:local}"
    port="${port:-18088}"
    expected_call_id="call_conformance_1"
    expected_call_id_prefix=""
    continuation_proof=""
    ;;
  agentkit)
    dockerfile="test/foundry-brokered-agentkit/Dockerfile"
    tag="${tag:-agentkit-foundry-brokered:local}"
    port="${port:-18092}"
    expected_call_id="auto"
    expected_call_id_prefix="call_"
    continuation_proof="local-dev-proof"
    ;;
  *)
    usage
    exit 2
    ;;
esac

if [[ -z "$transcript_dir" ]]; then
  transcript_dir="$(mktemp -d "${TMPDIR:-/tmp}/agentkit-foundry-brokered-${fixture}.XXXXXX")"
fi

for cmd in docker curl python3; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    printf 'missing command: %s\n' "$cmd" >&2
    exit 2
  fi
done

name_suffix="$(printf '%s-%s-%s' "$fixture" "$tag" "$port" | tr -c 'A-Za-z0-9_.-' '-')"
container_name="agentkit-foundry-brokered-${name_suffix}"
run_args=()
if [[ -n "$platform" ]]; then
  run_args+=(--platform "$platform")
fi
if [[ "$fixture" == "agentkit" ]]; then
  run_args+=(-e "AGENTKIT_AUTH_TOKEN=local-dummy-token")
fi
if [[ -n "$continuation_proof" ]]; then
  run_args+=(-e "AGENTKIT_FOUNDRY_BROKERED_CONTINUATION_PROOF=${continuation_proof}")
fi

cleanup() {
  docker rm -f "$container_name" >/dev/null 2>&1 || true
}
trap cleanup EXIT

cleanup

build_args=(.)
if [[ -n "$platform" ]]; then
  build_args=(--platform "$platform" "${build_args[@]}")
fi

docker build "${build_args[@]}" -f "$dockerfile" -t "$tag"

docker run -d --rm \
  "${run_args[@]}" \
  --name "$container_name" \
  -p "127.0.0.1:${port}:8088" \
  "$tag" >/dev/null

ready_url="http://127.0.0.1:${port}/readiness"
for _ in $(seq 1 80); do
  if curl -fsS "$ready_url" >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done
curl -fsS "$ready_url" >/dev/null

helper_env=(
  "AGENT_RESPONSES_ENDPOINT=http://127.0.0.1:${port}/responses"
  "AGENT_RESPONSES_BEARER_TOKEN=local-dummy-token"
  "AGENTKIT_EXPECTED_CALL_ID=${expected_call_id}"
)
if [[ -n "$expected_call_id_prefix" ]]; then
  helper_env+=("AGENTKIT_EXPECTED_CALL_ID_PREFIX=${expected_call_id_prefix}")
fi
if [[ -n "$continuation_proof" ]]; then
  helper_env+=("AGENTKIT_CONTINUATION_PROOF=${continuation_proof}")
fi

env "${helper_env[@]}" deploy/foundry/scripts/foundry_brokered_conformance.sh conformance_read "$transcript_dir"

image_info="$(docker image inspect "$tag" --format '{{.Id}} {{.Architecture}} {{.Os}}')"
printf 'Local Foundry brokered %s container passed.\n' "$fixture"
printf '  image: %s\n' "$tag"
printf '  image_info: %s\n' "$image_info"
printf '  transcript: %s\n' "$transcript_dir"
