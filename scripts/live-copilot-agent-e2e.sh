#!/usr/bin/env bash

set -Eeuo pipefail

log() {
  printf '==> %s\n' "$*" >&2
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work_dir="$(mktemp -d "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/agentkit-live-copilot.XXXXXX")"

copilot_token="${COPILOT_GITHUB_TOKEN:-}"
vekil_image="${VEKIL_IMAGE:-ghcr.io/sozercan/vekil@sha256:d13edeedf7bec319da8eb3ea4949a4d0802e244c14765a347e62e1b8b7be8e3d}"
vekil_container_name="${VEKIL_CONTAINER_NAME:-agentkit-vekil}"
vekil_host_port="${VEKIL_HOST_PORT:-1337}"
vekil_container_port="${VEKIL_CONTAINER_PORT:-1337}"
agent_container_name="${AGENTKIT_LIVE_CONTAINER_NAME:-agentkit-maf-live}"
network_name="${AGENTKIT_LIVE_NETWORK:-agentkit-live-copilot}"
agent_host_port="${AGENTKIT_LIVE_HOST_PORT:-18080}"
agent_auth_token="${AGENTKIT_AUTH_TOKEN:-agentkit-live-ci-token}"
tag="${TAG:-ci-live}"
platform="${PLATFORM:-linux/amd64}"
builder="${BUILDER:-}"

redact() {
  local text
  text="$(cat)"
  if [[ -n "${copilot_token}" ]]; then
    text="${text//${copilot_token}/[REDACTED]}"
  fi
  if [[ -n "${agent_auth_token}" ]]; then
    text="${text//${agent_auth_token}/[REDACTED]}"
  fi
  printf '%s' "${text}" | sed -E \
    -e 's/(Authorization: (Bearer|token) )[[:graph:]]+/\1[REDACTED]/g' \
    -e 's/COPILOT_GITHUB_TOKEN=[^[:space:]]+/COPILOT_GITHUB_TOKEN=[REDACTED]/g' \
    -e 's/GITHUB_TOKEN=[^[:space:]]+/GITHUB_TOKEN=[REDACTED]/g' \
    -e 's/gh[opusr]_[A-Za-z0-9_]+/[REDACTED_GITHUB_TOKEN]/g' \
    -e 's/github_pat_[A-Za-z0-9_]+/[REDACTED_GITHUB_TOKEN]/g' \
    -e 's/("access_token"[[:space:]]*:[[:space:]]*")[^"]+"/\1[REDACTED]"/g' \
    -e 's/("token"[[:space:]]*:[[:space:]]*")[^"]+"/\1[REDACTED]"/g'
}

cleanup() {
  docker rm -f "${agent_container_name}" "${vekil_container_name}" >/dev/null 2>&1 || true
  docker network rm "${network_name}" >/dev/null 2>&1 || true
  rm -rf "${work_dir}" >/dev/null 2>&1 || true
}

on_exit() {
  local status="$1"
  if [[ "${status}" -ne 0 ]]; then
    log "Collecting redacted diagnostics"
    {
      echo "=== docker ps -a ==="
      docker ps -a || true
      echo
      echo "=== Vekil logs ==="
      docker logs "${vekil_container_name}" 2>&1 || true
      echo
      echo "=== agent logs ==="
      docker logs "${agent_container_name}" 2>&1 || true
    } | redact >&2
    log "Live Vekil-backed AgentKit E2E failed"
  fi
  cleanup
}

wait_for_http() {
  local url="$1"
  local description="$2"
  local attempts="${3:-90}"

  for _ in $(seq 1 "${attempts}"); do
    if curl -fsS "${url}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done

  die "${description} never became available at ${url}"
}

wait_for_vekil_ready() {
  local url="http://127.0.0.1:${vekil_host_port}/readyz"
  local attempts="${1:-90}"

  for _ in $(seq 1 "${attempts}"); do
    if curl -fsS "${url}" >/dev/null 2>&1; then
      return 0
    fi

    # Vekil exits quickly when the supplied GitHub token cannot be exchanged for
    # a Copilot token. Treat that as an environment/credential skip for this
    # optional live job; real build, networking, and agent failures still fail.
    if ! docker inspect -f '{{.State.Running}}' "${vekil_container_name}" 2>/dev/null | grep -qx true; then
      logs="$(docker logs "${vekil_container_name}" 2>&1 || true)"
      if printf '%s' "${logs}" | grep -Eq 'copilot token request failed with status 403|authentication failed:.*status 403'; then
        log "Skipping live Vekil/Copilot E2E: COPILOT_GITHUB_TOKEN was rejected by Vekil's Copilot token exchange (HTTP 403)."
        log "Update the repository secret to a token for a Copilot-enabled user with the Copilot Requests permission to run this live check."
        return 2
      fi
    fi

    sleep 2
  done

  die "Vekil /readyz never became available at ${url}"
}

main() {
  require_cmd curl
  require_cmd docker
  require_cmd go
  require_cmd jq
  require_cmd make

  [[ -n "${copilot_token}" ]] || die "COPILOT_GITHUB_TOKEN is required"

  trap 'on_exit $?' EXIT

  cd "${repo_root}"

  log "Creating private Docker network ${network_name}"
  docker network rm "${network_name}" >/dev/null 2>&1 || true
  docker network create "${network_name}" >/dev/null

  log "Starting Vekil (${vekil_image})"
  docker rm -f "${vekil_container_name}" >/dev/null 2>&1 || true
  docker run -d --name "${vekil_container_name}" \
    --network "${network_name}" \
    --network-alias host.docker.internal \
    -p "127.0.0.1:${vekil_host_port}:${vekil_container_port}" \
    -e COPILOT_GITHUB_TOKEN="${copilot_token}" \
    -e PORT="${vekil_container_port}" \
    -e TOKEN_DIR=/home/nonroot/.config/vekil \
    "${vekil_image}" >/dev/null

  log "Waiting for Vekil /readyz"
  if ! wait_for_vekil_ready; then
    exit 0
  fi

  log "Validating Vekil /v1/models"
  curl -fsS "http://127.0.0.1:${vekil_host_port}/v1/models" >"${work_dir}/models.json"
  jq -e '.data | length > 0' "${work_dir}/models.json" >/dev/null
  jq -r '.data[].id' "${work_dir}/models.json" | sed 's/^/model: /' | redact >&2
  jq -e '.data[] | select(.id == "claude-haiku-4.5")' "${work_dir}/models.json" >/dev/null

  log "Building AgentKit frontend and MAF adapter"
  buildx_args=()
  if [[ -n "${builder}" ]]; then
    docker buildx inspect "${builder}" --bootstrap
    buildx_args=(--builder "${builder}")
  else
    docker buildx inspect --bootstrap
  fi
  make build-agentkit TAG="${tag}"
  make build-serve-maf TAG="${tag}"

  log "Building live MAF agent image"
  docker buildx build "${buildx_args[@]}" . -f test/agentkitfile-maf-live.yaml \
    --build-arg BUILDKIT_SYNTAX="agentkit:${tag}" \
    --build-arg adapter="agentkit-serve-maf:${tag}" \
    --platform "${platform}" \
    -t "maf-live-agent:${tag}" --load --provenance=false

  log "Starting live MAF agent"
  docker rm -f "${agent_container_name}" >/dev/null 2>&1 || true
  docker run -d --name "${agent_container_name}" \
    --platform "${platform}" \
    --network "${network_name}" \
    -p "127.0.0.1:${agent_host_port}:8080" \
    -e AGENTKIT_BIND=0.0.0.0 \
    -e AGENTKIT_AUTH_TOKEN="${agent_auth_token}" \
    -e MODEL_API_KEY=not-needed \
    "maf-live-agent:${tag}" >/dev/null

  log "Waiting for live MAF agent /healthz"
  wait_for_http "http://127.0.0.1:${agent_host_port}/healthz" "live MAF agent /healthz"

  log "Calling live MAF agent /v1/chat/completions"
  curl -fsS \
    -H "Authorization: Bearer ${agent_auth_token}" \
    -H "Content-Type: application/json" \
    --data '{"model":"claude-haiku-4.5","stream":false,"messages":[{"role":"user","content":"Reply with exactly one short sentence that includes the sentinel token DONE42."}]}' \
    "http://127.0.0.1:${agent_host_port}/v1/chat/completions" \
    >"${work_dir}/agent-response.json"

  jq '{model, content: .choices[0].message.content}' "${work_dir}/agent-response.json" | redact >&2
  jq -e '.choices[0].message.content | type == "string" and contains("DONE42")' "${work_dir}/agent-response.json" >/dev/null

  log "Live Vekil-backed AgentKit E2E passed"
}

main "$@"
