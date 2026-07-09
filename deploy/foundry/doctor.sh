#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: doctor.sh [--brokered-conformance]

Default mode checks the generic Foundry deployment helper prerequisites.
--brokered-conformance checks the env/tools needed to run
  deploy/foundry/scripts/foundry_brokered_conformance.sh
against a deployed hosted-agent /responses endpoint.
EOF
}

mode="default"
case "${1:-}" in
  "") ;;
  --brokered-conformance) mode="brokered-conformance" ;;
  -h|--help) usage; exit 0 ;;
  *) usage; exit 2 ;;
esac

missing=0
need() {
  if [[ -z "${!1:-}" ]]; then
    printf 'missing env: %s\n' "$1" >&2
    missing=1
  fi
}
need_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf '%s: %s\n' "$2" "$1" >&2
    if [[ "${3:-required}" == "required" ]]; then
      missing=1
    fi
  fi
}

if [[ "$mode" == "brokered-conformance" ]]; then
  need AGENT_RESPONSES_ENDPOINT
  need_command curl "missing command"
  need_command python3 "missing command"
  if [[ -z "${AGENT_RESPONSES_BEARER_TOKEN:-}" ]]; then
    need_command az "missing command"
    if command -v az >/dev/null 2>&1; then
      if [[ -n "${AZURE_SUBSCRIPTION_ID:-}" ]]; then
        az account set --subscription "$AZURE_SUBSCRIPTION_ID" >/dev/null 2>&1 || missing=1
      fi
      if ! az account show >/dev/null 2>&1; then
        printf 'missing auth: set AGENT_RESPONSES_BEARER_TOKEN or run az login/select an account
' >&2
        missing=1
      fi
    fi
  fi
  if [[ ! -x deploy/foundry/scripts/foundry_brokered_conformance.sh ]]; then
    printf 'missing executable: deploy/foundry/scripts/foundry_brokered_conformance.sh\n' >&2
    missing=1
  fi
  if [[ ! -f deploy/foundry/scripts/verify_brokered_transcript.py ]]; then
    printf 'missing verifier: deploy/foundry/scripts/verify_brokered_transcript.py\n' >&2
    missing=1
  fi
  if [[ "$missing" -ne 0 ]]; then
    exit 2
  fi
  printf 'foundry doctor: brokered conformance prerequisites checked\n'
  exit 0
fi

need FOUNDRY_PROJECT_ENDPOINT
need_command az "warning: az CLI not found; hosted resource/RBAC checks cannot run locally" optional
need_command azd "warning: azd CLI not found; hosted-agent deploy/invoke checks cannot run locally" optional
if [[ "$missing" -ne 0 ]]; then
  exit 2
fi
printf 'foundry doctor: basic local prerequisites checked\n'
