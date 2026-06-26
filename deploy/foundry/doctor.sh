#!/usr/bin/env bash
set -euo pipefail
missing=0
need() {
  if [[ -z "${!1:-}" ]]; then
    printf 'missing env: %s\n' "$1" >&2
    missing=1
  fi
}
need FOUNDRY_PROJECT_ENDPOINT
if ! command -v az >/dev/null 2>&1; then
  printf 'warning: az CLI not found; hosted resource/RBAC checks cannot run locally\n' >&2
fi
if ! command -v azd >/dev/null 2>&1; then
  printf 'warning: azd CLI not found; hosted-agent deploy/invoke checks cannot run locally\n' >&2
fi
if [[ "$missing" -ne 0 ]]; then
  exit 2
fi
printf 'foundry doctor: basic local prerequisites checked\n'
