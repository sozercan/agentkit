#!/usr/bin/env bash
set -euo pipefail
: "${MEMORY_ENDPOINT:?set MEMORY_ENDPOINT}"
: "${MEMORY_STORE_NAME:?set MEMORY_STORE_NAME}"
cat > "$(dirname "$0")/output.env" <<OUT
MEMORY_ENDPOINT=${MEMORY_ENDPOINT}
MEMORY_STORE_NAME=${MEMORY_STORE_NAME}
OUT
printf 'wrote %s\n' "$(dirname "$0")/output.env"
