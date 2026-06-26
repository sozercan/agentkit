#!/usr/bin/env bash
set -euo pipefail
: "${FOUNDRY_PROJECT_ENDPOINT:?set FOUNDRY_PROJECT_ENDPOINT}"
: "${TOOLBOX_NAME:?set TOOLBOX_NAME}"
api_version="${TOOLBOX_API_VERSION:-v1}"
endpoint="${FOUNDRY_PROJECT_ENDPOINT%/}/toolboxes/${TOOLBOX_NAME}/mcp?api-version=${api_version}"
cat > "$(dirname "$0")/output.env" <<OUT
TOOLBOX_NAME=${TOOLBOX_NAME}
TOOLBOX_ENDPOINT=${endpoint}
TOOLBOX_MCP_URL=${endpoint}
OUT
printf 'wrote %s\n' "$(dirname "$0")/output.env"
