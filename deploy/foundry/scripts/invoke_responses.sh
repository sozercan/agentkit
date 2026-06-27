#!/usr/bin/env bash
set -euo pipefail
: "${AGENT_RESPONSES_ENDPOINT:?set AGENT_RESPONSES_ENDPOINT}"
: "${AZURE_SUBSCRIPTION_ID:?set AZURE_SUBSCRIPTION_ID}"
prompt="${1:?usage: invoke_responses.sh '<prompt>'}"
az account set --subscription "$AZURE_SUBSCRIPTION_ID" >/dev/null
token=$(az account get-access-token --resource https://ai.azure.com --query accessToken -o tsv)
payload=$(PROMPT="$prompt" python3 -c 'import json,os; print(json.dumps({"input": os.environ["PROMPT"]}))')
curl -fsS -H "Authorization: Bearer ${token}" -H 'content-type: application/json' \
  "$AGENT_RESPONSES_ENDPOINT" -d "$payload"
