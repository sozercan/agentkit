#!/usr/bin/env bash
set -euo pipefail
: "${AZURE_SUBSCRIPTION_ID:?set AZURE_SUBSCRIPTION_ID}"
: "${AZURE_RESOURCE_GROUP:?set AZURE_RESOURCE_GROUP}"
: "${AZURE_AI_ACCOUNT_NAME:?set AZURE_AI_ACCOUNT_NAME}"
: "${AZURE_AI_PROJECT_NAME:?set AZURE_AI_PROJECT_NAME}"
role_name="${AGENTKIT_FOUNDRY_AGENT_ROLE:-Foundry User}"
show_json="${AGENTKIT_FOUNDRY_AGENT_SHOW_JSON:-}"
if [[ -n "$show_json" ]]; then
  principal_id=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["instance_identity"]["principal_id"])' "$show_json")
else
  principal_id=$(azd ai agent show -o json | python3 -c 'import json,sys; print(json.load(sys.stdin)["instance_identity"]["principal_id"])')
fi
project_scope="/subscriptions/${AZURE_SUBSCRIPTION_ID}/resourceGroups/${AZURE_RESOURCE_GROUP}/providers/Microsoft.CognitiveServices/accounts/${AZURE_AI_ACCOUNT_NAME}/projects/${AZURE_AI_PROJECT_NAME}"
account_scope="/subscriptions/${AZURE_SUBSCRIPTION_ID}/resourceGroups/${AZURE_RESOURCE_GROUP}/providers/Microsoft.CognitiveServices/accounts/${AZURE_AI_ACCOUNT_NAME}"
az account set --subscription "$AZURE_SUBSCRIPTION_ID" >/dev/null
for scope in "$project_scope" "$account_scope"; do
  if ! az role assignment create \
    --assignee-object-id "$principal_id" \
    --assignee-principal-type ServicePrincipal \
    --role "$role_name" \
    --scope "$scope" \
    -o none 2>/tmp/agentkit-foundry-rbac.err; then
    if ! grep -qi 'RoleAssignmentExists' /tmp/agentkit-foundry-rbac.err; then
      cat /tmp/agentkit-foundry-rbac.err >&2
      exit 1
    fi
  fi
  printf 'ensured %s for %s at %s\n' "$role_name" "$principal_id" "$scope"
done
