# Foundry deployment helpers

These helpers keep provider-specific resource wiring outside the core AgentKit
agent schema. They emit generic environment variables consumed by AgentKit
features such as remote MCP tools and context providers.

## Hosted-agent profile shape

```yaml
target: foundry-hosted-agent
agent: ./agentkitfile.yaml
image: docker.io/acme/hr-agent:v1
protocols:
  - invocations
  - responses
env:
  TOOLBOX_ENDPOINT: ${FOUNDRY_PROJECT_ENDPOINT}/toolboxes/${TOOLBOX_NAME}/mcp?api-version=v1
```

The reusable Python Foundry adapter in `agentkit_serve_common.foundry` is selected
with `AGENTKIT_PROTOCOL=foundry` (or `agentkit-serve --protocol foundry`) and
exposes `/readiness`, `/invocations`, and a minimal non-streaming `/responses`
endpoint without adding Foundry-specific keys to `/agent/agent.yaml`.

## Hosted-agent rendering

Use `render_agent.py` to map a provider-neutral Foundry deployment profile to
the `agent.yaml` shape consumed by azd hosted agents:

```sh
python deploy/foundry/render_agent.py deploy/foundry/agentkit.foundry.yaml.example -o agent.yaml
```

The renderer intentionally emits Foundry's sample-compatible
`environment_variables` list form:

```yaml
environment_variables:
  - name: TOOLBOX_ENDPOINT
    value: ${TOOLBOX_ENDPOINT}
```

Do not hand-author the camelCase `environmentVariables` map form for hosted
AgentKit validation: it can validate locally while failing to inject variables
into the hosted container.

## Resource setup

- `toolbox/setup.sh` references a Foundry Toolbox and writes `TOOLBOX_ENDPOINT`.
- `search/setup.sh` records Azure Search context-provider env names.
- `memory/setup.sh` records external memory context-provider env names.
- `rbac/assign-agent-identity.sh` grants the newly-created hosted agent identity
  the project/account role required for workload-identity model/tool calls.
- `scripts/invoke_responses.sh` sends the minimal portable hosted Responses
  payload (`{"input":"..."}`), avoiding gateway-specific optional fields.
- `doctor.sh` checks the expected Foundry/project env and local CLI prerequisites.

These scripts intentionally produce local `output.env` files that should not be
committed with live subscription or endpoint values.
