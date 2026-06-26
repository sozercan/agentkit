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

The reusable Python Foundry adapter in `agentkit_serve_common.foundry` exposes
`/readiness`, `/invocations`, and a minimal non-streaming `/responses` endpoint
without adding Foundry-specific keys to `/agent/agent.yaml`.

## Resource setup

- `toolbox/setup.sh` references a Foundry Toolbox and writes `TOOLBOX_ENDPOINT`.
- `search/setup.sh` records Azure Search context-provider env names.
- `memory/setup.sh` records external memory context-provider env names.
- `doctor.sh` checks the expected Foundry/project env and local CLI prerequisites.

These scripts intentionally produce local `output.env` files that should not be
committed with live subscription or endpoint values.
