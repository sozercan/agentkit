# AgentKit pydantic-ai runtime adapter

`runtimes/pydantic-ai` builds the default AgentKit runtime adapter image. It
loads `/agent/agent.yaml` through `agentkit-serve-common` and serves a
non-streaming OpenAI-compatible Chat Completions façade backed by a pydantic-ai
`Agent`.

## Responsibilities

- Build an `OpenAIChatModel` from `model.baseURL`, `model.name`, and the env var
  named by `model.apiKeyEnv`.
- Attach stdio MCP tools declared in the ABI.
- Support both pydantic-ai MCP APIs used by current 1.x and newer 2.x releases.
- Prefix tool names by tool server name to avoid collisions.
- Map pydantic-ai run output and usage into the neutral `RunResult` contract.

The shared server, ABI reader, CLI, network posture, auth behavior, and
conformance tests live in `runtimes/common`.

## Build and run

From the repository root:

```sh
make build-serve
make build-agentkit
make build-test-agent
make run-test-agent
```

The console script inside the adapter image is:

```sh
agentkit-serve --config /agent/agent.yaml
```

See `docs/runtime-adapters.md` and `docs/agent-abi.md` for the shared runtime
contract.
