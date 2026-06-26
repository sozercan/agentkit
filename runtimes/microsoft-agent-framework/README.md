# AgentKit Microsoft Agent Framework runtime adapter

`runtimes/microsoft-agent-framework` builds the AgentKit runtime adapter backed
by Microsoft Agent Framework (MAF). It consumes the same `/agent/agent.yaml` ABI
as the other adapters and serves the same non-streaming OpenAI-compatible
`/v1/chat/completions` façade.

Select it from an Agentkitfile with either spelling:

```yaml
runtime: microsoft-agent-framework
# or
runtime: maf
```

## Responsibilities

- Build the MAF chat client from the OpenAI-compatible model settings in the ABI.
- Attach stdio MCP tools declared in the ABI.
- Preserve the shared secret-hygiene rules: model API keys are read from the
  declared env var, and tool subprocesses receive only declared env vars.
- Map MAF run results and wrapped framework/model errors into the neutral
  `RunResult` / `AgentRunError` contract.

The shared server, ABI reader, CLI, network posture, auth behavior, and
conformance tests live in `runtimes/common`.

## Dependency boundary

This generic adapter depends on the minimal MAF core/OpenAI packages plus the MCP
SDK. It intentionally does not import Azure, Foundry, or CopilotStudio packages.
AST-based guardrail tests in `tests/test_guardrails.py` enforce that boundary.

## Build and run

From the repository root:

```sh
make build-serve-maf
make build-agentkit
make build-test-agent RUNTIME=maf
```

The console script inside the adapter image is:

```sh
agentkit-serve --config /agent/agent.yaml
```

See `docs/runtime-adapters.md` and `docs/agent-abi.md` for the shared runtime
contract.
