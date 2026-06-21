# agentkit-serve-maf

The AgentKit runtime adapter backed by the
[Microsoft Agent Framework](https://github.com/microsoft/agent-framework) (MAF).
It loads the frozen `/agent/agent.yaml` ABI (`docs/agent-abi.md`) and serves a
**non-streaming** OpenAI Chat-Completions facade (`POST /v1/chat/completions`)
whose tools are stdio MCP servers — byte-for-byte ABI-compatible with the
default `agentkit-serve` (pydantic-ai) adapter.

This package is also published as an **adapter image** used as the LLB base by
the AgentKit Go converter. Select it from an agentkitfile with:

```yaml
runtime: microsoft-agent-framework   # alias: maf
```

```
agentkit-serve --config /agent/agent.yaml
```

## What is and isn't framework-specific

Only `agent_factory.py` imports MAF. `config.py`, `server.py`, and `__main__.py`
are framework-agnostic (identical in shape to the pydantic-ai adapter): the run is
driven through `agent_factory.run_agent`, which returns a neutral `RunResult`. This
is the seam that lets the common core be shared across runtimes later.

## Lock-in boundary (plan §12)

`agent_factory.py` imports **only** `agent_framework` (core) and
`agent_framework.openai` — never an Azure / Foundry / CopilotStudio package,
including the first-party submodules `agent_framework.azure` / `.foundry` /
`.microsoft`. This is enforced by an AST-based check in `tests/test_guardrails.py`
(a naive source grep would false-positive on comments). The MCP SDK (`mcp`) is
declared as a direct dependency because MAF only bundles it via the heavy `[all]`
extra (which would cross that boundary).

See `docs/agent-abi.md` for the writer/reader contract.
