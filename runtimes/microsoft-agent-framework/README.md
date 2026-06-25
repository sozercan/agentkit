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

This adapter ships ONLY `agent_factory.py` (which imports MAF) plus a thin
`__main__.py`. The framework-neutral core — the `/agent/agent.yaml` ABI loader, the
OpenAI `/v1` facade, and the CLI/network posture — lives in the shared
`agentkit-serve-common` package. `agent_factory` satisfies that package's
`RuntimeFactory` protocol (`build_runtime` → `RuntimeSession.run` → a neutral
`RunResult`), which is the seam that keeps the shared core framework-agnostic.


## Lock-in boundary (plan §12)

`agent_factory.py` imports **only** `agent_framework` (core) and
`agent_framework.openai` — never an Azure / Foundry / CopilotStudio package,
including the first-party submodules `agent_framework.azure` / `.foundry` /
`.microsoft`. This is enforced by an AST-based check in `tests/test_guardrails.py`
(a naive source grep would false-positive on comments). The MCP SDK (`mcp`) is
declared as a direct dependency because MAF only bundles it via the heavy `[all]`
extra (which would cross that boundary).

See `docs/agent-abi.md` for the writer/reader contract.
