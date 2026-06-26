# agentkit-serve-common

The framework-neutral core shared by AgentKit's runtime adapters
(`agentkit-serve`, `agentkit-serve-maf`, …). It contains everything that does
**not** depend on a specific agent framework:

- `config.py` — the strict loader for the frozen `/agent/agent.yaml` ABI
  (`docs/agent-abi.md`) plus secret-free required-env validation.
- `server.py` — the non-streaming OpenAI `/v1` Chat-Completions facade (the 400
  guards, the Bearer auth gate, the single-`chat.completion` assembly).
- `cli.py` — the CLI entry point + network posture (loopback default; `0.0.0.0`
  requires `AGENTKIT_AUTH_TOKEN`).
- `runtime.py` — the **neutral run contract**: `RunResult`, `AgentRunError`,
  `RuntimeSession`, and the `RuntimeFactory` protocol each adapter implements.

## The seam

`server.create_app(spec, factory, auth_token)` is handed the adapter's
framework-specific `agent_factory` module (which satisfies `RuntimeFactory`). The
server uses only `factory.build_runtime(spec)` and the returned
`RuntimeSession.run(request) -> RunResult`, so it imports **no** agent framework
and never touches raw framework agent lifecycle. This is what keeps the lock-in
boundary (plan §12) confined to
each adapter's `agent_factory.py`.

## Adding a runtime adapter

A new single-agent adapter is just:

1. `agent_factory.py` implementing `build_runtime` (with any private framework
   helpers it needs; this is the only file that imports the framework), and
2. a thin `__main__.py` that calls `agentkit_serve_common.cli.run(agent_factory)`.

This package is imported (as a path dependency) by each adapter and shipped inside
each adapter image — but the adapters remain **separate images** with disjoint
framework deps, which is what physically guarantees the lock-in boundary.
