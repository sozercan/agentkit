# agentkit-serve-common

`agentkit-serve-common` is the framework-neutral Python core shared by all
AgentKit runtime adapters. It contains the runtime behavior that must be
identical whether the selected adapter is pydantic-ai, Microsoft Agent Framework,
or LangGraph.

## Modules

- `config.py` — strict `/agent/agent.yaml` ABI loader, version check, env
  requirement validation, and provider-neutral schema validation.
- `cli.py` — `agentkit-serve --config ...`, bind/port handling, and the startup
  auth gate for non-loopback binds.
- `server.py` — FastAPI app for `/healthz`, `/v1/models`, and
  `/v1/chat/completions`.
- `foundry.py` — reusable Foundry Hosted Agent protocol wrapper for
  `/readiness`, `/invocations`, and minimal non-streaming `/responses`.
- `conversation.py` — OpenAI message normalization into a framework-neutral
  `RunRequest`.
- `runtime.py` — `RuntimeFactory`, `RuntimeSession`, `RunResult`, and
  `AgentRunError`.
- `adapter_support.py` — API-key resolution, declared-only tool env projection,
  remote MCP URL/header/auth resolution, MCP HTTP client factories, MCP timeout
  parsing, and framework exception normalization.
- `conformance.py` — shared HTTP behavior tests imported by adapter test suites.

## Adapter seam

`server.create_app(spec, factory, auth_token)` and
`foundry.create_foundry_app(spec, factory)` receive an adapter module that
satisfies `RuntimeFactory`. The shared core calls only `factory.build_runtime(spec)`
and `RuntimeSession.run(request) -> RunResult`. It never imports framework
packages or touches raw framework agent lifecycle.

This keeps framework dependency lock-in inside each adapter's `agent_factory.py`.

## Adding a runtime adapter

A new single-agent adapter should provide:

1. `agent_factory.py` implementing `build_runtime(spec) -> RuntimeSession`,
2. a thin `__main__.py` that calls `agentkit_serve_common.cli.run(agent_factory)`,
3. adapter tests that import the shared conformance checks, and
4. an adapter image that installs this common package before the adapter package.

Adapters remain separate images with separate framework dependencies, while this
package is installed into each image as the shared façade/runtime core.
