# Runtime adapters

Runtime adapters execute the same baked `/agent/agent.yaml` contract through
different Python agent frameworks. They share one framework-neutral server core
and differ only at the `agent_factory.py` boundary.

## Shared runtime core

The `agentkit-serve-common` package under `runtimes/common/` owns behavior that
must be identical across adapters:

| Module | Responsibility |
|---|---|
| `config.py` | Strict `/agent/agent.yaml` reader and ABI version check. |
| `cli.py` | `agentkit-serve --config ... --protocol openai\|foundry\|orka`, bind/port handling, auth startup gates. |
| `server.py` | FastAPI app and OpenAI-compatible response/error envelopes. |
| `foundry.py` | Foundry `/readiness`, `/invocations`, and minimal `/responses` skin. |
| `orka.py` | Observed-mode `orka.harness.v1` HTTP+SSE skin. |
| `conversation.py` | Protocol request normalization into `RunRequest`. |
| `runtime.py` | `RuntimeFactory`, `RuntimeSession`, `RunResult`, `AgentRunError`. |
| `adapter_support.py` | API-key lookup, tool env projection, timeout parsing, error normalization. |
| `conformance.py` | Shared HTTP behavior tests adapter packages import. |

The protocol app factories receive an adapter module that satisfies
`RuntimeFactory`. The shared core calls only `factory.build_runtime(spec)` and
`RuntimeSession.run(request)`, so it never imports pydantic-ai, Microsoft Agent
Framework, LangChain, OpenAI SDK types, Azure, Foundry SDKs, or Orka controllers.

## HTTP surface

All adapters can serve the same selected protocol surface. `openai` is the
default. `foundry` and `orka` are selected with `--protocol` or
`AGENTKIT_PROTOCOL`.

OpenAI mode exposes:

- `GET /healthz` returns `{"status":"ok"}` and is always open.
- `GET /v1/models` returns the one configured model name.
- `POST /v1/chat/completions` runs the agent once and returns one
  `chat.completion` object with a single assistant message.

Foundry mode exposes `/readiness`, `/invocations`, and synchronous
`/responses`. It defaults to port `8088` when the ABI kept the generic default
port; generated images expose both `8080` and `8088` in OCI metadata for that
case. Orka mode exposes `orka.harness.v1` health, capabilities, turn
acceptance, SSE replay, and cancel endpoints.

Request behavior is intentionally narrow:

- `stream: true` returns HTTP 400 with code `stream_unsupported`.
- non-empty `tools` returns HTTP 400 with code `tools_unsupported`.
- `tool_choice` values other than missing, empty, `none`, or `auto` return HTTP
  400 with code `tool_choice_unsupported`.
- the final message must have role `user`.
- prior `system`, `user`, and `assistant` messages become history.
- prior `tool` and unknown roles are ignored because the built agent owns its
  tools.
- `X-AgentKit-Session-Id`, when present, is forwarded through the neutral
  `RunRequest` for runtime/session correlation. Orka mode additionally forwards
  `turn_id`, `correlation_id`, `deadline`, `metadata`, and per-run `env` fields.

Framework/model failures are normalized to an OpenAI-shaped error envelope with
`type: agent_error`. The adapters preserve upstream HTTP status codes when the
framework exposes them.

## Network posture

The generated image defaults to `AGENTKIT_BIND=127.0.0.1`. At runtime:

- loopback binds need no token except in Orka mode,
- non-loopback binds such as `0.0.0.0` require `AGENTKIT_AUTH_TOKEN`, and
- when a token is set, protected endpoints require
  `Authorization: Bearer <token>`.

OpenAI `/healthz` and Orka `/v1/health` and `/v1/capabilities` are intentionally
unauthenticated so container platforms and orchestrators can probe/discover the
service. Orka turn, event, cancel, and output endpoints always require a token.

## Tool lifecycle and env projection

Tools are MCP servers declared in the ABI. Stdio tools use `name`, `command`,
and an `env` allowlist; remote tools use `type: mcp`, `transport:
streamable-http`, `urlEnv`, optional headers, and generic auth. Adapter
factories are responsible for turning each tool spec into their framework's MCP
integration.

Shared invariants:

Per-run env supplied by Orka is forwarded in `RunRequest.env` and helper functions
can resolve credentials from that mapping before falling back to process env.
Startup-scoped model clients and long-lived MCP sessions still resolve their own
startup credentials at runtime initialization; they are not rebuilt for every turn.

- a missing or empty command fails before serving,
- `AGENTKIT_MCP_TIMEOUT` controls MCP initialization timeout,
- each tool subprocess receives only env vars declared in that tool's `env`,
- undeclared `${VAR}` interpolation inside a declared env value is rejected, and
- tool sessions are entered once for the app lifespan and reused across requests,
- remote MCP clients inject headers only for the configured origin and do not
  follow redirects with credentials.

## Adapter packages

### pydantic-ai

Path: `runtimes/pydantic-ai/`

- Console script package name: `agentkit-serve`.
- Adapter image target built by `make build-serve`.
- Uses `OpenAIChatModel` and `OpenAIProvider`.
- Supports both older `MCPServerStdio` and newer `MCPToolset` /
  `StdioTransport` APIs.
- Maps pydantic-ai message history and usage objects into the neutral contract.

### Microsoft Agent Framework

Path: `runtimes/microsoft-agent-framework/`

- Console script package name: `agentkit-serve-maf`.
- Adapter image target built by `make build-serve-maf`.
- Runtime selector: `microsoft-agent-framework` or alias `maf`.
- Depends on the bounded MAF core/OpenAI packages, the MCP SDK, and provider
  adapters needed by generic AgentKit capabilities such as workload-identity
  model auth, Azure AI Search context, and external memory.
- Guardrail tests prevent unrelated cloud packages such as CopilotStudio/Purview
  from crossing the adapter boundary.
- Supports session-aware runs, remote MCP, filesystem/MCP skills, search context,
  and memory context through generic ABI fields.

### LangGraph

Path: `runtimes/langgraph/`

- Console script package name: `agentkit-serve`.
- Adapter image target built by `make build-serve-langgraph`.
- Runtime selector: `langgraph`.
- Uses LangChain OpenAI chat models, LangGraph, `langchain-mcp-adapters`, and
  persistent MCP sessions.
- Aggregates token usage from every AI message in a tool-using graph run.
- Guardrail tests keep Azure and Foundry packages out of the generic adapter.

## Adding an adapter

To add a single-agent runtime:

1. create a new adapter package with `agentkit_serve/__main__.py` that calls
   `agentkit_serve_common.cli.run(agent_factory)`,
2. implement `agent_factory.build_runtime(spec) -> RuntimeSession`,
3. add an adapter Dockerfile that installs `runtimes/common` before the adapter,
4. add a `RuntimeSpec` in `pkg/agentkit/runtimes/catalog.go`,
5. add the matching `runtimes/catalog/*.yaml` entry and tests/fixtures, and
6. import the shared conformance tests in the adapter's test suite.

No shared server changes should be necessary when the adapter can satisfy the
neutral `RuntimeSession` contract.
