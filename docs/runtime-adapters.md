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
| `cli.py` | `agentkit-serve --config ...`, bind/port handling, auth startup gate. |
| `server.py` | FastAPI app and OpenAI-compatible response/error envelopes. |
| `conversation.py` | OpenAI message normalization into `RunRequest`. |
| `runtime.py` | `RuntimeFactory`, `RuntimeSession`, `RunResult`, `AgentRunError`. |
| `adapter_support.py` | API-key lookup, tool env projection, timeout parsing, error normalization. |
| `conformance.py` | Shared HTTP behavior tests adapter packages import. |

`server.create_app(spec, factory, auth_token)` receives an adapter module that
satisfies `RuntimeFactory`. The server calls only `factory.build_runtime(spec)`
and `RuntimeSession.run(request)`, so it never imports pydantic-ai, Microsoft
Agent Framework, LangChain, OpenAI SDK types, Azure, or Foundry packages.

## HTTP surface

All adapters serve the same endpoints:

- `GET /healthz` returns `{"status":"ok"}` and is always open.
- `GET /v1/models` returns the one configured model name.
- `POST /v1/chat/completions` runs the agent once and returns one
  `chat.completion` object with a single assistant message.

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
  `RunRequest` for runtime/session correlation.

Framework/model failures are normalized to an OpenAI-shaped error envelope with
`type: agent_error`. The adapters preserve upstream HTTP status codes when the
framework exposes them.

## Model endpoint compatibility

Adapters use the baked `model.baseURL` and `model.name` to construct their
OpenAI-compatible chat client. They do not special-case a provider: the endpoint
can be OpenAI, another hosted provider, a local gateway, an in-cluster service,
or a prebuilt or custom [AIKit](https://github.com/kaito-project/aikit) model
image. AIKit is just an example of an OpenAI-compatible endpoint. For no-auth
endpoints, omit `model.apiKeyEnv` unless you place an auth proxy in front of the
endpoint, and make sure the generated AgentKit container can resolve the
configured `baseURL` at runtime.

## Network posture

The generated image defaults to `AGENTKIT_BIND=127.0.0.1`. At runtime:

- loopback binds need no token,
- non-loopback binds such as `0.0.0.0` require `AGENTKIT_AUTH_TOKEN`, and
- when a token is set, `/v1/*` requires `Authorization: Bearer <token>`.

`/healthz` is intentionally unauthenticated so container platforms can probe the
service.

## Tool lifecycle and env projection

Tools are MCP servers declared in the ABI. Stdio tools use `name`, `command`,
and an `env` allowlist; remote tools use `type: mcp`, `transport:
streamable-http`, `urlEnv`, optional headers, and generic auth. Adapter
factories are responsible for turning each tool spec into their framework's MCP
integration.

Shared invariants:

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
