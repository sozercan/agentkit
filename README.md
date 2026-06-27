# AgentKit

AgentKit builds an agent from YAML into a normal OCI container image. The
container serves an OpenAI-compatible `/v1` Chat Completions API, can own MCP
tools, and keeps secret values out of the image.

Use AgentKit when you want to package an agent the same way you package any other
container: build it with Docker, run it locally, push it to a registry, and deploy
it anywhere containers run.

> **Experimental:** AgentKit is still early. APIs, file formats, and runtime
> behavior may change as the project evolves. Feedback, issues, and PRs are
> welcome.

## Quick start

Create `agentkitfile.yaml`:

```yaml
#syntax=ghcr.io/sozercan/agentkit/agentkit:latest
apiVersion: v1alpha1
kind: Agent
metadata:
  name: url-summarizer
model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
  name: gpt-4o-mini
  apiKeyEnv: OPENAI_API_KEY        # env var name only; never the secret value
instructions: |
  Summarize any URL the user gives you in three bullet points.
expose:
  openai: true
```

Build and run it:

```sh
docker buildx build . -f agentkitfile.yaml -t url-summarizer:latest --load

docker run --rm \
  -p 127.0.0.1:8080:8080 \
  -e AGENTKIT_BIND=0.0.0.0 \
  -e AGENTKIT_AUTH_TOKEN=dev-token \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  url-summarizer:latest
```

Call the agent:

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'authorization: Bearer dev-token' \
  -H 'content-type: application/json' \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"https://example.com"}]}'
```

The image also exposes:

- `GET /healthz`
- `GET /v1/models`
- `POST /v1/chat/completions`

## Use any OpenAI-compatible model endpoint

AgentKit is model-endpoint agnostic. `model.baseURL` can point at any
OpenAI-compatible `/v1` endpoint: OpenAI, another hosted provider, a local
gateway, an in-cluster service, or a model image served by
[AIKit](https://github.com/kaito-project/aikit). AIKit is only one example.

For an AIKit example, run any AIKit image that exposes the OpenAI-compatible API
on a Docker network. This can be a prebuilt CPU/GPU image or a custom model image
you create with AIKit:

```sh
docker network create agentkit-local 2>/dev/null || true

docker run -d --rm \
  --name aikit-llama \
  --network agentkit-local \
  ghcr.io/kaito-project/aikit/llama3.2:1b
```

Then point the Agentkitfile at that service and use the model name exposed by the
endpoint. No-auth local endpoints do not need `apiKeyEnv` unless you add your own
auth layer:

```yaml
model:
  provider: openai-compatible
  baseURL: http://aikit-llama:8080/v1
  name: llama-3.2-1b-instruct
```

For any other endpoint, replace `baseURL` and `model.name` with the values for
that service. For another prebuilt or custom AIKit image, also replace the image
reference and container name.

Run the generated AgentKit container on the same Docker network so it can reach
`aikit-llama`. If AIKit is exposed through the host instead, use an address that
is reachable from inside the AgentKit container, such as
`http://host.docker.internal:<port>/v1` on Docker Desktop.

## Add MCP tools

Declare MCP servers in `tools:`. Tools are owned by the built agent, not supplied
by each API request.

Stdio MCP servers use `command`:

```yaml
tools:
  - name: fetch
    command: ["uvx", "mcp-server-fetch"]
    env: ["FETCH_TIMEOUT"]
```

`env` entries are env var names that may be passed into that tool subprocess.
AgentKit does not pass the whole container environment to every tool.

Remote MCP servers over Streamable HTTP use `urlEnv` plus optional headers and
generic auth:

```yaml
tools:
  - name: toolbox
    type: mcp
    transport: streamable-http
    urlEnv: TOOLBOX_ENDPOINT
    headers:
      - name: Foundry-Features
        value: Toolboxes=V1Preview
    auth:
      type: bearer
      tokenEnv: TOOLBOX_TOKEN
```

Static credential headers such as `Authorization`, `Cookie`, or `X-API-Key` are
rejected; use `valueEnv` or `auth` so secrets are injected at runtime.

The `microsoft-agent-framework` runtime also supports
`auth.type: workload-identity-token` with an opaque `audience` when the
deployment environment provides `AGENTKIT_WORKLOAD_IDENTITY_TOKEN`,
`AGENTKIT_WORKLOAD_IDENTITY_TOKEN_COMMAND`, or an installed credential provider.

Cold `uvx` or `npx` tools may download packages before speaking MCP, so first
boot can be slower than later boots. Tune the tool initialization timeout with:

```sh
docker run -e AGENTKIT_MCP_TIMEOUT=180 ...
```

See [`docs/agentkitfile.md`](docs/agentkitfile.md) for the full Agentkitfile
schema.

## Add context providers

Context providers are provider-neutral and capability-gated by runtime. The
`microsoft-agent-framework` runtime currently supports filesystem/MCP skills,
Azure AI Search-style search providers, and external memory providers through
generic env names and auth declarations:

```yaml
context:
  providers:
    - name: knowledge
      type: search
      endpointEnv: SEARCH_ENDPOINT
      indexEnv: SEARCH_INDEX
      auth:
        type: workload-identity-token
        audience: https://search.azure.com/.default
    - name: user-memory
      type: memory
      endpointEnv: MEMORY_ENDPOINT
      storeNameEnv: MEMORY_STORE_NAME
      auth:
        type: workload-identity-token
        audience: https://ai.azure.com/.default
```

The deployment environment supplies the endpoint, index/store name, memory scope,
and identity material. Memory context providers require an explicit
`AGENTKIT_MEMORY_SCOPE` so durable memory is not accidentally shared across users
or sessions. Provider-specific provisioning remains in deployment profiles such
as `deploy/foundry/`; AgentKit core does not add keys like `foundry.memoryStore`.


## Declare runtime env requirements

Top-level `env:` entries declare runtime environment requirements by name only.
Required entries are checked at startup with secret-free errors.

```yaml
env:
  - name: REQUIRED_FOO
    required: true
  - name: OPTIONAL_BAR
```

## Choose a runtime

AgentKit files are runtime-neutral. Pick the agent framework with the optional
`runtime:` field:

| `runtime:` value | Framework |
|---|---|
| omitted / `pydantic-ai` | [pydantic-ai](https://ai.pydantic.dev) |
| `microsoft-agent-framework` / `maf` | [Microsoft Agent Framework](https://github.com/microsoft/agent-framework) |
| `langgraph` | [LangChain/LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) |

```yaml
runtime: langgraph
```

All runtimes read the same built agent config and serve the same non-streaming
OpenAI-compatible API. Runtime capabilities are explicit and validated before
build; see [`docs/runtime-capabilities.md`](docs/runtime-capabilities.md) and
[`docs/runtime-adapters.md`](docs/runtime-adapters.md).

## Configure the server

By default, generated images bind to `127.0.0.1` inside the container. If you bind
to a non-loopback address such as `0.0.0.0`, set `AGENTKIT_AUTH_TOKEN`; `/v1/*`
requests must then include `Authorization: Bearer <token>`.

```sh
docker run --rm \
  -p 8080:8080 \
  -e AGENTKIT_BIND=0.0.0.0 \
  -e AGENTKIT_AUTH_TOKEN="$AGENTKIT_AUTH_TOKEN" \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  url-summarizer:latest
```

`/healthz` remains unauthenticated for liveness checks.

## Agentkitfile basics

The most important rules are:

- `apiVersion: v1alpha1` and `kind: Agent` are required.
- Unknown YAML fields fail the build.
- `model.provider` must be `openai-compatible`.
- `instructions` can be inline text or a file source:

  ```yaml
  instructions:
    file: ./prompt.md
  ```

- `apiKeyEnv`, tool `env`, top-level `env`, and env-suffixed fields such as
  `urlEnv` or `valueEnv` are env var names, not secret values.
- `expose.openai` must be `true`; `expose.port` defaults to `8080`.
- Context provider, model workload-identity, OTel export, and tool
  approval schemas are capability-gated; log-level observability is reserved but
  rejected until a runtime wires it through. The MAF runtime currently declares
  skills, search, and memory context-provider support.

Full reference: [`docs/agentkitfile.md`](docs/agentkitfile.md).

## Develop AgentKit locally

Build the frontend image, the default runtime adapter image, and a test agent:

```sh
make build-agentkit
make build-serve
make build-test-agent
make run-test-agent
```

Build a test agent for another runtime:

```sh
make build-serve-maf
make build-test-agent RUNTIME=maf

make build-serve-langgraph
make build-test-agent RUNTIME=langgraph
```

See [`docs/development.md`](docs/development.md) for the full local test and CI
workflow.

## More docs

- [`docs/agentkitfile.md`](docs/agentkitfile.md) — Agentkitfile schema and build
  arguments.
- [`docs/runtime-capabilities.md`](docs/runtime-capabilities.md) — runtime feature
  capability names and current support.
- [`docs/runtime-adapters.md`](docs/runtime-adapters.md) — runtime behavior,
  adapters, auth, request handling, and tool lifecycle.
- [`docs/agent-abi.md`](docs/agent-abi.md) — built `/agent/agent.yaml` contract.
- [`docs/development.md`](docs/development.md) — local development and CI.
- [`docs/architecture.md`](docs/architecture.md) — codebase architecture map for
  contributors.
- [`deploy/foundry/README.md`](deploy/foundry/README.md) — Foundry deployment and
  resource helper scripts.
- [`test/foundry-hosted-agent/README.md`](test/foundry-hosted-agent/README.md) —
  Foundry Hosted Agents smoke-test wrapper.
