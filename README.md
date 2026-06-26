# AgentKit

AgentKit builds an agent from YAML into a normal OCI container image. The
container serves an OpenAI-compatible `/v1` Chat Completions API, can own stdio
MCP tools, and keeps secret values out of the image.

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

## Add MCP tools

Declare stdio MCP servers in `tools:`. Tool commands are started by the runtime
and are available to the agent, not supplied by each API request.

```yaml
tools:
  - name: fetch
    command: ["uvx", "mcp-server-fetch"]
    env: ["FETCH_TIMEOUT"]
```

`env` entries are env var names that may be passed into that tool subprocess.
AgentKit does not pass the whole container environment to every tool.

Cold `uvx` or `npx` tools may download packages before speaking MCP, so first
boot can be slower than later boots. Tune the tool initialization timeout with:

```sh
docker run -e AGENTKIT_MCP_TIMEOUT=180 ...
```

See [`docs/agentkitfile.md`](docs/agentkitfile.md) for the full Agentkitfile
schema.

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
OpenAI-compatible API. See [`docs/runtime-adapters.md`](docs/runtime-adapters.md)
for runtime behavior and adapter details.

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

- `apiKeyEnv` and tool `env` values are env var names, not secret values.
- `expose.openai` must be `true`; `expose.port` defaults to `8080`.

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
- [`docs/runtime-adapters.md`](docs/runtime-adapters.md) — runtime behavior,
  adapters, auth, request handling, and tool lifecycle.
- [`docs/agent-abi.md`](docs/agent-abi.md) — built `/agent/agent.yaml` contract.
- [`docs/development.md`](docs/development.md) — local development and CI.
- [`docs/architecture.md`](docs/architecture.md) — codebase architecture map for
  contributors.
- [`test/foundry-hosted-agent/README.md`](test/foundry-hosted-agent/README.md) —
  Foundry Hosted Agents smoke-test wrapper.
