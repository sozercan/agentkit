# Foundry Hosted Agent smoke test

This fixture validates that a built AgentKit image can run behind the Microsoft
Foundry Hosted Agents **invocations** and minimal non-streaming **responses** protocols.

Foundry containers use a different serving contract than AgentKit's native
OpenAI `/v1` façade: they listen on port `8088`, expose `/readiness`, and serve
protocol endpoints such as `/invocations` or `/responses`. This fixture wraps a normal AgentKit
image with those protocol surfaces.

The default smoke test intentionally uses an in-container OpenAI-compatible mock
model at `127.0.0.1:9000` so validation does not depend on external model
credentials.

## Live wrapper

`foundry_live.py` is the non-mock variant used for real parity validation. It
loads the baked `/agent/agent.yaml`, validates required generic env declarations,
and exposes the same Foundry `/readiness`, `/invocations`, and non-streaming
`/responses` surfaces against the selected AgentKit runtime. Use this wrapper
when testing an AgentKit-built image against real model, search, memory, or
remote MCP resources. Keep provider-specific endpoints and role assignments in
deployment tooling rather than AgentKit core schema.

## Runtime lifecycle

The wrapper must mirror the native `agentkit-serve` lifecycle: it enters the
built AgentKit runtime session once during server startup and stores that running
runtime on `app.state`. This is important for real AgentKit images that declare
stdio MCP tools, because the adapter lifecycle starts tool subprocesses and keeps
them warm for request handling. Calling the underlying framework agent without an
entered runtime session only works for trivial no-tool smoke images and is not
equivalent to the native `/v1` server.

## Build locally

From the repository root, first build the local frontend and one AgentKit runtime
adapter. The commands below use the default pydantic-ai adapter because the
fixture is a protocol smoke test, not a runtime-specific behavior test:

```sh
make build-agentkit
make build-serve
```

To smoke the wrapper with another runtime, build that adapter instead and use the
matching `adapter=` build arg; for example LangGraph uses
`make build-serve-langgraph` and `--build-arg adapter=agentkit-serve-langgraph:test`.

Build the AgentKit base image from this fixture:

```sh
docker buildx build --builder desktop-linux . \
  -f test/foundry-hosted-agent/agentkitfile.yaml \
  --build-arg BUILDKIT_SYNTAX=agentkit:test \
  --build-arg adapter=agentkit-serve:test \
  --platform linux/amd64 \
  -t foundry-agentkit-base:test --load --provenance=false
```

Wrap it with the mock Foundry hosted-agent protocol adapter:

```sh
docker buildx build --builder desktop-linux test/foundry-hosted-agent \
  --platform linux/amd64 \
  --build-arg BASE_IMAGE=foundry-agentkit-base:test \
  -t agentkit-foundry-invocations:test --load --provenance=false
```

For live-resource validation, wrap the same base image with `foundry_live.py`
instead of the mock wrapper, for example:

```Dockerfile
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
COPY foundry_live.py /opt/agentkit/foundry_live.py
ENTRYPOINT ["/opt/agentkit/bin/python", "/opt/agentkit/foundry_live.py"]
```

## Validate locally

```sh
docker run --rm --platform linux/amd64 -p 127.0.0.1:18088:8088 \
  agentkit-foundry-invocations:test
```

In another terminal:

```sh
curl -fsS http://127.0.0.1:18088/readiness
curl -fsS -H 'content-type: application/json' \
  http://127.0.0.1:18088/invocations \
  -d @test/foundry-hosted-agent/request.json
curl -fsS -H 'content-type: application/json' \
  http://127.0.0.1:18088/responses \
  -d '{"input":"hello from Foundry hosted AgentKit JSON"}'
```

Expected response body:

```json
{
  "response": "AgentKit Foundry smoke OK: hello from Foundry hosted AgentKit JSON DONE_FOUNDRY_AGENTKIT_123",
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 7,
    "total_tokens": 12
  }
}
```

Use the same `request.json` body when invoking the hosted invocations protocol.
For Responses protocol checks, compare the extracted final assistant text rather
than generated response/message IDs or timestamps.

## Deploy to Foundry with azd

1. Push the wrapper image to a registry reachable by Foundry.
2. Scaffold an azd hosted-agent project with the `microsoft.foundry` extension.
3. Copy `foundry.agent.yaml.example` to that azd project's `agent.yaml` and set
   `image:` to the pushed image tag.
4. Run `azd provision` and `azd deploy`.
5. Invoke with the invocations protocol and the JSON body from `request.json`; optionally invoke the responses protocol with an `input` payload.

`azd ai agent invoke --protocol invocations "message"` may send a non-JSON raw
body depending on CLI behavior. Prefer `-f request.json` for this fixture so the
wrapper receives the expected `{"message": ...}` payload.

Example invocation after deployment:

```sh
azd ai agent invoke --protocol invocations --new-session \
  -f test/foundry-hosted-agent/request.json -o raw
```
