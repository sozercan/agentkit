# Foundry Hosted Agent smoke test

This fixture validates that a built AgentKit image can run behind the Microsoft
Foundry Hosted Agents **invocations** protocol.

Foundry containers use a different serving contract than AgentKit's native
OpenAI `/v1` façade: they listen on port `8088`, expose `/readiness`, and serve a
protocol endpoint such as `/invocations`. This fixture wraps a normal AgentKit
image with that protocol surface.

The smoke test intentionally uses an in-container OpenAI-compatible mock model at
`127.0.0.1:9000` so validation does not depend on external model credentials.

## Runtime lifecycle

The wrapper must mirror the native `agentkit-serve` lifecycle: it enters the
built AgentKit agent's async context once during server startup and stores that
running agent on `app.state`. This is important for real AgentKit images that
declare stdio MCP tools, because the adapter lifecycle starts tool subprocesses
and keeps them warm for request handling. Calling `run_agent` on an un-entered
agent only works for trivial no-tool smoke images and is not equivalent to the
native `/v1` server.

## Build locally

From the repository root, first build the local frontend and pydantic-ai adapter:

```sh
make build-agentkit
make build-serve
```

Build the AgentKit base image from this fixture:

```sh
docker buildx build --builder desktop-linux . \
  -f test/foundry-hosted-agent/agentkitfile.yaml \
  --build-arg BUILDKIT_SYNTAX=agentkit:test \
  --build-arg adapter=agentkit-serve:test \
  --platform linux/amd64 \
  -t foundry-agentkit-base:test --load --provenance=false
```

Wrap it with the Foundry invocations protocol:

```sh
docker buildx build --builder desktop-linux test/foundry-hosted-agent \
  --platform linux/amd64 \
  --build-arg BASE_IMAGE=foundry-agentkit-base:test \
  -t agentkit-foundry-invocations:test --load --provenance=false
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

Use the same `request.json` body when invoking the hosted agent. The local and
Foundry-hosted JSON response bodies should be identical; platform HTTP headers
will differ.

## Deploy to Foundry with azd

1. Push the wrapper image to a registry reachable by Foundry.
2. Scaffold an azd hosted-agent project with the `microsoft.foundry` extension.
3. Copy `foundry.agent.yaml.example` to that azd project's `agent.yaml` and set
   `image:` to the pushed image tag.
4. Run `azd provision` and `azd deploy`.
5. Invoke with the invocations protocol and the JSON body from `request.json`.

`azd ai agent invoke --protocol invocations "message"` may send a non-JSON raw
body depending on CLI behavior. Prefer `-f request.json` for this fixture so the
wrapper receives the expected `{"message": ...}` payload.

Example invocation after deployment:

```sh
azd ai agent invoke --protocol invocations --new-session \
  -f test/foundry-hosted-agent/request.json -o raw
```
