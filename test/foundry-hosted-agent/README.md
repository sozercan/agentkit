# Foundry Hosted Agent smoke test

This fixture validates that a built AgentKit image can run behind the Microsoft
Foundry Hosted Agents **invocations** protocol.

Foundry containers use a different serving contract than AgentKit's native
OpenAI `/v1` façade: they listen on port `8088`, expose `/readiness`, and serve a
protocol endpoint such as `/invocations`. This fixture wraps a normal AgentKit
image with that protocol surface.

The smoke test intentionally uses an in-container OpenAI-compatible mock model at
`127.0.0.1:9000` so validation does not depend on external model credentials.

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

## Deploy to Foundry with azd

1. Push the wrapper image to a registry reachable by Foundry.
2. Scaffold an azd hosted-agent project with the `microsoft.foundry` extension.
3. Copy `foundry.agent.yaml.example` to that azd project's `agent.yaml` and set
   `image:` to the pushed image tag.
4. Run `azd provision` and `azd deploy`.
5. Invoke with the invocations protocol and the JSON body from `request.json`.

Example invocation after deployment:

```sh
azd ai agent invoke --protocol invocations --new-session \
  -f test/foundry-hosted-agent/request.json -o raw
```
