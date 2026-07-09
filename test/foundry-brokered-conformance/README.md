# Foundry brokered Responses conformance image

This fixture packages the Phase A0 SDK conformance app as a minimal Foundry
hosted-agent container. It is intentionally separate from production
`foundry.py`: its only job is to prove that a deployed Foundry hosted container
can return a Responses `function_call` item and later consume a matching
`function_call_output` continuation using SDK/platform response IDs.

The container entrypoint is:

```sh
agentkit-foundry-conformance --host 0.0.0.0 --port 8088
```

It serves:

- `GET /readiness`
- `POST /responses`

It does **not** accept request-level `tools`.

## Build locally

From the repository root:

```sh
docker buildx build --builder desktop-linux . \
  -f test/foundry-brokered-conformance/Dockerfile \
  --platform linux/amd64 \
  -t agentkit-foundry-brokered-conformance:test --load --provenance=false
```

## Validate locally

```sh
docker run --rm --platform linux/amd64 -p 127.0.0.1:18088:8088 \
  agentkit-foundry-brokered-conformance:test
```

In another terminal:

```sh
curl -fsS http://127.0.0.1:18088/readiness
curl -fsS -H 'content-type: application/json' \
  http://127.0.0.1:18088/responses \
  -d '{"input":"conformance_read"}'
```

The first `/responses` call should return one `function_call` named
`conformance_read` with `call_id: call_conformance_1` and a `caresp_...`
response id.

To exercise the full initial/continuation loop locally with the same transcript
helper used for live validation:

```sh
AGENT_RESPONSES_ENDPOINT=http://127.0.0.1:18088/responses \
AGENT_RESPONSES_BEARER_TOKEN=local-dummy-token \
deploy/foundry/scripts/foundry_brokered_conformance.sh conformance_read ./foundry-brokered-local-transcript
```

Or run the all-in-one local build/run/transcript smoke from the repository root:

```sh
deploy/foundry/scripts/local_brokered_conformance_container.sh \
  --fixture sdk \
  --platform linux/amd64 \
  --transcript-dir ./foundry-brokered-local-transcript
```

## Deploy to Foundry

1. Push the image to a registry reachable by Foundry.
2. Copy `foundry.agent.yaml.example` to the azd hosted-agent project as
   `agent.yaml` and set `image:` to the pushed tag.
3. Run `azd provision` and `azd deploy`.
4. Run the live transcript helper from the repository root:

```sh
export AGENT_RESPONSES_ENDPOINT="https://<hosted-agent>/responses"
# Optional: export AZURE_SUBSCRIPTION_ID="<subscription>" to select an account.
deploy/foundry/scripts/foundry_brokered_conformance.sh conformance_read ./foundry-brokered-transcript
```

The helper writes request/response JSON plus `summary.json`. Re-run `python3 deploy/foundry/scripts/verify_brokered_transcript.py <transcript-dir>` to verify an archived transcript later. Keep bearer tokens
out of transcripts.
