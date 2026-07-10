# Production AgentKit Foundry brokered-only fixture

This fixture packages the shared production `create_foundry_app` brokered path,
not the separate SDK conformance spike app. It uses a static `/agent/agent.yaml`
with a safe `conformance_read` brokered schema and the brokered-only entrypoint:

```sh
agentkit-foundry-brokered --config /agent/agent.yaml --host 0.0.0.0 --port 8088
```

Set `AGENTKIT_FOUNDRY_BROKERED_CONTINUATION_PROOF` at runtime so transcript
validation can exercise the Orka-only continuation proof. AgentKit accepts the
proof from the compatibility header or top-level JSON body field. For real
deployments, inject this value from runtime configuration or a secret.

Build and validate locally:

```sh
docker build . -f test/foundry-brokered-agentkit/Dockerfile \
  -t agentkit-foundry-brokered:local

docker run --rm \
  -e AGENTKIT_AUTH_TOKEN=local-dummy-token \
  -e AGENTKIT_FOUNDRY_BROKERED_CONTINUATION_PROOF=local-dev-proof \
  -p 127.0.0.1:18092:8088 \
  agentkit-foundry-brokered:local
```

Then run:

```sh
AGENT_RESPONSES_ENDPOINT=http://127.0.0.1:18092/responses \
AGENT_RESPONSES_BEARER_TOKEN=local-dummy-token \
AGENTKIT_CONTINUATION_PROOF=local-dev-proof \
AGENTKIT_EXPECTED_CALL_ID=auto \
AGENTKIT_EXPECTED_CALL_ID_PREFIX=call_ \
deploy/foundry/scripts/foundry_brokered_conformance.sh \
  conformance_read ./foundry-brokered-agentkit-transcript
```


Or run the all-in-one local build/run/transcript smoke from the repository root:

```sh
deploy/foundry/scripts/local_brokered_conformance_container.sh \
  --fixture agentkit \
  --platform linux/amd64 \
  --transcript-dir ./foundry-brokered-agentkit-transcript
```
