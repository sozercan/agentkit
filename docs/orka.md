# Register an AgentKit image with Orka

AgentKit images can expose observed-mode `orka.harness.v1` without rebuilding the
agent. Start the same image with Orka mode enabled:

```sh
docker run --rm \
  -e AGENTKIT_PROTOCOL=orka \
  -e AGENTKIT_AUTH_TOKEN=dev-token \
  -e AGENTKIT_BIND=0.0.0.0 \
  -p 127.0.0.1:8080:8080 \
  ghcr.io/acme/fibey@sha256:...
```

Open endpoints:

- `GET /v1/health`
- `GET /v1/capabilities`

Bearer-authenticated endpoints:

- `POST /v1/turns`
- `GET /v1/turns/{turnID}/events?afterSeq=...`
- `POST /v1/turns/{turnID}/cancel`
- `GET /v1/turns/{turnID}/output?ref=...` (reserved; returns 404 unless a future
  adapter stores large outputs by reference)

AgentKit maps one Orka turn to one `RuntimeSession.run(RunRequest)` call. It
emits `TurnStarted` and exactly one terminal frame (`TurnCompleted`,
`TurnFailed`, or `TurnCancelled`). AgentKit-owned tools and MCP servers continue
to execute inside the runtime; Orka observes the run and remains responsible for
policy, approvals, trust tiers, Tool CRDs, idempotency, and side-effect
governance.

## Render an AgentRuntime manifest

The current Orka `AgentRuntime` CRD supports external endpoints first. Deploy the
AgentKit image yourself (for example as a Kubernetes Deployment/Service) with:

- `AGENTKIT_PROTOCOL=orka`
- `AGENTKIT_AUTH_TOKEN` sourced from the same Secret referenced by
  `spec.clientAuth.bearerTokenSecretRef`

Then render the AgentRuntime registration for that endpoint:

```sh
agentkit render --target orka-agentruntime \
  --external-endpoint http://fibey-agentkit.default.svc.cluster.local:8080 \
  --name fibey-agentkit
```

Output shape:

```yaml
apiVersion: core.orka.ai/v1alpha1
kind: AgentRuntime
metadata:
  name: fibey-agentkit
spec:
  contractVersion: orka.harness.v1
  deployment:
    mode: external-endpoint
    endpoint: http://fibey-agentkit.default.svc.cluster.local:8080
  clientAuth:
    bearerTokenSecretRef:
      name: fibey-agentkit-harness-token
      key: token
  capabilities:
    toolExecutionModes:
      - observed
    supportsCancel: true
    supportsRuntimeSessions: true
```

Use `--auth-secret-name` and `--auth-secret-key` when your client-auth Secret
uses a different name or key. The `--image` flag is reserved for a future Orka
managed-image CRD mode and fails clearly against the current external-endpoint
schema.
