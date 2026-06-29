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
emits Orka-native `HarnessEventFrame` SSE frames and exactly one terminal frame
(`TurnCompleted`, `TurnFailed`, or `TurnCancelled`). AgentKit-owned tools and MCP
servers continue to execute inside the runtime; Orka observes the run and remains
responsible for policy, approvals, trust tiers, Tool CRDs, idempotency, and
side-effect governance.

## Orka wire contract

`GET /v1/health` returns Orka `HealthResponse`:

```json
{
  "version": "orka.harness.v1",
  "status": "ok",
  "ready": true,
  "checkedAt": "2026-06-27T00:00:00Z",
  "metadata": {"agentName": "fibey-agentkit"}
}
```

`GET /v1/capabilities` returns flat Orka `CapabilitiesResponse` fields:

```json
{
  "version": "orka.harness.v1",
  "protocolVersion": "orka.harness.v1",
  "transport": "http+sse",
  "runtimeName": "agentkit-serve",
  "runtimeVersion": "0.0.0",
  "providerKind": "kubernetes-service",
  "toolExecutionModes": ["observed"],
  "supportsCancel": true,
  "supportsRuntimeSessions": true,
  "supportsSuspend": false,
  "supportsWorkspaceSnapshot": false,
  "maxConcurrentTurns": 1,
  "metadata": {
    "agentName": "fibey-agentkit",
    "model": "gpt-4o-mini",
    "agentkitProvider": "openai-compatible"
  }
}
```

Start a turn with Orka `StartTurnRequest`:

```json
{
  "version": "orka.harness.v1",
  "namespace": "default",
  "taskName": "fibey-task",
  "sessionName": "fibey-session",
  "runtimeSessionID": "runtime-session-1",
  "turnID": "turn-1",
  "correlationID": "corr-1",
  "deadline": "2026-06-27T00:05:00Z",
  "authIdentity": {"subject": "system:serviceaccount:default:orka"},
  "input": {
    "prompt": "Investigate alert A-123",
    "contextRefs": [],
    "env": [
      {"name": "FOO", "value": "BAR"}
    ]
  },
  "toolExecutionMode": "observed",
  "metadata": {}
}
```

AgentKit responds with Orka `StartTurnResponse`:

```json
{
  "version": "orka.harness.v1",
  "accepted": true,
  "runtimeSessionID": "runtime-session-1",
  "turnID": "turn-1",
  "correlationID": "corr-1",
  "eventStreamPath": "/v1/turns/turn-1/events"
}
```

`input.contextRefs` are accepted as safe Orka references. AgentKit does not fetch
Orka-owned context objects in observed mode yet; instead it forwards the reference
list to runtime adapters only in `RunRequest.metadata["contextRefs"]`. It does not
promote request-controlled references into model prompt or system history.

SSE `data:` payloads are Orka `HarnessEventFrame` objects. A successful observed
turn normally emits `TurnStarted`, `RuntimeOutput`, then `TurnCompleted`:

```json
{
  "version": "orka.harness.v1",
  "type": "TurnCompleted",
  "runtimeSessionID": "runtime-session-1",
  "turnID": "turn-1",
  "correlationID": "corr-1",
  "seq": 3,
  "createdAt": "2026-06-27T00:00:02Z",
  "severity": "info",
  "summary": "turn completed",
  "content": {},
  "contentText": "",
  "completed": {
    "result": "assistant response text",
    "finalEventSeq": 3
  },
  "failed": null,
  "error": null,
  "metadata": {}
}
```

Cancel with Orka `CancelTurnRequest`:

```json
{
  "version": "orka.harness.v1",
  "namespace": "default",
  "taskName": "fibey-task",
  "sessionName": "fibey-session",
  "runtimeSessionID": "runtime-session-1",
  "turnID": "turn-1",
  "correlationID": "corr-1",
  "reason": "user requested cancel"
}
```

AgentKit responds with Orka `CancelTurnResponse`:

```json
{
  "version": "orka.harness.v1",
  "accepted": true,
  "runtimeSessionID": "runtime-session-1",
  "turnID": "turn-1",
  "correlationID": "corr-1",
  "message": "cancel accepted"
}
```

## Render an AgentRuntime manifest

The current Orka `AgentRuntime` CRD supports external endpoints first. Deploy the
AgentKit image yourself (for example as a Kubernetes Deployment/Service) with:

- `AGENTKIT_PROTOCOL=orka`
- `AGENTKIT_BIND=0.0.0.0` so the Kubernetes Service can reach the harness outside
  the container network namespace
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
