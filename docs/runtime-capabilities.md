# Runtime capabilities

AgentKit keeps runtime feature support explicit. Runtime identity and capabilities
are declared in `pkg/agentkit/runtimes` and mirrored in `runtimes/catalog/*.yaml`;
the catalog consistency test ensures they stay in sync.

Capabilities are feature/protocol flags used by validation, image metadata, and
orchestrator registration before an image build. Unsupported requested features
should fail clearly instead of silently producing a degraded runtime image.

## Capability names

Current and reserved names:

- `stdio-mcp` — stdio MCP servers declared with `tools[].command`.
- `streamable-http-mcp` — remote MCP over Streamable HTTP.
- `foundry-invocations-protocol` — Foundry hosted-agent `/readiness` +
  `/invocations` wrapper.
- `foundry-responses-minimal` — current Foundry `/responses` wrapper: synchronous
  and non-streaming. Do not treat this as full Responses parity for background,
  streaming, polling, cancel, or durable response IDs.
- `orka-harness-v1` — observed-mode `orka.harness.v1` over HTTP+SSE.
- `orka-observed-tools` — AgentKit-owned tools/MCP execute inside the runtime;
  Orka observes lifecycle/output frames and governs externally.
- `orka-brokered-tools` — reserved for a future mode where Orka brokers tool
  execution and approval frames. Not advertised by default.
- `filesystem-skills` — local filesystem skill sources.
- `mcp-skills` — MCP-backed skill sources.
- `context-provider-search` — external search/RAG context provider.
- `context-provider-memory` — external memory context provider.
- `context-provider-skills` — skills represented through a context-provider ABI.
- `workload-identity-token-auth` — workload identity tokens for tools/resources.
- `model-workload-identity-auth` — workload identity tokens for model calls.
- `otel-export` — OpenTelemetry export support.
- `tool-approval` — tool approval / human-in-the-loop policy support.

Avoid provider-specific resource names such as `foundry-toolbox`; those should map
to generic capabilities such as `streamable-http-mcp` plus deployment-profile env
and auth wiring. Orka-specific names here describe the protocol contract AgentKit
exposes; Orka remains responsible for policy, approval, idempotency, and
side-effect governance.

## Current support

| Runtime | Capabilities |
|---|---|
| `pydantic-ai` | `stdio-mcp`, `streamable-http-mcp`, `foundry-invocations-protocol`, `foundry-responses-minimal`, `orka-harness-v1`, `orka-observed-tools` |
| `microsoft-agent-framework` / `maf` | `stdio-mcp`, `streamable-http-mcp`, `foundry-invocations-protocol`, `foundry-responses-minimal`, `orka-harness-v1`, `orka-observed-tools`, `workload-identity-token-auth`, `model-workload-identity-auth`, `context-provider-skills`, `filesystem-skills`, `mcp-skills`, `context-provider-search`, `context-provider-memory` |
| `langgraph` | `stdio-mcp`, `streamable-http-mcp`, `foundry-invocations-protocol`, `foundry-responses-minimal`, `orka-harness-v1`, `orka-observed-tools` |

Context-provider schemas are capability-gated per runtime; the MAF adapter
currently declares skills, search, and memory support. OTel export, local tool
approval enforcement, and Orka brokered-tool mode remain gated until a runtime and
protocol contract declare support.
