# Runtime capabilities

AgentKit keeps runtime feature support explicit. Runtime identity and capabilities
are declared in `pkg/agentkit/runtimes` and mirrored in `runtimes/catalog/*.yaml`;
the catalog consistency test ensures they stay in sync.

Capabilities are provider-neutral feature flags used by validation before an
image build. Unsupported requested features should fail clearly instead of
silently producing a degraded runtime image.

## Capability names

Current and reserved names:

- `stdio-mcp` — stdio MCP servers declared with `tools[].command`.
- `streamable-http-mcp` — remote MCP over Streamable HTTP.
- `foundry-invocations-protocol` — Foundry hosted-agent `/invocations` wrapper.
- `foundry-responses-protocol` — Foundry hosted-agent `/responses` wrapper.
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
and auth wiring.

## Current support

| Runtime | Capabilities |
|---|---|
| `pydantic-ai` | `stdio-mcp`, `streamable-http-mcp` |
| `microsoft-agent-framework` / `maf` | `stdio-mcp`, `streamable-http-mcp`, `workload-identity-token-auth`, `context-provider-skills`, `context-provider-search`, `context-provider-memory` |
| `langgraph` | `stdio-mcp`, `streamable-http-mcp` |

Context-provider schemas are capability-gated per runtime; the MAF adapter currently declares skills, search, and memory support. OTel export and tool approval schemas remain gated until a runtime declares support.
