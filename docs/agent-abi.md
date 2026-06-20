# The `agent.yaml` ABI (v0, FROZEN)

This file is the contract between the two halves of AgentKit:

- **Writer** — the Go frontend (`pkg/agentkit2llb/agent`) renders this file and
  bakes it at `/agent/agent.yaml` in the built image.
- **Reader** — `agentkit-serve` (Python) loads it at startup and serves the agent.

Neither side may change this shape without updating the other. v0 keeps it
minimal; new fields are additive.

## Location

`/agent/agent.yaml` (constant `utils.AgentConfigPath`).

## Schema

```yaml
# schema/version of THIS file, not the agentkitfile apiVersion.
abiVersion: v0

metadata:
  name: url-summarizer          # agent name (from agentkitfile metadata.name)

model:
  provider: openai-compatible   # v0: always openai-compatible
  baseURL: https://api.openai.com/v1
  name: gpt-4o-mini
  apiKeyEnv: OPENAI_API_KEY      # NAME of env var; serve reads os.environ[apiKeyEnv]

instructions: |                  # fully-resolved system prompt (inline OR file contents)
  Summarize any URL the user gives you in three bullet points.

tools:                           # MCP servers, stdio transport (v0)
  - name: fetch
    command: ["npx", "-y", "@modelcontextprotocol/server-fetch"]
    env: ["FETCH_TIMEOUT"]       # NAMES only; serve passes ONLY these into the subprocess env

expose:
  openai: true
  port: 8080
```

## Reader (agentkit-serve) contract

1. Load and validate this file. On invalid/missing → exit non-zero with a clear error.
2. Construct an OpenAI-compatible model client pointed at `model.baseURL`, using
   model `model.name` and the API key from `os.environ[model.apiKeyEnv]`.
3. For each tool, spawn a stdio MCP subprocess from `command`, passing **only**
   the env vars NAMED in that tool's `env:` list (plan §10 secret-bleed rule) —
   never the full container environment.
4. Serve:
   - `POST /v1/chat/completions` — non-streaming Chat-Completions façade.
     - Reject `stream: true` → HTTP 400.
     - Reject non-empty `tools` / `tool_choice` in the request → HTTP 400
       ("this agent owns its tools").
     - Drive the agent loop; collapse intermediate MCP tool calls to a single
       assistant message with `finish_reason: "stop"`.
   - `GET /v1/models` — optional SDK-compatibility listing (returns `model.name`).
   - `GET /healthz` — liveness.
5. Network posture (plan §10):
   - Bind `127.0.0.1` by default.
   - Binding `0.0.0.0` (env `AGENTKIT_BIND=0.0.0.0`) REQUIRES `AGENTKIT_AUTH_TOKEN`;
     requests must then present `Authorization: Bearer <token>`.
   - Run as non-root.

## Writer (frontend) contract

- Resolve `instructions` (inline string or file contents from the build context)
  into the single `instructions:` scalar — serve never fetches sources.
- Emit `tools[].command` and `tools[].env` verbatim from the agentkitfile.
- Never write secret values; only env var NAMES (enforced by config.Validate).
