# The `agent.yaml` ABI

`/agent/agent.yaml` is the build-time/runtime contract between AgentKit's Go
frontend and Python runtime adapters.

- **Writer:** `pkg/agentkit/abi` renders the file from an effective Agent, and
  `pkg/agentkit2llb/agent` bakes it into the final image.
- **Reader:** `runtimes/common/agentkit_serve_common/config.py` loads and
  validates the file before any adapter starts serving.

The writer and reader are intentionally strict. Any shape change must update both
sides and the ABI tests.

## Location

`/agent/agent.yaml` (`pkg/agentkit/abi.Path`).

## Version

`abiVersion: v0` is the ABI schema value currently emitted by the Go writer and
accepted by the Python reader. This is not the same field as the user-authored
Agentkitfile `apiVersion`.

## Schema

```yaml
abiVersion: v0

metadata:
  name: url-summarizer

model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
  name: gpt-4o-mini
  apiKeyEnv: OPENAI_API_KEY

instructions: |
  Summarize any URL the user gives you in three bullet points.

tools:
  - name: fetch
    command: ["uvx", "mcp-server-fetch"]
    env: ["FETCH_TIMEOUT"]

expose:
  openai: true
  port: 8080
```

## Field semantics

- `metadata.name` comes from `metadata.name` in the Agentkitfile.
- `model.provider` is `openai-compatible`.
- `model.baseURL` and `model.name` are passed to the selected adapter's
  OpenAI-compatible chat model client.
- `model.apiKeyEnv` is optional and is an env var name. The runtime reads the
  value from the process environment when it constructs the model client.
- `instructions` is fully resolved text. Runtime adapters never fetch prompt
  files or reinterpret instruction sources.
- `tools` is a list of stdio MCP servers. `command` is argv; `env` is the
  allowlist of env var names that may be copied into that subprocess.
- `expose.openai` must be `true`.
- `expose.port` is the port the runtime serves.

The Python reader forbids unknown keys, validates env var names, rejects empty
commands, and rejects unsupported ABI versions.

## Writer contract

The Go frontend must:

1. validate the authored Agentkitfile before rendering,
2. resolve instruction sources into a scalar string,
3. canonicalize runtime aliases and default the port before ABI rendering,
4. emit exactly the keys the reader expects,
5. preserve tool command argv and env allowlist names, and
6. never write secret values.

## Reader contract

The shared Python runtime core must:

1. load and validate `/agent/agent.yaml`, exiting non-zero on missing, invalid,
   or unsupported files;
2. construct the selected adapter runtime from the validated `AgentSpec`;
3. resolve `model.apiKeyEnv` from the process environment or use the no-auth
   placeholder when no key env is declared;
4. start stdio MCP tool sessions for the application lifespan, passing only
   declared tool env vars; and
5. serve the OpenAI-compatible façade consistently across adapters.

## Served HTTP contract

The runtime serves:

- `GET /healthz` — liveness.
- `GET /v1/models` — one-model listing containing `model.name`.
- `POST /v1/chat/completions` — non-streaming run that returns one
  `chat.completion` object.

`POST /v1/chat/completions` rejects:

- `stream: true`,
- request-supplied `tools`,
- request-supplied `tool_choice` values other than missing, empty, `none`, or
  `auto`,
- an empty `messages` array, and
- requests whose final message is not a `user` message.

The response collapses any intermediate framework/tool loop into one assistant
message with `finish_reason: "stop"`.

## Network and process contract

Generated images run `agentkit-serve --config /agent/agent.yaml` as user
`1000:1000`. They bind `127.0.0.1` by default. If `AGENTKIT_BIND` is set to a
non-loopback host such as `0.0.0.0`, startup requires `AGENTKIT_AUTH_TOKEN`; then
`/v1/*` requests must include `Authorization: Bearer <token>`. `/healthz` remains
open.
