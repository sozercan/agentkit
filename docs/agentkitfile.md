# Agentkitfile reference

An Agentkitfile is the user-authored BuildKit frontend input for AgentKit. It is
a YAML file with `kind: Agent`, usually named `agentkitfile.yaml` and referenced
with Docker's `#syntax=` directive.

```yaml
#syntax=ghcr.io/sozercan/agentkit/agentkit:latest
apiVersion: v1alpha1
kind: Agent
metadata:
  name: url-summarizer
runtime: pydantic-ai
model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
  name: gpt-4o-mini
  apiKeyEnv: OPENAI_API_KEY
instructions: |
  Summarize any URL in three bullet points.
tools:
  - name: fetch
    command: ["uvx", "mcp-server-fetch"]
    env: ["FETCH_TIMEOUT"]
expose:
  openai: true
  port: 8080
```

## Load and validation behavior

The Go loader in `pkg/agentkit/config` first probes `apiVersion` and `kind`, then
strictly decodes the full schema. This means:

- `kind` is required and must be `Agent`.
- `apiVersion` is required and must be `v1alpha1`.
- Unknown or misspelled fields fail the build with line/column context.
- Validation reports all detected schema problems together where possible.

## Fields

### `metadata`

```yaml
metadata:
  name: url-summarizer
  labels:
    com.example.team: platform
```

- `name` is required and becomes both AgentKit metadata and the OCI image title.
- `labels` is optional. User-supplied labels are copied into the final image
  config alongside AgentKit labels.


### `debug`

```yaml
debug: true
```

`debug` is accepted by the schema for build-time diagnostics, but the current
frontend does not branch on it when producing the image.

### `runtime`

`runtime` selects the runtime adapter image used as the final image base.
Omitting it selects `pydantic-ai`.

| Value | Meaning |
|---|---|
| `pydantic-ai` | Default adapter backed by pydantic-ai. |
| `microsoft-agent-framework` | Adapter backed by Microsoft Agent Framework. |
| `maf` | Alias for `microsoft-agent-framework`. |
| `langgraph` | Adapter backed by LangChain/LangGraph. |

The build arg `--build-arg runtime=<name>` overrides the file value. The build
arg `--build-arg adapter=<image-ref>` overrides the selected runtime's default
adapter image ref, which is how the local dev loop tests unpublished adapters.

### `model`

```yaml
model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
  name: gpt-4o-mini
  apiKeyEnv: OPENAI_API_KEY
```

- `provider` must be `openai-compatible`.
- `baseURL` is the OpenAI-compatible `/v1` endpoint the adapter uses.
- `name` is the model name sent to that endpoint.
- `apiKeyEnv` is optional. When present, it is the name of an env var read at
  container startup. Do not put secret values in YAML.

If `apiKeyEnv` is omitted, the runtime supplies a non-secret placeholder key for
OpenAI-compatible endpoints that do not require authentication.

### `instructions`

Inline form:

```yaml
instructions: |
  You are concise.
```

File-backed form:

```yaml
instructions:
  file: ./prompt.md
```

Exactly one source is allowed. File paths are read from the BuildKit context
during the build; the final `/agent/agent.yaml` contains the resolved prompt
text, not the file reference.

### `tools`

```yaml
tools:
  - name: fetch
    command: ["uvx", "mcp-server-fetch"]
    env: ["FETCH_TIMEOUT"]
```

Each tool is a stdio MCP server:

- `name` is required and must be unique within the agent.
- `command` is required and must contain at least the executable; empty command
  parts are rejected.
- `env` is optional and lists env var names that may be passed to this tool
  subprocess.

Runtime adapters pass only declared, present env vars into tool subprocesses. A
tool env value that references `${OTHER_VAR}` must also list `OTHER_VAR` in the
same tool's allowlist, preventing accidental secret bleed from the parent process.

### `expose`

```yaml
expose:
  openai: true
  port: 8080
```

- `openai` must be `true`.
- `port` is optional in the authored file. When omitted, the effective Agent uses
  port `8080` and the final image exposes that port.

## Build inputs and targets

AgentKit supports the normal BuildKit frontend options used by `docker buildx`:

- `-f <file>` / `filename` selects the Agentkitfile. The default is
  `agentkitfile.yaml`.
- `--platform` can contain one or more target platforms. The frontend builds each
  platform in parallel and returns a multi-platform result when requested.
- `--target` routes to `<runtime>/image`. Empty target and a bare runtime target
  both mean the runtime's image output.
- cache import options are passed through to the BuildKit solve.

The only output kind registered by the router is an OCI image.

## What gets baked

The built image contains:

- the selected runtime adapter filesystem,
- `/agent/agent.yaml`, rendered from the effective Agent,
- an entrypoint of `/opt/agentkit/bin/agentkit-serve --config /agent/agent.yaml`,
- non-root user `1000:1000`,
- `AGENTKIT_BIND=127.0.0.1`, and
- AgentKit OCI labels for runtime, agent name, and ABI version.

Secret values are never written into the image.
