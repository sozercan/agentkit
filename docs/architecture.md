# Architecture

AgentKit has two halves:

1. a Go BuildKit gateway frontend that turns an Agentkitfile into an OCI image,
2. Python runtime adapter images that read the baked agent contract and serve the
   agent.

The stable seam between the halves is `/agent/agent.yaml`, documented in
[`agent-abi.md`](agent-abi.md).

## Build-time flow

```text
Docker/BuildKit
  -> cmd/frontend
  -> pkg/build.Build
  -> pkg/agentkit/config.NewFromBytes + Validate
  -> pkg/build.resolveInstructions
  -> pkg/agentkit/effective.FromConfig
  -> pkg/agentkit/abi.Render
  -> pkg/agentkit2llb/agent.Agentkit2LLB
  -> final OCI image
```

### `cmd/frontend`

`cmd/frontend/main.go` is the BuildKit gateway entrypoint. It configures logging
and delegates every build to `build.Build` via `grpcclient.RunFromEnvironment`.

### `pkg/agentkit/config`

This package owns the authored Agentkitfile schema and validation rules:

- `load.go` probes `{apiVersion, kind}` before strict decoding.
- `specs.go` declares `AgentConfig`, `Metadata`, `Model`, and `Expose`.
- `source.go` models instruction sources as a tagged union: inline text or a
  file path in the build context.
- `tool.go` models tools as a source union. The accepted tool source is a stdio
  `command`; an `image` field is present in the Go shape but validation rejects
  it.
- `validate.go` enforces supported runtime names, OpenAI-compatible models,
  one instruction source, unique tool names, secret-name hygiene, and
  `expose.openai: true`.

The loader is intentionally strict: kind-less files, unsupported kinds, unknown
fields, invalid sources, and likely secret literals all fail before LLB work
starts.

### `pkg/agentkit/runtimes`

`catalog.go` is the single source of truth for shipped runtime identities:

- `pydantic-ai` (default),
- `microsoft-agent-framework` with alias `maf`,
- `langgraph`.

It also carries each runtime's default adapter image ref. The config validator
and the build router both consult this package, avoiding duplicate runtime
registries.

The YAML files in `runtimes/catalog/` mirror this Go catalog and are covered by
`catalog_file_test.go`.

### `pkg/build`

`build.go` is the frontend orchestrator:

- loads the Agentkitfile from local, Git, or HTTP BuildKit context,
- applies build-arg overrides such as `runtime`,
- validates the config,
- resolves the route for `target` + runtime,
- resolves file-backed instructions from the build context,
- canonicalizes the config into an effective Agent,
- solves one LLB graph per target platform, and
- wires image config metadata into the BuildKit result.

`router.go` derives `<runtime>/image` routes from the runtime catalog. Empty
`target`, a bare runtime target, and exact `<runtime>/image` all route to the
same image handler. Runtime aliases are canonicalized before lookup.

`instructions.go` is the seam between authored instruction sources and the
BuildKit context.

### `pkg/agentkit/effective`

`effective.FromConfig` turns a validated authored config plus resolved
instructions into a build-ready Agent. It applies defaults and canonicalization:

- empty runtime becomes `pydantic-ai`,
- runtime aliases become canonical names,
- empty port becomes `8080`,
- mutable maps and slices are copied before downstream rendering.

Downstream packages consume this effective shape so they do not reinterpret raw
authoring defaults.

### `pkg/agentkit/abi`

`abi.Render` renders the effective Agent into the exact YAML shape consumed by
the Python reader. It writes `abiVersion`, metadata, model, resolved
instructions, tool commands/env allowlists, and expose information. `abi.Path` is
`/agent/agent.yaml`.

### `pkg/agentkit2llb/agent`

`Agentkit2LLB` starts from the selected adapter image and overlays one file:
`/agent/agent.yaml`. `NewImageConfig` then sets the final image metadata:

- user `1000:1000`,
- workdir `/`,
- entrypoint `/opt/agentkit/bin/agentkit-serve`,
- command `--config /agent/agent.yaml`,
- `PATH`, `AGENTKIT_BIND=127.0.0.1`, and `PYTHONUNBUFFERED=1`,
- exposed serve port,
- AgentKit and OCI labels.

No tool root filesystem or secret value is merged into the image.

## Runtime flow

```text
agentkit-serve --config /agent/agent.yaml
  -> runtimes/common.config.load
  -> runtimes/common.cli.run
  -> runtimes/common.server.create_app
  -> adapter agent_factory.build_runtime
  -> RuntimeSession.run(RunRequest)
  -> OpenAI-compatible response
```

### `runtimes/common`

`agentkit_serve_common` is framework-neutral:

- `config.py` strictly loads the baked ABI and checks the ABI version.
- `cli.py` loads config, chooses bind/port, and refuses non-loopback binds unless
  `AGENTKIT_AUTH_TOKEN` is set.
- `server.py` serves `/healthz`, `/v1/models`, and `/v1/chat/completions`.
- `conversation.py` converts OpenAI messages into a `RunRequest` whose final
  prompt is the last user message and whose history contains prior system, user,
  and assistant turns.
- `runtime.py` defines `RunRequest` consumers: `RuntimeFactory`,
  `RuntimeSession`, `RunResult`, and `AgentRunError`.
- `adapter_support.py` provides shared helpers for API-key resolution, tool env
  allowlists, MCP timeout parsing, and framework exception normalization.
- `conformance.py` defines adapter-neutral HTTP behavior tests reused by each
  adapter package.

The shared server imports no agent framework.

### Runtime adapters

Each adapter package contributes a thin CLI binding plus `agent_factory.py`:

- `runtimes/pydantic-ai` builds a pydantic-ai `Agent`, supports both pydantic-ai
  MCP APIs used across 1.x and 2.x, prefixes tool names, and maps pydantic-ai
  run results into `RunResult`.
- `runtimes/microsoft-agent-framework` builds a Microsoft Agent Framework chat
  client and agent, enforces guardrails that keep Azure/Foundry/Copilot packages
  out of the generic adapter, and maps MAF results into `RunResult`.
- `runtimes/langgraph` builds a LangGraph ReAct-style graph with LangChain
  OpenAI chat models and persistent MCP sessions, aggregates token usage across
  model calls, and maps the final AI message into `RunResult`.

Adapters remain separate images with separate framework dependencies. The shared
core is installed into each adapter image first, then the adapter package adds its
framework-specific dependencies.

## Security and network posture

- Agent images run as non-root (`1000:1000`).
- The default bind is loopback (`127.0.0.1`).
- Setting `AGENTKIT_BIND` to a non-loopback host requires
  `AGENTKIT_AUTH_TOKEN`; `/v1/*` then requires `Authorization: Bearer <token>`.
- `/healthz` remains unauthenticated for liveness checks.
- `model.apiKeyEnv` and tool `env` values are names, not secret values.
- Tool subprocesses receive only their declared env names and never inherit the
  full container environment.
