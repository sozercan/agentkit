# AgentKit

Docker build your agent into a standard OCI image that serves an OpenAI `/v1`
Chat-Completions façade (plus MCP tools). AIKit is Docker-for-models; AgentKit is
Docker-for-agents.

AgentKit is a [BuildKit](https://github.com/moby/buildkit) gateway frontend. You
write an `agentkitfile.yaml` (`kind: Agent`) describing a model, instructions,
and tools; `docker build` turns it into a normal, runnable container image. No
new runtime, no orchestration — the output is an OCI image you ship anywhere.

## The v0 agent (four keys)

```yaml
#syntax=ghcr.io/sozercan/agentkit/agentkit:latest
apiVersion: v1alpha1
kind: Agent
metadata:
  name: url-summarizer
model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
  name: gpt-4o-mini
  apiKeyEnv: OPENAI_API_KEY        # NAME of an env var, never the secret value
instructions: |
  Summarize any URL the user gives you in three bullet points.
expose:
  openai: true
```

Build and run it:

```sh
docker buildx build . -f agentkitfile.yaml -t url-summarizer:latest --load
docker run --rm -p 127.0.0.1:8080:8080 -e OPENAI_API_KEY=$OPENAI_API_KEY url-summarizer:latest
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"https://example.com"}]}'
```

Add stdio MCP tools with a `tools:` list (see `test/agentkitfile-tools.yaml`).

> **First-boot note.** v0 tools are stdio MCP servers launched at runtime via
> `uvx`/`npx`. The *first* launch downloads and installs the server package before
> it speaks MCP, so the initial container start can take 30–90s. The init timeout
> defaults to 120s; tune it with `-e AGENTKIT_MCP_TIMEOUT=<seconds>`. (Building the
> tool package into the image for instant, offline starts is a v1 item —
> `build.resolvers`.)

## Runtimes

The agentkitfile is **target-neutral**: it describes the agent (model,
instructions, tools), and *which agent library executes it* is a swappable
implementation detail behind a frozen ABI (`docs/agent-abi.md`). Select one with
the optional `runtime:` key:

| `runtime:` value | Adapter | Framework |
|---|---|---|
| *(omitted)* / `pydantic-ai` | `agentkit-serve` | [pydantic-ai](https://ai.pydantic.dev) (default) |
| `microsoft-agent-framework` (alias `maf`) | `agentkit-serve-maf` | [Microsoft Agent Framework](https://github.com/microsoft/agent-framework) |

```yaml
runtime: microsoft-agent-framework   # or: maf
```

Both runtimes consume the **same** baked `/agent/agent.yaml` and serve the
**same** non-streaming OpenAI `/v1` façade with the same guards — so the same
agentkitfile produces a behavior-compatible image under either. Only the in-image
runtime adapter differs. The `AGENTKIT_MCP_TIMEOUT` knob applies to both.

## Local dev loop (3 steps)

AgentKit is itself a frontend image, so iterating means rebuilding the frontend,
the runtime adapter, and a test agent that uses both:

```sh
make build-agentkit      # 1. frontend (gateway) image -> agentkit:test
make build-serve         # 2. runtime adapter image    -> agentkit-serve:test
make build-test-agent    # 3. test/agentkitfile-hello.yaml using both locals
make run-test-agent      # run it (needs OPENAI_API_KEY)
```

`build-test-agent` pins the gateway with `--build-arg BUILDKIT_SYNTAX=agentkit:test`
and overrides the adapter base with `--build-arg adapter=agentkit-serve:test`.

To iterate on the **Microsoft Agent Framework** runtime instead, build its adapter
and target it with `RUNTIME=maf` (which also selects the MAF fixture and output
tag):

```sh
make build-serve-maf                      # agentkit-serve-maf:test
make build-test-agent RUNTIME=maf         # test/agentkitfile-maf-hello.yaml -> maf-agent:test
```

## CI

GitHub Actions runs the full closeout loop on pushes and pull requests:

- Go lint, formatting, vet, race tests, and frontend build.
- Python compile, pytest, and wheel checks for `runtimes/common/`,
  `runtimes/pydantic-ai/`, and `runtimes/microsoft-agent-framework/`.
- Docker builds for the frontend and both runtime adapters, followed by offline
  `/healthz` smoke tests for generated pydantic-ai and MAF agent images.
- Optional live Vekil-backed Copilot E2E, using the official pinned `ghcr.io/sozercan/vekil` image and a repository secret named
  `COPILOT_GITHUB_TOKEN`. If that secret is unavailable (for example on forks or
  unconfigured repos), or Vekil reports that the token lacks Copilot access/
  permissions, the live job is skipped while the offline checks still run.

The Docker-heavy jobs enable Docker's containerd image store because AgentKit's
BuildKit gateway frontend uses merge/diff operations that require it on
GitHub-hosted runners.

## Foundry Hosted Agents smoke test

AgentKit images natively expose the OpenAI `/v1` façade on port `8080`. Microsoft
Foundry Hosted Agents use a different container contract: a hosted protocol such
as `/invocations`, a `/readiness` probe, and port `8088`. The fixture under
`test/foundry-hosted-agent/` wraps a normal AgentKit image with that Foundry
`invocations` surface without changing the baked `/agent/agent.yaml` ABI.

The wrapper enters the same AgentKit agent async lifecycle as the native server,
so stdio MCP tool subprocesses are started and kept warm before requests are
handled. The fixture also includes a deterministic in-container
OpenAI-compatible mock model, which makes local and hosted response-body
comparison possible without external model credentials.

See `test/foundry-hosted-agent/README.md` for the exact build, local validation,
and `azd` deployment flow.

## Architecture

The runtime adapter (`agentkit-serve`, Python) is used as the LLB **base** image.
The frontend resolves your `instructions` and `tools` into the frozen
`/agent/agent.yaml` ABI (see `docs/agent-abi.md`) and merges that single layer on
top. At runtime the adapter reads it and serves the façade. Which adapter is used
as the base is chosen by `runtime:` (see [Runtimes](#runtimes)); the baked
`/agent/agent.yaml` is identical across runtimes.

Each adapter is a thin shell over a shared core:

- `runtimes/common/` — the **framework-neutral** package: the `agent.yaml`
  ABI loader, the OpenAI `/v1` façade, the CLI/network posture, and the neutral run
  contract (`RunResult`, `AgentRunError`, `RuntimeSession`, the
  `RuntimeFactory` protocol). Imports no agent framework.
- `runtimes/pydantic-ai/`, `runtimes/microsoft-agent-framework/` — each ships
  only an `agent_factory.py` (the one file that imports its framework) plus a
  thin `__main__.py`, implementing `RuntimeFactory` / `RuntimeSession`. They stay
  **separate images** with disjoint framework deps; that physical separation is
  what guarantees the lock-in boundary.

Adding a single-agent runtime is therefore one `agent_factory.py` + one Go
`runtimes.RuntimeSpec` entry; it inherits the shared `/v1` façade and the conformance
test suite for free.

## v0 scope / not yet

- **v0**: two runtimes (pydantic-ai default + microsoft-agent-framework),
  `provider: openai-compatible` only, stdio `command` MCP tools, the OpenAI `/v1`
  façade, single OCI image output.
- **Not yet**: image-based MCP tools, evals, lock file / SBOM / signing,
  agentpack, `extends`/patches, knowledge/RAG, memory/state, model fallback,
  streaming, and embedded/BYO serving targets.

Secrets are never baked: `apiKeyEnv` and tool `env:` carry env var **names**;
values are injected at `docker run` time.
