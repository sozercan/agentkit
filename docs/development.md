# Development guide

This repository contains a Go BuildKit frontend, Python runtime adapter packages,
Docker adapter images, and integration fixtures. The Makefile is the source of
truth for the local Docker loop.

## Local prerequisites

- Go matching `go.mod`.
- Python 3.12 for runtime package work.
- Docker Buildx.
- A daemon-backed Buildx builder for `--load` workflows. The Makefile defaults
  `BUILDER=desktop-linux` for test-agent builds because docker-container builders
  cannot see local images unless they are pushed to a registry.

## Core loop

```sh
make build-agentkit      # frontend gateway image -> agentkit:test
make build-serve         # pydantic-ai adapter    -> agentkit-serve:test
make build-test-agent    # fixture agent image    -> hello-agent:test
make run-test-agent      # run the image; needs OPENAI_API_KEY
```

`build-test-agent` connects the local images with build args:

- `BUILDKIT_SYNTAX=agentkit:test` makes the fixture use the local frontend.
- `adapter=agentkit-serve:test` makes the Go converter use the local adapter
  image as the LLB base.

## Runtime-specific loops

Microsoft Agent Framework:

```sh
make build-serve-maf
make build-test-agent RUNTIME=maf
```

LangGraph:

```sh
make build-serve-langgraph
make build-test-agent RUNTIME=langgraph
```

`RUNTIME` selects the adapter image, fixture file, and output image tag:

| `RUNTIME` | Adapter image | Fixture | Output image |
|---|---|---|---|
| `pydantic-ai` | `agentkit-serve:test` | `test/agentkitfile-hello.yaml` | `hello-agent:test` |
| `maf` / `microsoft-agent-framework` | `agentkit-serve-maf:test` | `test/agentkitfile-maf-hello.yaml` | `maf-agent:test` |
| `langgraph` | `agentkit-serve-langgraph:test` | `test/agentkitfile-langgraph-hello.yaml` | `langgraph-agent:test` |

## Go checks

```sh
golangci-lint run ./... --timeout 5m
golangci-lint fmt --diff
go vet ./...
go test ./... -race
go build -o /tmp/agentkit-frontend ./cmd/frontend
```

Go tests cover:

- strict Agentkitfile loading and validation,
- instruction source resolution,
- runtime aliasing and route lookup,
- effective Agent defaults and copy semantics,
- ABI rendering and golden round trips,
- OCI image config generation, and
- runtime catalog file parity.

## Python checks

For one adapter package:

```sh
cd runtimes/langgraph
python3.12 -m venv .venv
. .venv/bin/activate
pip install -e ../common -e '.[dev]' build
python -m compileall agentkit_serve ../common/agentkit_serve_common
python -m pytest -q
python -m build --wheel
```

For `runtimes/common`, omit `-e ../common` and compile/test
`agentkit_serve_common` directly.

Python tests cover:

- ABI reader validation,
- OpenAI façade conformance shared by every adapter,
- conversation normalization,
- runtime lifecycle startup/shutdown,
- tool env allowlist behavior,
- MCP timeout parsing,
- framework import guardrails, and
- adapter-specific result/usage mapping.

## Docker and smoke checks

The CI Docker job builds:

1. the frontend image,
2. all three adapter images,
3. a fixture agent image for each runtime, and
4. each generated agent enough to pass `/healthz`.

The smoke containers bind `0.0.0.0` inside the container and set
`AGENTKIT_AUTH_TOKEN`, proving the startup auth gate is satisfied while `/healthz`
remains probeable.

## Live Copilot/Vekil E2E

The optional live job runs `scripts/live-copilot-agent-e2e.sh` when
`COPILOT_GITHUB_TOKEN` is available and the run is allowed to access repository
secrets. It uses the pinned `ghcr.io/sozercan/vekil` image to provide an
OpenAI-compatible endpoint and validates a real built AgentKit container through
`/v1/chat/completions`.

Fork PRs, Dependabot runs without secrets, and repos without the token skip this
job while normal offline checks still run.

## Foundry Hosted Agents fixture

`AGENTKIT_PROTOCOL=foundry` now exposes Foundry Hosted Agents' `/readiness`,
`/invocations`, and minimal `/responses` surfaces directly from every adapter
image. `test/foundry-hosted-agent/` remains as a compatibility smoke fixture for
older wrapping flows; AgentKit still owns `/agent/agent.yaml` and the runtime
lifecycle in both modes.
