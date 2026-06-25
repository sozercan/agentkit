# AgentKit Makefile.
#
# ─── Local dev loop (plan §15.6) ────────────────────────────────────────────
# AgentKit is a BuildKit gateway frontend, so the inner loop is three builds:
#
#   1. make build-agentkit     # build the frontend image      -> agentkit:$(TAG)
#   2. make build-serve        # build the runtime adapter      -> agentkit-serve:$(TAG)
#   3. make build-test-agent   # build a test agentkitfile using BOTH locals
#                              #   -> hello-agent:$(TAG)
#   then: make run-test-agent  # run it and curl the OpenAI /v1 façade
#
# build-test-agent wires the locals together via two build-args the Go frontend
# reads: BUILDKIT_SYNTAX pins the gateway to the local frontend image, and
# `adapter` overrides the runtime adapter ref (router.go AdapterRef) so the
# converter uses your freshly-built agentkit-serve:$(TAG) as the LLB base instead
# of the published ghcr.io/sozercan/agentkit/serve-pydantic-ai:latest default.
# ────────────────────────────────────────────────────────────────────────────

REGISTRY ?= ghcr.io/sozercan
TAG ?= test

# The agent build must run on a builder that can SEE the locally-built frontend
# and adapter images. A `docker-container`/remote buildx driver pulls from a
# registry and cannot resolve `agentkit:$(TAG)` from the local daemon store, so
# build-test-agent defaults to the daemon-backed `desktop-linux` builder. Override
# with `make build-test-agent BUILDER=<name>` (use a `docker` driver builder).
BUILDER ?= desktop-linux

# The runtime adapter image (agentkit-serve) is linux/amd64 (its uv base is
# amd64-only), so the test agent is built for the same platform.
PLATFORM ?= linux/amd64

# RUNTIME selects which runtime adapter the test-agent targets: `pydantic-ai`
# (default), Microsoft Agent Framework (`maf` alias or canonical name), or
# LangGraph (`langgraph`). build-test-agent derives the adapter image, fixture,
# and output tag from it, so you can build the SAME logical agent under any
# supported runtime (the §10.4 equivalence proof):
#   make build-serve build-test-agent                        # pydantic-ai → hello-agent
#   make build-serve-maf build-test-agent RUNTIME=maf        # MAF         → maf-agent
#   make build-serve-langgraph build-test-agent RUNTIME=langgraph # LangGraph → langgraph-agent
RUNTIME ?= pydantic-ai
# Per-runtime adapter image, fixture, and output tag (overridable). Branches
# match all accepted spellings so a canonical name does not silently fall through
# to the pydantic-ai default.
ifneq ($(filter langgraph,$(RUNTIME)),)
SERVE_IMAGE ?= agentkit-serve-langgraph:$(TAG)
FIXTURE     ?= test/agentkitfile-langgraph-hello.yaml
AGENT_IMAGE ?= langgraph-agent:$(TAG)
else ifneq ($(filter maf microsoft-agent-framework,$(RUNTIME)),)
SERVE_IMAGE ?= agentkit-serve-maf:$(TAG)
FIXTURE     ?= test/agentkitfile-maf-hello.yaml
AGENT_IMAGE ?= maf-agent:$(TAG)
else
SERVE_IMAGE ?= agentkit-serve:$(TAG)
FIXTURE     ?= test/agentkitfile-hello.yaml
AGENT_IMAGE ?= hello-agent:$(TAG)
endif

# LDFLAGS is passed into the frontend image build. Kept empty by default; the
# Dockerfile always appends `-w -s -extldflags '-static'`. Override to inject
# version stamping, e.g. `make build-agentkit LDFLAGS=-X main.version=$(TAG)`.
LDFLAGS ?=

.PHONY: lint
lint:
	golangci-lint run ./... --timeout 5m

.PHONY: test
test:
	go test ./... -race

# Build the frontend (gateway) image. Tagged with the short local name so it can
# be referenced as `#syntax=agentkit:$(TAG)` / BUILDKIT_SYNTAX below.
.PHONY: build-agentkit
build-agentkit:
	docker buildx build . -t agentkit:$(TAG) \
		--build-arg LDFLAGS="$(LDFLAGS)" \
		--load

# Build the runtime adapter (agentkit-serve) image from runtimes/pydantic-ai/Dockerfile.
# This is the image the converter uses as the LLB base. The build context is the
# REPO ROOT (not the adapter subdir) so the Dockerfile can COPY the shared
# `runtimes/common/` package alongside the adapter (the root .dockerignore
# keeps the context small).
.PHONY: build-serve
build-serve:
	docker buildx build . -f runtimes/pydantic-ai/Dockerfile -t agentkit-serve:$(TAG) --load

# Build the Microsoft Agent Framework runtime adapter (agentkit-serve-maf) image.
# This is the LLB base used when an agentkitfile selects
# `runtime: microsoft-agent-framework` (alias `maf`).
.PHONY: build-serve-maf
build-serve-maf:
	docker buildx build . -f runtimes/microsoft-agent-framework/Dockerfile -t agentkit-serve-maf:$(TAG) --load

# Build the LangGraph runtime adapter (agentkit-serve-langgraph) image.
# This is the LLB base used when an agentkitfile selects `runtime: langgraph`.
.PHONY: build-serve-langgraph
build-serve-langgraph:
	docker buildx build . -f runtimes/langgraph/Dockerfile -t agentkit-serve-langgraph:$(TAG) --load

# Build a test agent against the LOCAL frontend (BUILDKIT_SYNTAX) and the LOCAL
# adapter (--build-arg adapter). The runtime, fixture, adapter image, and output
# tag all derive from RUNTIME (default pydantic-ai; `RUNTIME=maf` for MAF;
# `RUNTIME=langgraph` for LangGraph).
# --provenance=false keeps the output a plain single-platform image for --load.
.PHONY: build-test-agent
build-test-agent:
	docker buildx build --builder $(BUILDER) . -f $(FIXTURE) \
		--build-arg BUILDKIT_SYNTAX=agentkit:$(TAG) \
		--build-arg adapter=$(SERVE_IMAGE) \
		--platform $(PLATFORM) \
		-t $(AGENT_IMAGE) --load --provenance=false

# Run the built test agent. Expects OPENAI_API_KEY in the environment; the agent
# binds 127.0.0.1 inside the container and serves the OpenAI /v1 façade on :8080.
.PHONY: run-test-agent
run-test-agent:
	docker run --rm --platform $(PLATFORM) -p 127.0.0.1:8080:8080 -e OPENAI_API_KEY=$$OPENAI_API_KEY $(AGENT_IMAGE)
