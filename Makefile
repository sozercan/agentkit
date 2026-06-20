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

# Build the runtime adapter (agentkit-serve) image from agentkit-serve/Dockerfile.
# This is the image the converter uses as the LLB base.
.PHONY: build-serve
build-serve:
	docker buildx build agentkit-serve -t agentkit-serve:$(TAG) --load

# Build the canonical test agent from test/agentkitfile-hello.yaml against the
# LOCAL frontend (BUILDKIT_SYNTAX) and the LOCAL adapter (--build-arg adapter).
# --provenance=false keeps the output a plain single-platform image for --load.
.PHONY: build-test-agent
build-test-agent:
	docker buildx build --builder $(BUILDER) . -f test/agentkitfile-hello.yaml \
		--build-arg BUILDKIT_SYNTAX=agentkit:$(TAG) \
		--build-arg adapter=agentkit-serve:$(TAG) \
		--platform $(PLATFORM) \
		-t hello-agent:$(TAG) --load --provenance=false

# Run the built test agent. Expects OPENAI_API_KEY in the environment; the agent
# binds 127.0.0.1 inside the container and serves the OpenAI /v1 façade on :8080.
.PHONY: run-test-agent
run-test-agent:
	docker run --rm --platform $(PLATFORM) -p 127.0.0.1:8080:8080 -e OPENAI_API_KEY=$$OPENAI_API_KEY hello-agent:$(TAG)
