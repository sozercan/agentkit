# AgentKit agent instructions

## Git and GitHub

- Use Conventional Commit / semantic commit subjects, e.g. `feat(runtime): add Orka smoke`.
- Sign commits with `git commit -s`.
- Do not add `[codex]` to PR titles.

## Validation

Before pushing non-trivial code changes, run the narrowest relevant checks and prefer these full checks when practical:

```sh
uv run --directory runtimes/common --extra dev pytest -q
go test ./...
make lint
```

When editing GitHub Actions, also run:

```sh
go run github.com/rhysd/actionlint/cmd/actionlint@v1.7.12 .github/workflows/*.yml
```

## Orka harness protocol

Orka's `orka.harness.v1` wire contract is defined by Orka's `internal/harness/protocol.go` and client/conformance packages. AgentKit must conform to that contract; do not invent Orka-like alternate field names or response shapes.

For Orka adapter changes:

- Keep `GET /v1/health` and `GET /v1/capabilities` unauthenticated.
- Keep `/v1/turns`, `/v1/turns/{turnID}/events`, `/cancel`, and `/output` bearer-authenticated.
- Preserve native Orka JSON shapes: `HealthResponse`, flat `CapabilitiesResponse`, `StartTurnRequest`, `StartTurnResponse`, `CancelTurnRequest`, `CancelTurnResponse`, and `HarnessEventFrame`.
- Keep observed mode as the default/only advertised Orka tool mode unless brokered mode is explicitly implemented.
- Do not promote request-controlled `contextRefs` into system prompts or model instructions.
- Treat `input.env` as sensitive per-turn material: only allow ABI-declared names and reject AgentKit control variables such as `AGENTKIT_*`.

When practical, prove Orka changes with a live local Orka conformance smoke against `agentkit_serve_common.orka.create_orka_app`, in addition to the common Python tests.
