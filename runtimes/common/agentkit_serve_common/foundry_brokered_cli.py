"""Brokered-only Foundry hosted Responses entrypoint.

This entrypoint is intentionally small: it loads a baked ``agent.yaml`` with
static ``brokeredTools`` and serves the shared Foundry `/responses` brokered
adapter without constructing a framework runtime. It is useful for deterministic
Foundry/Orka brokered smokes and for the lower-level model-loop fallback where
AgentKit-owned direct tools must stay disabled.
"""

from __future__ import annotations

import argparse
import json
import os
from types import TracebackType
from typing import Sequence

import uvicorn

from .config import AgentSpec, load_or_exit
from .foundry import create_foundry_app
from .runtime import AgentRunError, RunResult
from .conversation import RunRequest

DEFAULT_CONFIG_PATH = "/agent/agent.yaml"
DEFAULT_PORT = 8088
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "::ffff:127.0.0.1"}


def _is_loopback(host: str) -> bool:
    return host.strip().lower() in _LOOPBACK_HOSTS


class _NoDirectRuntime:
    async def __aenter__(self):
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        return None

    async def run(self, request: RunRequest) -> RunResult:  # noqa: ARG002 - direct run is intentionally disabled.
        raise AgentRunError(
            "direct runtime execution is disabled in Foundry brokered-only mode",
            status=400,
            code="DirectRuntimeDisabled",
        )


class _NoDirectFactory:
    def build_runtime(self, spec: AgentSpec):  # noqa: ARG002 - spec-independent guard runtime.
        return _NoDirectRuntime()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="agentkit-foundry-brokered",
        description="Serve a brokered-only AgentKit Foundry /responses endpoint from agent.yaml.",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help=f"path to agent.yaml (default: {DEFAULT_CONFIG_PATH})")
    parser.add_argument("--host", default=os.environ.get("AGENTKIT_BIND", "0.0.0.0"), help="host interface to bind")
    parser.add_argument("--port", type=int, default=int(os.environ.get("AGENTKIT_PORT", os.environ.get("PORT", str(DEFAULT_PORT)))), help="port to bind")
    parser.add_argument("--dry-run", action="store_true", help="load config and print selected serving metadata without binding")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    spec = load_or_exit(args.config)
    if not spec.brokered_tools:
        raise SystemExit("agentkit-foundry-brokered: agent.yaml must declare at least one brokeredTools entry")
    auth_token = os.environ.get("AGENTKIT_AUTH_TOKEN") or None
    if not args.dry_run and not _is_loopback(args.host) and not auth_token:
        raise SystemExit(
            f"agentkit-foundry-brokered: refusing to bind {args.host!r} without AGENTKIT_AUTH_TOKEN; "
            "set a bearer token or bind 127.0.0.1 for local-only use"
        )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "host": args.host,
                    "port": args.port,
                    "agent": spec.metadata.name,
                    "brokeredTools": [tool.name for tool in spec.brokered_tools],
                    "auth": "configured" if auth_token else "none",
                    "continuationProof": "configured" if os.environ.get("AGENTKIT_FOUNDRY_BROKERED_CONTINUATION_PROOF") else "missing",
                },
                sort_keys=True,
            )
        )
        return 0
    app = create_foundry_app(spec, _NoDirectFactory(), auth_token=auth_token)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", access_log=True)
    return 0


__all__ = ["main"]


if __name__ == "__main__":  # pragma: no cover - exercised by console script/dry-run tests.
    raise SystemExit(main())
