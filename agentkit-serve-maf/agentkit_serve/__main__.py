"""``agentkit-serve`` CLI entrypoint.

Loads ``/agent/agent.yaml``, applies the network posture (plan §10), and runs
uvicorn. Network posture:

* Bind ``127.0.0.1`` by default (``AGENTKIT_BIND`` overrides).
* A NON-loopback bind (e.g. ``0.0.0.0``) REQUIRES ``AGENTKIT_AUTH_TOKEN`` — we
  refuse to start an unauthenticated agent on a routable interface.
* When a token is set, ``/v1/*`` requires ``Authorization: Bearer <token>``;
  ``/healthz`` stays open.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import NoReturn

import uvicorn

from .config import load_or_exit
from .server import create_app

# Hosts that mean "loopback only" — a bind to any of these needs no auth token.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "::ffff:127.0.0.1"})

DEFAULT_CONFIG_PATH = "/agent/agent.yaml"
DEFAULT_PORT = 8080


def _is_loopback(bind: str) -> bool:
    return bind.strip().lower() in _LOOPBACK_HOSTS


def _fail(message: str) -> NoReturn:
    print(f"agentkit-serve: {message}", file=sys.stderr)
    raise SystemExit(2)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="agentkit-serve",
        description="Serve an AgentKit agent (OpenAI Chat-Completions facade).",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"path to agent.yaml (default: {DEFAULT_CONFIG_PATH})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    spec = load_or_exit(args.config)

    # --- resolve bind/port ------------------------------------------------
    bind = os.environ.get("AGENTKIT_BIND", "127.0.0.1").strip()
    if not bind:
        bind = "127.0.0.1"
    port = spec.expose.port or DEFAULT_PORT

    # --- network posture gate (plan §10) ----------------------------------
    auth_token = os.environ.get("AGENTKIT_AUTH_TOKEN") or None
    if not _is_loopback(bind) and not auth_token:
        _fail(
            f"refusing to bind {bind!r} without authentication: set "
            f"AGENTKIT_AUTH_TOKEN to require `Authorization: Bearer <token>` on "
            f"/v1/*, or bind 127.0.0.1 (the default) for loopback-only access"
        )

    app = create_app(spec, auth_token=auth_token)

    # uvicorn's default access log can echo paths/headers; keep it but never log
    # bodies. The app itself never logs secrets.
    uvicorn.run(app, host=bind, port=port, log_level="info", access_log=True)


if __name__ == "__main__":  # pragma: no cover
    main()
