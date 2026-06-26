#!/usr/bin/env python3
"""Render an azd/Foundry ContainerAgent YAML from a provider-neutral profile.

The profile stays outside AgentKit core. It maps generic image/protocol/env
requirements onto Foundry's hosted-agent deployment shape, including the
snake_case ``environment_variables`` list form used by Foundry samples and azd.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import yaml

_ENV_NAME_RE = re.compile(r"^[A-Z0-9_]+$")
_DEFAULT_RESOURCES = {"cpu": "0.25", "memory": "0.5Gi"}


def _load(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"profile {path} must be a YAML mapping")
    return data


def _protocols(values: Any) -> list[dict[str, str]]:
    if values is None:
        values = ["invocations", "responses"]
    out: list[dict[str, str]] = []
    for item in values:
        if isinstance(item, str):
            out.append({"protocol": item, "version": "1.0.0"})
        elif isinstance(item, dict) and "protocol" in item:
            out.append({
                "protocol": str(item["protocol"]),
                "version": str(item.get("version", "1.0.0")),
            })
        else:
            raise SystemExit(f"invalid protocol entry: {item!r}")
    return out


def _env_entries(profile: dict[str, Any]) -> list[dict[str, str]]:
    raw = profile.get("env") or profile.get("environment") or {}
    if isinstance(raw, list):
        iterable = []
        for item in raw:
            if not isinstance(item, dict) or "name" not in item:
                raise SystemExit(f"env list entries must be mappings with name/value: {item!r}")
            iterable.append((item["name"], item.get("value", f"${{{item['name']}}}")))
    elif isinstance(raw, dict):
        iterable = list(raw.items())
    else:
        raise SystemExit("env must be a mapping or list")

    entries: list[dict[str, str]] = []
    for name, value in iterable:
        name = str(name)
        if not _ENV_NAME_RE.fullmatch(name):
            raise SystemExit(f"env name {name!r} must match [A-Z0-9_]+")
        if value is None:
            value = f"${{{name}}}"
        entries.append({"name": name, "value": str(value)})
    return entries


def render(profile: dict[str, Any]) -> dict[str, Any]:
    if profile.get("target") not in (None, "foundry-hosted-agent"):
        raise SystemExit("target must be foundry-hosted-agent")
    name = profile.get("name") or profile.get("agentName") or profile.get("metadata", {}).get("name")
    image = profile.get("image")
    if not name:
        raise SystemExit("profile must set name")
    if not image:
        raise SystemExit("profile must set image")

    out: dict[str, Any] = {
        "kind": "hosted",
        "name": str(name),
        "image": str(image),
        "protocols": _protocols(profile.get("protocols")),
        "resources": profile.get("resources") or _DEFAULT_RESOURCES,
    }
    env = _env_entries(profile)
    if env:
        # Foundry/azd sample-compatible shape. Do not use camelCase here: it can
        # validate locally yet fail to inject variables into hosted containers.
        out["environment_variables"] = env
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args(argv)
    rendered = render(_load(args.profile))
    text = yaml.safe_dump(rendered, sort_keys=False)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
