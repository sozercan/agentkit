"""Helpers for safe Orka-brokered tool schemas.

The hosted Foundry `/responses` endpoint cannot receive request-level tool
schemas, so brokered mode uses static schema-only declarations baked into
`agent.yaml`. This module keeps that schema surface small and deterministic:
only name, description, brokered class, JSON parameters schema, and an optional
schema digest may cross into hosted AgentKit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from .config import AgentSpec, BrokeredToolSpec, brokered_tool_schema_digest
from .runtime import BrokeredToolDefinition


def brokered_tool_definitions(spec: AgentSpec) -> list[BrokeredToolDefinition]:
    """Return runtime-safe brokered tool definitions from an AgentSpec."""

    return [
        BrokeredToolDefinition(
            name=tool.name,
            description=tool.description,
            brokered_class=tool.brokered_class,
            parameters=dict(tool.parameters),
            schema_digest=tool.schema_digest,
        )
        for tool in spec.brokered_tools
    ]


def brokered_tool_config_entry(
    *,
    name: str,
    description: str,
    brokered_class: str,
    parameters: Mapping[str, Any],
    include_digest: bool = True,
) -> dict[str, Any]:
    """Build one validated `agent.yaml` brokeredTools entry."""

    payload: dict[str, Any] = {
        "name": name,
        "description": description,
        "brokeredClass": brokered_class,
        "parameters": dict(parameters),
    }
    if include_digest:
        payload["schemaDigest"] = brokered_tool_schema_digest(
            name=name,
            description=description,
            brokered_class=brokered_class,
            parameters=dict(parameters),
        )
    validated = BrokeredToolSpec.model_validate(payload)
    out: dict[str, Any] = {
        "name": validated.name,
        "description": validated.description,
        "brokeredClass": validated.brokered_class,
        "parameters": validated.parameters,
    }
    if validated.schema_digest is not None:
        out["schemaDigest"] = validated.schema_digest
    return out


def _nested_get(mapping: Mapping[str, Any], *path: str) -> Any:
    current: Any = mapping
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _first_present(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def generate_brokered_tools_from_orka_tool_crds(
    documents: Iterable[Mapping[str, Any]],
    *,
    include_digest: bool = True,
) -> list[dict[str, Any]]:
    """Generate deterministic safe AgentKit brokeredTools config from Orka Tool CRDs.

    The Orka Tool CRD remains the source of truth. This helper deliberately
    extracts only the safe model-facing subset and validates it with the same
    `BrokeredToolSpec` model used by `agent.yaml` loading.
    """

    entries: list[dict[str, Any]] = []
    for idx, document in enumerate(documents):
        if not isinstance(document, Mapping) or not document:
            continue
        kind = str(document.get("kind") or "")
        if kind and kind.lower() != "tool":
            continue
        metadata = document.get("metadata") if isinstance(document.get("metadata"), Mapping) else {}
        spec = document.get("spec") if isinstance(document.get("spec"), Mapping) else {}
        name = _first_present(spec.get("name"), metadata.get("name"))
        if not isinstance(name, str):
            raise ValueError(f"Tool CRD document {idx} is missing metadata.name")
        description = _first_present(spec.get("description"), spec.get("summary"), name)
        if not isinstance(description, str):
            raise ValueError(f"Tool CRD {name!r} description must be a string")
        brokered_class = _first_present(
            spec.get("brokeredClass"),
            spec.get("brokered_class"),
            spec.get("class"),
            _nested_get(spec, "brokered", "class"),
            "read",
        )
        if not isinstance(brokered_class, str):
            raise ValueError(f"Tool CRD {name!r} brokered class must be a string")
        parameters = _first_present(
            spec.get("parameters"),
            spec.get("inputSchema"),
            spec.get("input_schema"),
            spec.get("schema"),
            _nested_get(spec, "input", "schema"),
        )
        if parameters is None:
            parameters = {"type": "object"}
        if not isinstance(parameters, Mapping):
            raise ValueError(f"Tool CRD {name!r} parameters schema must be an object")
        entries.append(
            brokered_tool_config_entry(
                name=name,
                description=description,
                brokered_class=brokered_class,
                parameters=parameters,
                include_digest=include_digest,
            )
        )
    return sorted(entries, key=lambda item: item["name"])


def load_orka_tool_crd_file(path: str | Path, *, include_digest: bool = True) -> list[dict[str, Any]]:
    """Load Tool CRD YAML/JSON documents and return safe brokeredTools entries."""

    raw = Path(path).read_text(encoding="utf-8")
    docs = [doc for doc in yaml.safe_load_all(raw) if isinstance(doc, Mapping)]
    return generate_brokered_tools_from_orka_tool_crds(docs, include_digest=include_digest)


def load_orka_tool_crd_files(paths: Sequence[str | Path], *, include_digest: bool = True) -> list[dict[str, Any]]:
    """Load one or more Tool CRD files and merge deterministic safe entries."""

    documents: list[Mapping[str, Any]] = []
    for path in paths:
        raw = Path(path).read_text(encoding="utf-8")
        documents.extend(doc for doc in yaml.safe_load_all(raw) if isinstance(doc, Mapping))
    entries = generate_brokered_tools_from_orka_tool_crds(documents, include_digest=include_digest)
    seen: set[str] = set()
    duplicates: set[str] = set()
    for entry in entries:
        name = str(entry["name"])
        if name in seen:
            duplicates.add(name)
        seen.add(name)
    if duplicates:
        names = ", ".join(sorted(duplicates))
        raise ValueError(f"duplicate brokered tool names across input files: {names}")
    return entries


def render_brokered_tools_yaml(entries: list[dict[str, Any]], *, bare: bool = False) -> str:
    """Render safe brokered tool entries as deterministic YAML."""

    payload: Any = entries if bare else {"brokeredTools": entries}
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="agentkit-brokered-tools",
        description="Export safe AgentKit brokeredTools YAML from Orka Tool CRD YAML/JSON files.",
    )
    parser.add_argument("paths", nargs="+", help="Orka Tool CRD YAML/JSON file(s) to export")
    parser.add_argument("--no-digest", action="store_true", help="omit schemaDigest fields")
    parser.add_argument("--bare", action="store_true", help="emit only the brokeredTools list, not a top-level brokeredTools key")
    parser.add_argument("--output", "-o", help="write output YAML to this file instead of stdout")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        entries = load_orka_tool_crd_files(args.paths, include_digest=not args.no_digest)
        rendered = render_brokered_tools_yaml(entries, bare=args.bare)
        if args.output:
            Path(args.output).write_text(rendered, encoding="utf-8")
        else:
            sys.stdout.write(rendered)
    except Exception as exc:  # noqa: BLE001 - CLI must print concise validation failures.
        print(f"agentkit-brokered-tools: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main() tests and console script.
    raise SystemExit(main())
