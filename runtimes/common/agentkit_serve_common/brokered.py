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
from .yaml_support import safe_load_all_lossless

_ORKA_TOOL_API_VERSION = "core.orka.ai/v1alpha1"
_ORKA_TOOL_KIND = "Tool"
_ORKA_BROKERED_TOOL_CLASSES = ("read", "write", "coordination")


def _load_orka_tool_crd_documents(raw: str) -> list[Any]:
    return [doc for doc in safe_load_all_lossless(raw) if doc is not None]

_ORKA_TOOL_API_VERSION = "core.orka.ai/v1alpha1"
_ORKA_TOOL_KIND = "Tool"
_ORKA_BROKERED_TOOL_CLASSES = ("read", "write", "coordination")


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


def generate_brokered_tools_from_orka_tool_crds(
    documents: Iterable[Any],
    *,
    include_digest: bool = True,
) -> list[dict[str, Any]]:
    """Generate deterministic safe AgentKit brokeredTools config from Orka Tool CRDs.

    The Orka Tool CRD remains the source of truth. Only the canonical
    `core.orka.ai/v1alpha1` `Tool` shape is accepted. Tools without
    `spec.brokeredToolClass` are not brokered and are skipped; an input set with
    no brokered tools is rejected. Safe model-facing fields are validated with
    the same `BrokeredToolSpec` model used by `agent.yaml` loading.
    """

    entries: list[dict[str, Any]] = []
    for idx, document in enumerate(documents):
        if document is None:
            continue
        if not isinstance(document, Mapping):
            raise ValueError(f"Orka Tool CRD document {idx} must be an object")
        api_version = document.get("apiVersion")
        kind = document.get("kind")
        if api_version != _ORKA_TOOL_API_VERSION or kind != _ORKA_TOOL_KIND:
            raise ValueError(
                f"unsupported Orka Tool GVK in document {idx}: apiVersion={api_version!r}, kind={kind!r}; "
                f"expected apiVersion={_ORKA_TOOL_API_VERSION!r}, kind={_ORKA_TOOL_KIND!r}"
            )
        metadata = document.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"Orka Tool CRD document {idx} metadata must be an object")
        spec = document.get("spec")
        if not isinstance(spec, Mapping):
            raise ValueError(f"Orka Tool CRD document {idx} spec must be an object")
        if "brokeredToolClass" not in spec:
            continue
        name = metadata.get("name")
        if not isinstance(name, str):
            raise ValueError(f"Tool CRD document {idx} is missing metadata.name")
        description = spec.get("description")
        if not isinstance(description, str):
            raise ValueError(f"Tool CRD {name!r} description must be a string")
        brokered_class = spec["brokeredToolClass"]
        if not isinstance(brokered_class, str):
            raise ValueError(f"Tool CRD {name!r} spec.brokeredToolClass must be a string")
        if brokered_class not in _ORKA_BROKERED_TOOL_CLASSES:
            supported = ", ".join(_ORKA_BROKERED_TOOL_CLASSES)
            raise ValueError(
                f"Tool CRD {name!r} has unsupported spec.brokeredToolClass {brokered_class!r}; expected {supported}"
            )
        parameters = spec.get("parameters")
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
    if not entries:
        raise ValueError(
            "no brokered Orka Tool CRDs found; set spec.brokeredToolClass to read, write, or coordination"
        )
    return sorted(entries, key=lambda item: item["name"])


def load_orka_tool_crd_file(path: str | Path, *, include_digest: bool = True) -> list[dict[str, Any]]:
    """Load Tool CRD YAML/JSON documents and return safe brokeredTools entries."""

    raw = Path(path).read_text(encoding="utf-8")
    docs = _load_orka_tool_crd_documents(raw)
    return generate_brokered_tools_from_orka_tool_crds(docs, include_digest=include_digest)


def load_orka_tool_crd_files(paths: Sequence[str | Path], *, include_digest: bool = True) -> list[dict[str, Any]]:
    """Load one or more Tool CRD files and merge deterministic safe entries."""

    documents: list[Any] = []
    for path in paths:
        raw = Path(path).read_text(encoding="utf-8")
        documents.extend(_load_orka_tool_crd_documents(raw))
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
