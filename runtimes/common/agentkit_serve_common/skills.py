"""Image-bundled skill instructions for governed runtimes.

Only SKILL.md documents are loaded. The resulting catalog is immutable and its
load_skill function does no I/O: it cannot read references, execute scripts, or
contact the services mentioned in a skill. Operational tools remain brokered.
"""

from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

import yaml

from .config import AgentSpec
from .yaml_support import safe_load_lossless

_SKILLS_ROOT = Path("/agent/skills")
_MAX_SKILL_BYTES = 128 * 1024
_MAX_CATALOG_BYTES = 1024 * 1024
_MAX_SKILLS = 64
_MAX_DIRECTORY_ENTRIES = 256
_MAX_DISCOVERY_DEPTH = 2
_NAME_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


class SkillConfigurationError(ValueError):
    """A skill configuration or document cannot be used in governed mode."""


@dataclass(frozen=True)
class _Skill:
    name: str
    description: str
    content: str


@dataclass(frozen=True)
class SkillCatalog:
    """Immutable instructions with a local load_skill tool that never performs I/O."""

    documents: tuple[_Skill, ...] = ()

    @classmethod
    def from_spec(cls, spec: AgentSpec) -> SkillCatalog:
        """Snapshot bounded, UTF-8 SKILL.md files from the image's skill directories."""
        validate_packaged_skill_providers(spec)
        documents: list[_Skill] = []
        total_bytes = 0
        names: set[str] = set()
        try:
            for provider in spec.context.providers:
                path = Path(provider.path or "")
                directory_fd = _open_directory(path)
                try:
                    discovered = _read_directory(directory_fd, path.name, depth=0)
                finally:
                    os.close(directory_fd)
                if not discovered:
                    raise SkillConfigurationError("filesystem skill directory contains no SKILL.md documents")
                for skill in discovered:
                    if skill.name in names:
                        raise SkillConfigurationError("duplicate bundled skill name")
                    names.add(skill.name)
                    total_bytes += len(skill.content.encode("utf-8"))
                    if len(names) > _MAX_SKILLS or total_bytes > _MAX_CATALOG_BYTES:
                        raise SkillConfigurationError("bundled skill catalog is too large")
                    documents.append(skill)
        except OSError as exc:
            # Keep filesystem exception details and any unexpected file paths out of
            # protocol responses. Symlinks and non-directories fail here as well.
            raise SkillConfigurationError(
                "filesystem skills must be readable image directories without symlinks"
            ) from exc
        return cls(tuple(sorted(documents, key=lambda skill: skill.name)))

    def __bool__(self) -> bool:
        return bool(self.documents)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(skill.name for skill in self.documents)

    @property
    def instructions(self) -> str:
        if not self.documents:
            return ""
        entries = "\n".join(
            f"<skill><name>{skill.name}</name><description>{escape(skill.description)}</description></skill>"
            for skill in self.documents
        )
        return (
            "The agent image includes these instruction-only skills:\n"
            f"<available_skills>\n{entries}\n</available_skills>\n"
            "When a skill applies, use load_skill with its skill_name to retrieve its "
            "instructions before following them. Skills supply guidance, not execution "
            "permissions. Use the authorized tools for operational data and actions."
        )

    def tool_schema(self) -> dict[str, Any]:
        """Return the local load_skill definition in OpenAI Chat tool format."""
        return {
            "type": "function",
            "function": {
                "name": "load_skill",
                "description": "Loads the full instructions for a bundled skill.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "skill_name": {
                            "type": "string",
                            "description": "The name of the skill to load.",
                            "enum": list(self.names),
                        }
                    },
                    "required": ["skill_name"],
                    "additionalProperties": False,
                },
            },
        }

    def load_skill(self, skill_name: str) -> str:
        """Read an already-loaded document, never a model-supplied file path."""
        if isinstance(skill_name, str):
            for skill in self.documents:
                if skill.name == skill_name:
                    return skill.content
        raise SkillConfigurationError("unknown bundled skill")


def validate_packaged_skill_providers(spec: AgentSpec) -> None:
    """Accept only filesystem skill declarations without external provider fields."""
    for provider in spec.context.providers:
        if provider.type != "skills" or provider.source != "filesystem":
            raise SkillConfigurationError("governed context providers must be instruction-only filesystem skills")
        if any(
            value is not None
            for value in (
                provider.tool_ref,
                provider.endpoint_env,
                provider.index,
                provider.index_env,
                provider.store_name_env,
                provider.auth,
            )
        ):
            raise SkillConfigurationError("filesystem skills must not configure external providers")
        path = Path(provider.path or "")
        if not path.is_absolute() or ".." in path.parts or not path.is_relative_to(_SKILLS_ROOT):
            raise SkillConfigurationError("filesystem skills must be under /agent/skills")


def _open_directory(path: Path) -> int:
    """Open each path component without following symlinks, including ancestors."""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_fd = os.open(path.anchor, flags)
    try:
        for component in path.parts[1:]:
            child_fd = os.open(component, flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = child_fd
        return directory_fd
    except BaseException:
        os.close(directory_fd)
        raise


def _read_directory(directory_fd: int, name: str, *, depth: int) -> list[_Skill]:
    entries: list[os.DirEntry[str]] = []
    with os.scandir(directory_fd) as scan:
        for entry in scan:
            if len(entries) >= _MAX_DIRECTORY_ENTRIES:
                raise SkillConfigurationError("bundled skill directory is too large")
            entries.append(entry)
    if any(entry.name == "SKILL.md" for entry in entries):
        return [_read_document(directory_fd, name)]
    documents: list[_Skill] = []
    for entry in sorted(entries, key=lambda value: value.name):
        if entry.is_symlink():
            raise SkillConfigurationError("bundled skill directories must not contain symlinks")
        if depth < _MAX_DISCOVERY_DEPTH and entry.is_dir(follow_symlinks=False):
            child_fd = os.open(entry.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory_fd)
            try:
                documents.extend(_read_directory(child_fd, entry.name, depth=depth + 1))
            finally:
                os.close(child_fd)
            if len(documents) > _MAX_SKILLS:
                raise SkillConfigurationError("bundled skill catalog is too large")
    return documents


def _read_document(directory_fd: int, directory_name: str) -> _Skill:
    document_fd = os.open("SKILL.md", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
    try:
        info = os.fstat(document_fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise SkillConfigurationError("SKILL.md must be a regular file without hard links")
        with os.fdopen(document_fd, "rb", closefd=False) as source:
            raw = source.read(_MAX_SKILL_BYTES + 1)
    finally:
        os.close(document_fd)
    if len(raw) > _MAX_SKILL_BYTES:
        raise SkillConfigurationError("SKILL.md exceeds the document size limit")
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SkillConfigurationError("SKILL.md must be UTF-8") from exc
    match = re.match(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", content, re.DOTALL)
    if match is None:
        raise SkillConfigurationError("SKILL.md must contain YAML frontmatter")
    try:
        metadata = safe_load_lossless(match.group(1))
    except (yaml.YAMLError, ValueError, RecursionError) as exc:
        raise SkillConfigurationError("SKILL.md frontmatter is invalid") from exc
    if not isinstance(metadata, dict):
        raise SkillConfigurationError("SKILL.md frontmatter must be a mapping")
    name, description = metadata.get("name"), metadata.get("description")
    if not isinstance(name, str) or len(name) > 64 or not _NAME_PATTERN.fullmatch(name) or name != directory_name:
        raise SkillConfigurationError(
            "SKILL.md name must match its directory and use lowercase letters, digits, and single hyphens"
        )
    if not isinstance(description, str) or not description.strip() or len(description) > 1024:
        raise SkillConfigurationError("SKILL.md description must contain 1 to 1024 characters")
    try:
        description.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise SkillConfigurationError("SKILL.md description must be valid Unicode") from exc
    if not content[match.end() :].strip() or "\x00" in content:
        raise SkillConfigurationError("SKILL.md must contain nonempty text instructions")
    return _Skill(name=name, description=description, content=content)
