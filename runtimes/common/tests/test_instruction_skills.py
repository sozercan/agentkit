from __future__ import annotations

import os
from pathlib import Path

import pytest

from agentkit_serve_common import skills
from agentkit_serve_common.config import AgentSpec
from agentkit_serve_common.skills import SkillCatalog, SkillConfigurationError


def _spec(*, providers=None) -> AgentSpec:
    return AgentSpec.model_validate(
        {
            "abiVersion": "v0",
            "metadata": {"name": "instruction-agent"},
            "model": {
                "provider": "openai-compatible",
                "baseURL": "https://model.example.invalid/v1",
                "name": "test-model",
            },
            "instructions": "Use the relevant skill.",
            "context": {
                "providers": providers
                if providers is not None
                else [
                    {
                        "type": "skills",
                        "source": "filesystem",
                        "path": "/agent/skills",
                    }
                ]
            },
            "expose": {"openai": True, "port": 8080},
        }
    )


@pytest.fixture
def skill_root(tmp_path, monkeypatch):
    root = tmp_path.resolve() / "skills"
    root.mkdir()
    open_directory = skills._open_directory

    # The packaged path is fixed; relocate it for tests without needing a host
    # /agent directory. All actual traversal and file-read checks still run.
    def open_test_directory(path: Path):
        return open_directory(root / path.relative_to("/agent/skills"))

    monkeypatch.setattr(skills, "_open_directory", open_test_directory)
    return root


def _write_skill(root, name="inspection", *, content=None):
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    content = content or (
        f"---\nname: {name}\ndescription: Inspect the relevant equipment.\n---\n"
        "Use the approved lookup tool to obtain the latest inspection record.\n"
    )
    (directory / "SKILL.md").write_text(content, encoding="utf-8")
    return content


def test_catalog_loads_exact_documents_and_never_reopens_files(skill_root):
    original = _write_skill(skill_root)
    _write_skill(
        skill_root,
        "parts",
        content=(
            "---\nname: parts\ndescription: Find parts & specifications.\n---\nRetrieve parts using authorized tools.\n"
        ),
    )
    (skill_root / "inspection" / "script.py").write_text("raise RuntimeError('must not execute')")
    (skill_root / "inspection" / "secret.txt").write_text("not a skill document")
    catalog = SkillCatalog.from_spec(_spec())

    assert catalog.names == ("inspection", "parts")
    assert "Find parts &amp; specifications." in catalog.instructions
    assert "secret.txt" not in catalog.instructions
    assert "script.py" not in catalog.instructions
    assert catalog.load_skill("inspection") == original
    (skill_root / "inspection" / "SKILL.md").write_text("changed after startup")
    (skill_root / "parts" / "SKILL.md").unlink()
    assert catalog.load_skill("inspection") == original
    assert "Retrieve parts" in catalog.load_skill("parts")
    definition = catalog.tool_schema()["function"]
    assert definition["name"] == "load_skill"
    assert definition["parameters"]["properties"]["skill_name"]["enum"] == ["inspection", "parts"]
    assert definition["parameters"]["additionalProperties"] is False


def test_nested_and_individual_skill_directories(skill_root):
    _write_skill(skill_root / "operations", "inspection")
    assert SkillCatalog.from_spec(_spec()).names == ("inspection",)
    spec = _spec(
        providers=[
            {
                "type": "skills",
                "source": "filesystem",
                "path": "/agent/skills/operations/inspection",
            }
        ]
    )
    assert SkillCatalog.from_spec(spec).names == ("inspection",)


@pytest.mark.parametrize("name", ["", "../inspection", "inspection/SKILL.md", None, {}, "unknown", "INSPECTION"])
def test_model_cannot_request_file_paths_or_unknown_skills(skill_root, name):
    _write_skill(skill_root)
    catalog = SkillCatalog.from_spec(_spec())
    with pytest.raises(SkillConfigurationError, match="unknown bundled skill"):
        catalog.load_skill(name)


def test_no_providers_needs_no_skill_directory(monkeypatch):
    monkeypatch.setattr(skills, "_open_directory", lambda _: pytest.fail("unexpected filesystem access"))
    catalog = SkillCatalog.from_spec(_spec(providers=[]))
    assert not catalog
    assert catalog.instructions == ""


@pytest.mark.parametrize(
    "provider",
    [
        {"type": "search", "endpointEnv": "SEARCH_ENDPOINT", "indexEnv": "SEARCH_INDEX"},
        {"type": "memory", "endpointEnv": "MEMORY_ENDPOINT", "storeNameEnv": "MEMORY_STORE"},
        {"type": "skills", "source": "filesystem", "path": "/agent/skills", "toolRef": "remote"},
        {"type": "skills", "source": "filesystem", "path": "/agent/skills", "endpointEnv": "REMOTE"},
    ],
)
def test_external_context_providers_remain_prohibited(provider):
    with pytest.raises(SkillConfigurationError, match="providers"):
        SkillCatalog.from_spec(_spec(providers=[provider]))


def test_path_traversal_remains_prohibited():
    spec = _spec(
        providers=[
            {
                "type": "skills",
                "source": "filesystem",
                "path": "/agent/skills/../skills",
            }
        ]
    )
    with pytest.raises(SkillConfigurationError, match="under /agent/skills"):
        SkillCatalog.from_spec(spec)


@pytest.mark.parametrize("target", ["root", "directory", "document"])
def test_skill_path_symlinks_fail_closed(skill_root, target):
    _write_skill(skill_root)
    if target == "root":
        link = skill_root
    elif target == "directory":
        link = skill_root / "inspection"
    else:
        link = skill_root / "inspection" / "SKILL.md"
    moved = link.with_name(link.name + "-original")
    link.rename(moved)
    link.symlink_to(moved, target_is_directory=moved.is_dir())
    with pytest.raises(SkillConfigurationError, match="symlinks"):
        SkillCatalog.from_spec(_spec())


@pytest.mark.parametrize("kind", ["hard-link", "fifo", "directory"])
def test_skill_document_must_be_an_unlinked_regular_file(skill_root, monkeypatch, kind):
    _write_skill(skill_root)
    path = skill_root / "inspection" / "SKILL.md"
    if kind == "hard-link":
        os.link(path, skill_root / "another-file")
    else:
        path.unlink()
        if kind == "fifo":
            os.mkfifo(path)
        else:
            path.mkdir()
    open_file = os.open
    opened = []

    def open_and_capture(name, flags, **kwargs):
        descriptor = open_file(name, flags, **kwargs)
        if name == "SKILL.md":
            opened.append(descriptor)
        return descriptor

    monkeypatch.setattr(skills.os, "open", open_and_capture)
    with pytest.raises(SkillConfigurationError):
        SkillCatalog.from_spec(_spec())
    assert len(opened) == 1
    with pytest.raises(OSError):
        os.fstat(opened[0])


@pytest.mark.parametrize(
    "content",
    [
        "No frontmatter.",
        "---\nname: another\ndescription: Description.\n---\nInstructions.",
        "---\nname: inspection\ndescription: 42\n---\nInstructions.",
        "---\nname: inspection\ndescription: Description.\n---\n  ",
        "---\nname: inspection\ndescription: Description.\n---\nInstructions.\x00",
        "---\nname: inspection\ndescription: !!python/object:unsafe {}\n---\nInstructions.",
        "---\n- inspection\n---\nInstructions.",
        "---\nname: inspection\nname: another\ndescription: Description.\n---\nInstructions.",
        '---\nname: inspection\ndescription: "\\ud800"\n---\nInstructions.',
    ],
)
def test_invalid_skill_documents_fail_startup(skill_root, content):
    _write_skill(skill_root, content=content)
    with pytest.raises(SkillConfigurationError, match="SKILL.md"):
        SkillCatalog.from_spec(_spec())


def test_invalid_utf8_document_fails_startup(skill_root):
    _write_skill(skill_root)
    (skill_root / "inspection" / "SKILL.md").write_bytes(b"\xff")
    with pytest.raises(SkillConfigurationError, match="UTF-8"):
        SkillCatalog.from_spec(_spec())


@pytest.mark.parametrize("limit", ["_MAX_SKILL_BYTES", "_MAX_CATALOG_BYTES", "_MAX_SKILLS", "_MAX_DIRECTORY_ENTRIES"])
def test_skill_catalog_limits_are_enforced(skill_root, monkeypatch, limit):
    _write_skill(skill_root)
    _write_skill(skill_root, "parts")
    monkeypatch.setattr(skills, limit, 1)
    with pytest.raises(SkillConfigurationError, match="size limit|too large"):
        SkillCatalog.from_spec(_spec())


def test_duplicate_skill_names_fail_startup(skill_root):
    _write_skill(skill_root / "first")
    _write_skill(skill_root / "second")
    with pytest.raises(SkillConfigurationError, match="duplicate"):
        SkillCatalog.from_spec(_spec())


def test_empty_or_missing_skill_directory_fails_startup(skill_root):
    with pytest.raises(SkillConfigurationError, match="no SKILL.md"):
        SkillCatalog.from_spec(_spec())
    skill_root.rmdir()
    with pytest.raises(SkillConfigurationError, match="readable image directories"):
        SkillCatalog.from_spec(_spec())


def test_directory_swap_cannot_redirect_the_open_skill_read(skill_root, monkeypatch):
    original = _write_skill(skill_root)
    outside = skill_root.parent / "replacement"
    _write_skill(
        outside, content=("---\nname: inspection\ndescription: Replacement.\n---\nReplacement instructions.\n")
    )
    read_document = skills._read_document

    def swap_before_read(directory_fd, directory_name):
        directory = skill_root / "inspection"
        directory.rename(skill_root / "original")
        directory.symlink_to(outside / "inspection", target_is_directory=True)
        return read_document(directory_fd, directory_name)

    monkeypatch.setattr(skills, "_read_document", swap_before_read)
    assert SkillCatalog.from_spec(_spec()).load_skill("inspection") == original
