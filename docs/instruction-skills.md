# Bundled skills in governed mode

Skills let an agent load written guidance for a task before using its tools.
The Microsoft Agent Framework runtime can use the same bundled skills when
running under Orka harness v2 or the Foundry hosted brokered model loop.
Loading a skill reads instructions from the agent image; Orka still controls
operational tool calls and approvals.

Declare the skill directory in your AgentKitfile:

```yaml
runtime: microsoft-agent-framework
context:
  providers:
    - type: skills
      source: filesystem
      path: /agent/skills
```

Arrange each skill in a directory with a matching name:

```text
skills/
  inspection/
    SKILL.md
  parts-lookup/
    SKILL.md
```

For example, `skills/inspection/SKILL.md`:

```markdown
---
name: inspection
description: Prepare an equipment inspection using the current work order.
---
Retrieve the work order using the authorized lookup tool. Summarize the required
checks, identify missing information, and cite the returned record.
```

AgentKit resolves an `instructions.file` into the agent configuration, but does
not copy skill directories automatically. Add them to the built agent image
before composing its Orka runtime or deploying it to Foundry:

```dockerfile
FROM ghcr.io/acme/inspection-agent@sha256:<built-agent-image-digest>
COPY --chown=0:0 skills/ /agent/skills/
```

Make the directories and documents readable by the runtime user. Pin the image
that contains both the agent configuration and skills. For Orka ACP composition,
use that image's digest as `AGENTKIT_ADAPTER_DIGEST`; the existing
`agentConfigurationDigest` continues to cover the exact `agent.yaml` bytes.
Keep `/agent/skills` image-owned instead of mounting user or task workspace files
there. Changing a skill requires building and registering the updated image.

The runtime lists each skill's name and description for the model. The model
calls `load_skill` with `{"skill_name":"inspection"}` to receive the original
`SKILL.md` text, including its frontmatter. This works with upstream instruction
skills that already use `load_skill`.

Only `SKILL.md` documents are loaded. Files are snapshotted during startup and
tool calls use that immutable snapshot. Sibling files and scripts are not made
available, and the runtime does not expose `read_skill_resource` or
`run_skill_script`. A skill's text cannot grant tool access or bypass an
approval. Remote/search/memory context providers remain rejected in governed
mode.

Each skill needs a matching lowercase name of at most 64 characters, a
description of at most 1024 characters, and nonempty instructions. Discovery
covers the selected directory and two directory levels below it. Duplicate
names, symlinked paths, nonregular or hardlinked documents, invalid UTF-8, and
invalid frontmatter fail startup. Each document is limited to 128 KiB; a catalog
is limited to 64 skills and 1 MiB of document text. A directory may contain at
most 256 entries.

The hosted brokered loop must reserve `load_skill` for the local catalog when
skills are configured. Give operational tools their own names and schemas.
For the rest of the hosted setup, see
[Foundry hosted brokered tools](foundry-hosted-brokered.md).
