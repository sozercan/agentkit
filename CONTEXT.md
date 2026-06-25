# AgentKit context

## Domain glossary

### Agent
A target-neutral description of a model, instructions, tools, and serving surface.
Users author an Agent in an `agentkitfile.yaml`; the frontend builds it into a
runtime Adapter image plus a baked `agent.yaml` file.

### Agentkitfile
The user-authored YAML input consumed by the Go BuildKit frontend. It is strict,
kind-discriminated input (`kind: Agent`) and may contain authored sources such as
inline instructions or an instructions file path.

### Effective Agent
The build-ready Agent value derived after the Agentkitfile is validated and all
build-time defaults/sources are resolved. It carries canonical runtime identity,
the effective serve port, and fully-resolved instructions so ABI and image writers
do not reinterpret raw authoring defaults.

### Agent YAML ABI
The frozen `/agent/agent.yaml` contract between the Go frontend writer and the
Python runtime Reader. The writer renders it from an Effective Agent; every
runtime Adapter reads the same ABI shape.

### Runtime Adapter
A concrete in-image implementation that reads the Agent YAML ABI and serves the
OpenAI-compatible `/v1` facade using a particular agent framework, such as
pydantic-ai or Microsoft Agent Framework.
