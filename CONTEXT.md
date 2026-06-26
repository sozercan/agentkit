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

### Instruction Source
The authored source of an Agent's system prompt. v0 supports inline instructions
and file-backed instructions in the BuildKit context. The build resolves an
Instruction Source into the Effective Agent's instruction scalar before rendering
the Agent YAML ABI.

### Agent YAML ABI
The frozen `/agent/agent.yaml` contract between the Go frontend writer and the
Python runtime Reader. The writer renders it from an Effective Agent; every
runtime Adapter reads the same ABI shape.

### Conversation
The ordered OpenAI message list supplied to the `/v1/chat/completions` facade.
The shared runtime core normalizes it into a Run Request before any runtime
Adapter maps it to framework-specific message objects.

### Run Request
The framework-neutral request for one non-streaming agent run. It contains prior
Conversation turns plus the final user prompt, with client-owned tool turns and
unsupported roles removed. Runtime Adapters consume Run Requests instead of raw
OpenAI messages.

### Runtime Session
The live runtime Adapter handle used by the shared server during its lifespan. It
owns framework-specific agent lifecycle and runs normalized Run Requests, keeping
raw framework agent objects behind the Runtime Adapter Interface.

### Runtime Adapter
A concrete in-image implementation that reads the Agent YAML ABI and serves the
OpenAI-compatible `/v1` facade using a particular agent framework, such as
pydantic-ai or Microsoft Agent Framework.

### Runtime Capability
A provider-neutral feature flag declared by a Runtime Adapter (for example
`stdio-mcp`). The Agentkitfile validator checks requested features against the
runtime catalog before build so unsupported features fail clearly.

### Runtime Env Requirement
A top-level `env:` declaration in an Agentkitfile/Agent YAML ABI. It records only
an environment variable name and whether it is required; values stay in the
runtime environment and are never baked into the image.
