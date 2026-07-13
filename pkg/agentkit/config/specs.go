// Package config defines the AgentKit agentkitfile schema (kind: Agent) and the
// strict, kind-discriminated loader.
//
// This package deliberately diverges from AIKit's config loader in one
// correctness-critical way (plan §5.1 / §16.1): AIKit's NewFromBytes
// disambiguates inference-vs-finetune *implicitly* via a string-vs-struct type
// collision on the `config:` field, with non-strict yaml.v2. An agent file fed
// to that loader silently unmarshals as an empty InferenceConfig. AgentKit
// instead reads an explicit {apiVersion, kind} probe first and dispatches on
// it, using goccy/go-yaml in strict mode so field typos become load-time
// errors with line:column positions instead of silently-wrong builds.
package config

// Metadata carries identifying information for the agent image.
type Metadata struct {
	Name   string            `yaml:"name"`
	Labels map[string]string `yaml:"labels,omitempty"`
}

// Model describes the (hosted, openai-compatible) model the agent talks to.
// v0 supports provider: openai-compatible only. Local models are composed with
// a co-located AIKit container over baseURL — never baked (plan §6.5).
type Model struct {
	// Provider is the model provider. v0: "openai-compatible" only.
	Provider string `yaml:"provider"`
	// BaseURL is the OpenAI-compatible /v1 endpoint.
	BaseURL string `yaml:"baseURL"`
	// Name is the model name passed to the provider.
	Name string `yaml:"name"`
	// APIKeyEnv is the NAME of the env var holding the API key — never the value
	// (plan §3 INJECT axis). The value is provided via `docker run -e` at runtime.
	APIKeyEnv string `yaml:"apiKeyEnv,omitempty"`
	// Auth is reserved for generic model auth beyond apiKeyEnv. v0 validates this
	// behind runtime capability gates; no Foundry/Azure-specific keys live here.
	Auth *Auth `yaml:"auth,omitempty"`
}

// EnvVar declares one runtime environment variable the agent expects. Values are
// never stored in the agentkitfile or baked ABI; only the variable NAME and
// whether it must be present at runtime are recorded.
type EnvVar struct {
	Name     string `yaml:"name"`
	Required bool   `yaml:"required,omitempty"`
}

// Context groups provider-neutral external context sources such as search,
// skills, and memory. Provider-specific provisioning stays outside core.
type Context struct {
	Providers []ContextProvider `yaml:"providers,omitempty"`
}

// ContextProvider is a deliberately small provider-neutral shape. Only the
// fields needed by a provider type/source are populated.
type ContextProvider struct {
	Name         string `yaml:"name,omitempty"`
	Type         string `yaml:"type"`
	Source       string `yaml:"source,omitempty"`
	Path         string `yaml:"path,omitempty"`
	ToolRef      string `yaml:"toolRef,omitempty"`
	Index        string `yaml:"index,omitempty"`
	EndpointEnv  string `yaml:"endpointEnv,omitempty"`
	IndexEnv     string `yaml:"indexEnv,omitempty"`
	StoreNameEnv string `yaml:"storeNameEnv,omitempty"`
	Auth         *Auth  `yaml:"auth,omitempty"`
}

// Observability declares provider-neutral runtime observability knobs. Values
// are env var names, never literal connection strings.
type Observability struct {
	OTel ObservabilityOTel `yaml:"otel,omitempty"`
	Logs ObservabilityLogs `yaml:"logs,omitempty"`
}

type ObservabilityOTel struct {
	EndpointEnv string `yaml:"endpointEnv,omitempty"`
}

type ObservabilityLogs struct {
	LevelEnv string `yaml:"levelEnv,omitempty"`
}

// Expose declares how the built agent is reachable. v0 supports the OpenAI
// Chat-Completions façade only; mcp/a2a are v1 (strict parsing rejects them).
type Expose struct {
	// OpenAI enables POST /v1/chat/completions.
	OpenAI bool `yaml:"openai"`
	// Port is the serve port (default 8080).
	Port int `yaml:"port,omitempty"`
}

// AgentConfig is the parsed agentkitfile (kind: Agent). It models a
// target-neutral agent: model + instructions + tools + expose. The delivery
// target (standalone image, embedded worker) is a build-time --target choice
// and never enters this schema (plan §6.4).
type AgentConfig struct {
	APIVersion string   `yaml:"apiVersion"`
	Kind       string   `yaml:"kind"`
	Metadata   Metadata `yaml:"metadata"`
	Debug      bool     `yaml:"debug,omitempty"`
	// Runtime selects the runtime adapter. v0: "pydantic-ai" (the default).
	Runtime string `yaml:"runtime,omitempty"`
	// Model is the hosted model the agent uses.
	Model Model `yaml:"model"`
	// Instructions is the system prompt, authored inline (bare string or
	// {inline: ...}) or sourced from a file (plan §7 source union).
	Instructions Source `yaml:"instructions"`
	// Tools are AgentKit-owned MCP servers.
	Tools []Tool `yaml:"tools,omitempty"`
	// BrokeredTools are static safe Orka-brokered schemas for Foundry hosted Responses mode.
	BrokeredTools []BrokeredTool `yaml:"brokeredTools,omitempty"`
	// Env declares runtime env var requirements by NAME only. Values are injected
	// by the deployment/runtime environment and never baked into the image.
	Env []EnvVar `yaml:"env,omitempty"`
	// Context declares provider-neutral external context sources.
	Context Context `yaml:"context,omitempty"`
	// Observability declares optional logging/tracing env wiring.
	Observability Observability `yaml:"observability,omitempty"`
	// Expose declares the serving surface.
	Expose Expose `yaml:"expose"`
}
