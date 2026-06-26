package config

// ToolHeader declares one HTTP header for a remote MCP tool. Values may be
// static non-secret literals (for feature flags) or come from env vars by NAME.
type ToolHeader struct {
	Name     string `yaml:"name"`
	Value    string `yaml:"value,omitempty"`
	ValueEnv string `yaml:"valueEnv,omitempty"`
}

// Auth declares generic authentication for remote tools/resources. The audience
// value is opaque to AgentKit core; deployment/runtime adapters decide how to
// mint workload tokens for it.
type Auth struct {
	Type     string `yaml:"type"`
	TokenEnv string `yaml:"tokenEnv,omitempty"`
	Audience string `yaml:"audience,omitempty"`
}

// Tool is a tagged-union reference to an MCP server (plan §4.2: "tools are
// EXCLUSIVELY MCP servers"). Stdio command tools use Command. Remote MCP tools
// use Type=mcp, Transport=streamable-http, and URLEnv.
//
// The Image variant is declared so the schema shape is stable, but v0
// validation rejects it: arbitrary-OCI MCP staging via sidecar + captured OCI
// config is v1 (plan §11). Adding it later is filling in a code path, not a
// schema change.
type Tool struct {
	// Name is the tool's identifier, surfaced to the model.
	Name string `yaml:"name"`
	// Type is optional for legacy stdio tools. Remote MCP tools must set "mcp".
	Type string `yaml:"type,omitempty"`
	// Transport selects the MCP transport. Empty means stdio for command tools;
	// remote tools currently support "streamable-http".
	Transport string `yaml:"transport,omitempty"`
	// Command is a stdio MCP server spawned as a subprocess, e.g.
	// ["npx", "-y", "@modelcontextprotocol/server-fetch"].
	Command []string `yaml:"command,omitempty"`
	// URLEnv is the NAME of the env var containing a Streamable HTTP MCP URL.
	URLEnv string `yaml:"urlEnv,omitempty"`
	// Headers are extra HTTP headers for remote MCP tools. Header values are either
	// static non-secret values or env-derived values.
	Headers []ToolHeader `yaml:"headers,omitempty"`
	// Auth declares remote MCP authentication. Stdio tools must not set auth.
	Auth *Auth `yaml:"auth,omitempty"`
	// Approval is a provider-neutral policy placeholder: "never", "auto", or
	// "always". Current runtimes do not implement HITL approval, so non-empty
	// policies are capability-gated.
	Approval string `yaml:"approval,omitempty"`
	// Image is an OCI MCP server image. Declared for schema stability; rejected
	// by v0 validation (v1: sidecar with captured OCI config, plan §5.2).
	Image string `yaml:"image,omitempty"`
	// Env lists the NAMES of env vars this stdio tool's subprocess may read. Each
	// MCP subprocess is spawned with ONLY its declared env, not the full container
	// env (plan §10 "secret bleed across tools"). Values are injected at runtime.
	Env []string `yaml:"env,omitempty"`
}

// variantsSet returns the names of the populated tool-source variants.
func (t Tool) variantsSet() []string {
	var set []string
	if len(t.Command) > 0 {
		set = append(set, "command")
	}
	if t.URLEnv != "" {
		set = append(set, "urlEnv")
	}
	if t.Image != "" {
		set = append(set, "image")
	}
	return set
}

func (t Tool) isRemoteMCP() bool {
	return t.URLEnv != "" || t.Transport == "streamable-http"
}
