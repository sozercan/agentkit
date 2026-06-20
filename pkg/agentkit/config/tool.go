package config

// Tool is a tagged-union reference to an MCP server (plan §4.2: "tools are
// EXCLUSIVELY MCP servers"). v0 wires exactly one variant — a stdio command
// server (Command), matching the npm/PyPI reality of most MCP servers today and
// avoiding arbitrary-OCI rootfs staging (plan §5.2 ⚠, §11 v0 contract).
//
// The Image variant is declared so the schema shape is stable, but v0
// validation rejects it: arbitrary-OCI MCP staging via sidecar + captured OCI
// config is v1 (plan §11). Adding it later is filling in a code path, not a
// schema change.
type Tool struct {
	// Name is the tool's identifier, surfaced to the model.
	Name string `yaml:"name"`
	// Command is a stdio MCP server spawned as a subprocess, e.g.
	// ["npx", "-y", "@modelcontextprotocol/server-fetch"]. v0's only variant.
	Command []string `yaml:"command,omitempty"`
	// Image is an OCI MCP server image. Declared for schema stability; rejected
	// by v0 validation (v1: sidecar with captured OCI config, plan §5.2).
	Image string `yaml:"image,omitempty"`
	// Env lists the NAMES of env vars this tool's subprocess may read. Each MCP
	// subprocess is spawned with ONLY its declared env, not the full container
	// env (plan §10 "secret bleed across tools"). Values are injected at runtime.
	Env []string `yaml:"env,omitempty"`
}

// variantsSet returns the names of the populated tool-source variants.
func (t Tool) variantsSet() []string {
	var set []string
	if len(t.Command) > 0 {
		set = append(set, "command")
	}
	if t.Image != "" {
		set = append(set, "image")
	}
	return set
}
