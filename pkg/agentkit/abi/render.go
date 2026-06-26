// Package abi renders the frozen agent.yaml writer contract shared by the Go
// frontend and Python runtime readers.
package abi

import (
	"github.com/goccy/go-yaml"
	"github.com/sozercan/agentkit/pkg/agentkit/effective"
)

// Version is the schema version of the baked agent.yaml (docs/agent-abi.md).
// It MUST equal agentkit-serve's config.ABI_VERSION ("v0").
const Version = "v0"

// Path is where the rendered agent.yaml is baked into the agent image.
const Path = "/agent/agent.yaml"

// The following types are the WRITER half of the frozen agent.yaml ABI
// (docs/agent-abi.md). agentkit-serve reads this file with pydantic
// extra="forbid", so these structs MUST emit EXACTLY the keys
// abiVersion/metadata/model/instructions/tools/env/expose — with env omitted when
// empty — and model using the baseURL/apiKeyEnv aliases. Field order here is
// the emit order.

type abiMetadata struct {
	Name string `yaml:"name"`
}

type abiModel struct {
	Provider  string `yaml:"provider"`
	BaseURL   string `yaml:"baseURL"`
	Name      string `yaml:"name"`
	APIKeyEnv string `yaml:"apiKeyEnv,omitempty"`
}

type abiTool struct {
	Name    string   `yaml:"name"`
	Command []string `yaml:"command,omitempty"`
	Env     []string `yaml:"env,omitempty"`
}

type abiEnvVar struct {
	Name     string `yaml:"name"`
	Required bool   `yaml:"required,omitempty"`
}

type abiExpose struct {
	OpenAI bool `yaml:"openai"`
	Port   int  `yaml:"port"`
}

type abiAgent struct {
	ABIVersion   string      `yaml:"abiVersion"`
	Metadata     abiMetadata `yaml:"metadata"`
	Model        abiModel    `yaml:"model"`
	Instructions string      `yaml:"instructions"`
	Tools        []abiTool   `yaml:"tools"`
	Env          []abiEnvVar `yaml:"env,omitempty"`
	Expose       abiExpose   `yaml:"expose"`
}

// Render produces the baked /agent/agent.yaml from an effective Agent. The
// output is byte-compatible with agentkit-serve's strict (extra=forbid) reader.
func Render(agent effective.Agent) ([]byte, error) {
	out := abiAgent{
		ABIVersion: Version,
		Metadata:   abiMetadata{Name: agent.Metadata.Name},
		Model: abiModel{
			Provider:  agent.Model.Provider,
			BaseURL:   agent.Model.BaseURL,
			Name:      agent.Model.Name,
			APIKeyEnv: agent.Model.APIKeyEnv,
		},
		Instructions: agent.Instructions,
		Tools:        make([]abiTool, 0, len(agent.Tools)),
		Env:          make([]abiEnvVar, 0, len(agent.Env)),
		Expose:       abiExpose{OpenAI: agent.Expose.OpenAI, Port: agent.Expose.Port},
	}
	for _, t := range agent.Tools {
		out.Tools = append(out.Tools, abiTool{Name: t.Name, Command: t.Command, Env: t.Env})
	}
	for _, e := range agent.Env {
		out.Env = append(out.Env, abiEnvVar{Name: e.Name, Required: e.Required})
	}
	if len(out.Env) == 0 {
		out.Env = nil
	}

	return yaml.Marshal(out)
}
