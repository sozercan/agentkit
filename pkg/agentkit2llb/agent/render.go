package agent

import (
	"github.com/goccy/go-yaml"
	"github.com/sozercan/agentkit/pkg/agentkit/config"
	"github.com/sozercan/agentkit/pkg/utils"
)

// abiVersion is the schema version of the baked agent.yaml (docs/agent-abi.md).
// It MUST equal agentkit-serve's config.ABI_VERSION ("v0").
const abiVersion = "v0"

// The following types are the WRITER half of the frozen agent.yaml ABI
// (docs/agent-abi.md). agentkit-serve reads this file with pydantic
// extra="forbid", so these structs MUST emit EXACTLY the keys
// abiVersion/metadata/model/instructions/tools/expose — no more, no fewer — with
// model using the baseURL/apiKeyEnv aliases. Field order here is the emit order.

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
	Expose       abiExpose   `yaml:"expose"`
}

// renderAgentYAML produces the baked /agent/agent.yaml from the validated config
// and the already-resolved instructions scalar. The output is byte-compatible
// with agentkit-serve's strict (extra=forbid) reader.
func renderAgentYAML(cfg *config.AgentConfig, instructions string) ([]byte, error) {
	port := cfg.Expose.Port
	if port == 0 {
		port = utils.DefaultPort
	}

	out := abiAgent{
		ABIVersion: abiVersion,
		Metadata:   abiMetadata{Name: cfg.Metadata.Name},
		Model: abiModel{
			Provider:  cfg.Model.Provider,
			BaseURL:   cfg.Model.BaseURL,
			Name:      cfg.Model.Name,
			APIKeyEnv: cfg.Model.APIKeyEnv,
		},
		Instructions: instructions,
		Tools:        make([]abiTool, 0, len(cfg.Tools)),
		Expose:       abiExpose{OpenAI: cfg.Expose.OpenAI, Port: port},
	}
	for _, t := range cfg.Tools {
		out.Tools = append(out.Tools, abiTool{Name: t.Name, Command: t.Command, Env: t.Env})
	}

	return yaml.Marshal(out)
}
