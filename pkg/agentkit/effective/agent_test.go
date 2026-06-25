package effective

import (
	"testing"

	"github.com/sozercan/agentkit/pkg/agentkit/config"
	"github.com/sozercan/agentkit/pkg/agentkit/runtimes"
	"github.com/sozercan/agentkit/pkg/utils"
)

func baseConfig() *config.AgentConfig {
	return &config.AgentConfig{
		Metadata: config.Metadata{
			Name:   "acme-support",
			Labels: map[string]string{"team": "agentkit"},
		},
		Model: config.Model{
			Provider:  utils.ProviderOpenAICompatible,
			BaseURL:   "https://api.openai.com/v1",
			Name:      "gpt-4o-mini",
			APIKeyEnv: "OPENAI_API_KEY",
		},
		Tools: []config.Tool{
			{Name: "fetch", Command: []string{"uvx", "mcp-server-fetch"}, Env: []string{"FETCH_TIMEOUT"}},
		},
		Expose: config.Expose{OpenAI: true},
	}
}

func TestFromConfigDefaultsRuntimePortAndInstructions(t *testing.T) {
	agent := FromConfig(baseConfig(), "resolved prompt")

	if agent.Runtime != runtimes.PydanticAI {
		t.Fatalf("Runtime = %q, want %q", agent.Runtime, runtimes.PydanticAI)
	}
	if agent.Expose.Port != utils.DefaultPort {
		t.Fatalf("Expose.Port = %d, want %d", agent.Expose.Port, utils.DefaultPort)
	}
	if agent.Instructions != "resolved prompt" {
		t.Fatalf("Instructions = %q", agent.Instructions)
	}
}

func TestFromConfigCanonicalizesRuntimeAliasAndPreservesPort(t *testing.T) {
	cfg := baseConfig()
	cfg.Runtime = runtimes.MAFAlias
	cfg.Expose.Port = 9090

	agent := FromConfig(cfg, "prompt")

	if agent.Runtime != runtimes.MAF {
		t.Fatalf("Runtime = %q, want %q", agent.Runtime, runtimes.MAF)
	}
	if agent.Expose.Port != 9090 {
		t.Fatalf("Expose.Port = %d, want 9090", agent.Expose.Port)
	}
}

func TestFromConfigCopiesMutableFields(t *testing.T) {
	cfg := baseConfig()
	agent := FromConfig(cfg, "prompt")

	cfg.Metadata.Labels["team"] = "mutated"
	cfg.Tools[0].Command[0] = "mutated"
	cfg.Tools[0].Env[0] = "MUTATED"

	if agent.Metadata.Labels["team"] != "agentkit" {
		t.Fatalf("label was not copied: %#v", agent.Metadata.Labels)
	}
	if got := agent.Tools[0].Command[0]; got != "uvx" {
		t.Fatalf("command was not copied: %q", got)
	}
	if got := agent.Tools[0].Env[0]; got != "FETCH_TIMEOUT" {
		t.Fatalf("env was not copied: %q", got)
	}
}
