package abi

import (
	"os"
	"strings"
	"testing"

	"github.com/goccy/go-yaml"
	"github.com/sozercan/agentkit/pkg/agentkit/config"
)

// testAPIKeyEnvName is the NAME of an env var (not a secret). Hoisted to a const
// so gosec's G101 string-literal credential heuristic does not false-positive on
// the struct literal below.
const (
	testAPIKeyEnvName = "OPENAI_API_KEY" //nolint:gosec // G101: env var NAME, not a credential
	testInstructions  = "Be helpful and cite sources."
)

func sampleConfig() *config.AgentConfig {
	return &config.AgentConfig{
		APIVersion: "v1alpha1",
		Kind:       "Agent",
		Metadata:   config.Metadata{Name: "acme-support"},
		Runtime:    "pydantic-ai",
		Model: config.Model{
			Provider:  "openai-compatible",
			BaseURL:   "https://api.openai.com/v1",
			Name:      "gpt-4o-mini",
			APIKeyEnv: testAPIKeyEnvName,
		},
		Tools: []config.Tool{
			{Name: "fetch", Command: []string{"uvx", "mcp-server-fetch"}, Env: []string{"FETCH_TIMEOUT"}},
		},
		Expose: config.Expose{OpenAI: true, Port: 8080},
	}
}

func TestVersionAndPath(t *testing.T) {
	if Version != "v0" {
		t.Fatalf("Version = %q, want v0", Version)
	}
	if Path != "/agent/agent.yaml" {
		t.Fatalf("Path = %q, want /agent/agent.yaml", Path)
	}
}

func TestRenderAgentYAMLMatchesGolden(t *testing.T) {
	out, err := Render(sampleConfig(), testInstructions)
	if err != nil {
		t.Fatalf("render error: %v", err)
	}
	want, err := os.ReadFile("testdata/agent.yaml")
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	if string(out) != string(want) {
		t.Fatalf("rendered agent.yaml drifted from golden\n--- got ---\n%s\n--- want ---\n%s", out, want)
	}
}

func TestRenderAgentYAMLShape(t *testing.T) {
	out, err := Render(sampleConfig(), testInstructions)
	if err != nil {
		t.Fatalf("render error: %v", err)
	}
	s := string(out)

	// The reader is extra=forbid: assert exactly the allowed top-level keys are
	// present and no foreign key leaks in.
	mustContain := []string{"abiVersion:", "metadata:", "model:", "instructions:", "tools:", "expose:", "baseURL:", "apiKeyEnv:"}
	for _, k := range mustContain {
		if !strings.Contains(s, k) {
			t.Errorf("rendered agent.yaml missing %q\n---\n%s", k, s)
		}
	}
	// Keys the writer must NEVER emit (would trip extra=forbid or leak schema).
	mustNotContain := []string{"apiVersion:", "kind:", "provider: \n", "inline:", "file:"}
	for _, k := range mustNotContain {
		if strings.Contains(s, k) {
			t.Errorf("rendered agent.yaml unexpectedly contains %q\n---\n%s", k, s)
		}
	}
	if !strings.Contains(s, "abiVersion: v0") {
		t.Errorf("expected abiVersion: v0\n---\n%s", s)
	}
}

// TestRenderAgentYAMLRoundTrips renders the agent.yaml and parses it back with a
// strict (DisallowUnknownField) decoder to guard against the writer emitting any
// key the extra=forbid Python reader would reject. The cross-language proof —
// loading this package's golden with agentkit_serve_common.config.load — lives
// in runtimes/common/tests/test_abi_contract.py.
func TestRenderAgentYAMLRoundTrips(t *testing.T) {
	out, err := Render(sampleConfig(), testInstructions)
	if err != nil {
		t.Fatalf("render error: %v", err)
	}

	// A mirror of the ABI shape; strict decode fails on any unexpected key.
	var got struct {
		ABIVersion string `yaml:"abiVersion"`
		Metadata   struct {
			Name string `yaml:"name"`
		} `yaml:"metadata"`
		Model struct {
			Provider  string `yaml:"provider"`
			BaseURL   string `yaml:"baseURL"`
			Name      string `yaml:"name"`
			APIKeyEnv string `yaml:"apiKeyEnv"`
		} `yaml:"model"`
		Instructions string `yaml:"instructions"`
		Tools        []struct {
			Name    string   `yaml:"name"`
			Command []string `yaml:"command"`
			Env     []string `yaml:"env"`
		} `yaml:"tools"`
		Expose struct {
			OpenAI bool `yaml:"openai"`
			Port   int  `yaml:"port"`
		} `yaml:"expose"`
	}
	if err := yaml.UnmarshalWithOptions(out, &got, yaml.Strict()); err != nil {
		t.Fatalf("rendered agent.yaml has a key the strict reader rejects: %v\n---\n%s", err, out)
	}
	if got.ABIVersion != "v0" {
		t.Errorf("abiVersion = %q, want v0", got.ABIVersion)
	}
	if got.Model.BaseURL != "https://api.openai.com/v1" || got.Model.APIKeyEnv != "OPENAI_API_KEY" {
		t.Errorf("model aliases not emitted correctly: %+v", got.Model)
	}
	if len(got.Tools) != 1 || got.Tools[0].Name != "fetch" {
		t.Errorf("tools not round-tripped: %+v", got.Tools)
	}
}
