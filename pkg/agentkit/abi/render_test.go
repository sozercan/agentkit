package abi

import (
	"encoding/json"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/goccy/go-yaml"
	"github.com/sozercan/agentkit/pkg/agentkit/config"
	"github.com/sozercan/agentkit/pkg/agentkit/effective"
)

// testAPIKeyEnvName is the NAME of an env var (not a secret). Hoisted to a const
// so gosec's G101 string-literal credential heuristic does not false-positive on
// the struct literal below.
const (
	testAPIKeyEnvName = "OPENAI_API_KEY" //nolint:gosec // G101: env var NAME, not a credential
	testInstructions  = "Be helpful and cite sources."
)

const (
	jsonSchemaTypeKey       = "type"
	jsonSchemaTypeObject    = "object"
	jsonSchemaTypeNumber    = "number"
	jsonSchemaPropertiesKey = "properties"
	jsonSchemaMinimumKey    = "minimum"
	jsonSchemaDefaultKey    = "default"
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

func sampleAgent() effective.Agent {
	return effective.FromConfig(sampleConfig(), testInstructions)
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
	out, err := Render(sampleAgent())
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
	out, err := Render(sampleAgent())
	if err != nil {
		t.Fatalf("render error: %v", err)
	}
	s := string(out)

	// The reader is extra=forbid: assert the required top-level keys are present
	// and no foreign key leaks in. Optional env is covered below.
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
	out, err := Render(sampleAgent())
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
		Env []struct {
			Name     string `yaml:"name"`
			Required bool   `yaml:"required"`
		} `yaml:"env"`
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

func TestRenderAgentYAMLIncludesEnvRequirements(t *testing.T) {
	cfg := sampleConfig()
	cfg.Env = []config.EnvVar{
		{Name: "REQUIRED_FOO", Required: true},
		{Name: "OPTIONAL_BAR"},
	}
	out, err := Render(effective.FromConfig(cfg, testInstructions))
	if err != nil {
		t.Fatalf("render error: %v", err)
	}

	s := string(out)
	for _, want := range []string{"env:", "name: REQUIRED_FOO", "required: true", "name: OPTIONAL_BAR"} {
		if !strings.Contains(s, want) {
			t.Fatalf("rendered agent.yaml missing %q\n---\n%s", want, s)
		}
	}
	if strings.Contains(s, "required: false") {
		t.Fatalf("optional env vars should omit required: false\n---\n%s", s)
	}
}

func TestRenderAgentYAMLIncludesRemoteMCPTool(t *testing.T) {
	cfg := sampleConfig()
	cfg.Tools = []config.Tool{{
		Name:      "toolbox",
		Type:      config.ToolTypeMCP,
		Transport: config.ToolTransportStreamableHTTP,
		URLEnv:    "TOOLBOX_ENDPOINT",
		Headers: []config.ToolHeader{{
			Name:  "Foundry-Features",
			Value: "Toolboxes=V1Preview",
		}},
		Auth: &config.Auth{Type: config.AuthTypeBearer, TokenEnv: "TOOLBOX_TOKEN"},
	}}
	out, err := Render(effective.FromConfig(cfg, testInstructions))
	if err != nil {
		t.Fatalf("render error: %v", err)
	}

	s := string(out)
	for _, want := range []string{
		"type: mcp",
		"transport: streamable-http",
		"urlEnv: TOOLBOX_ENDPOINT",
		"name: Foundry-Features",
		"value: Toolboxes=V1Preview",
		"auth:",
		"type: bearer",
		"tokenEnv: TOOLBOX_TOKEN",
	} {
		if !strings.Contains(s, want) {
			t.Fatalf("rendered agent.yaml missing %q\n---\n%s", want, s)
		}
	}
}

func TestRenderAgentYAMLIncludesContextAndObservability(t *testing.T) {
	cfg := sampleConfig()
	cfg.Context.Providers = []config.ContextProvider{{
		Name:        "knowledge",
		Type:        config.ContextTypeSearch,
		EndpointEnv: "SEARCH_ENDPOINT",
		IndexEnv:    "SEARCH_INDEX",
	}}
	cfg.Observability.OTel.EndpointEnv = "OTEL_EXPORTER_OTLP_ENDPOINT"
	out, err := Render(effective.FromConfig(cfg, testInstructions))
	if err != nil {
		t.Fatalf("render error: %v", err)
	}

	s := string(out)
	for _, want := range []string{
		"context:",
		"providers:",
		"name: knowledge",
		"type: search",
		"endpointEnv: SEARCH_ENDPOINT",
		"indexEnv: SEARCH_INDEX",
		"observability:",
		"endpointEnv: OTEL_EXPORTER_OTLP_ENDPOINT",
	} {
		if !strings.Contains(s, want) {
			t.Fatalf("rendered agent.yaml missing %q\n---\n%s", want, s)
		}
	}
}

func TestRenderAgentYAMLIncludesModelWorkloadIdentityAuth(t *testing.T) {
	cfg := sampleConfig()
	cfg.Model.Auth = &config.Auth{Type: config.AuthTypeWorkloadIdentity, Audience: "https://ai.azure.com/.default"}
	out, err := Render(effective.FromConfig(cfg, testInstructions))
	if err != nil {
		t.Fatalf("render error: %v", err)
	}
	s := string(out)
	for _, want := range []string{"auth:", "type: workload-identity-token", "audience: https://ai.azure.com/.default"} {
		if !strings.Contains(s, want) {
			t.Fatalf("rendered agent.yaml missing %q\n---\n%s", want, s)
		}
	}
}

func TestRenderAgentYAMLIncludesBrokeredTools(t *testing.T) {
	cfg := sampleConfig()
	cfg.Tools = nil
	cfg.BrokeredTools = []config.BrokeredTool{{
		Name:          "check-network-telemetry",
		Description:   "Read telemetry.",
		BrokeredClass: config.BrokeredClassRead,
		Parameters: map[string]any{
			jsonSchemaTypeKey: jsonSchemaTypeObject,
			jsonSchemaPropertiesKey: map[string]any{
				"site":        map[string]any{jsonSchemaTypeKey: "string", jsonSchemaMinimumKey: 0.000001},
				"typedFloats": map[string]float64{jsonSchemaMinimumKey: 0.000001},
				"empty":       map[string]any{},
			},
		},
	}}
	out, err := Render(effective.FromConfig(cfg, testInstructions))
	if err != nil {
		t.Fatalf("render error: %v", err)
	}
	s := string(out)
	for _, want := range []string{"brokeredTools:", "name: check-network-telemetry", "description: Read telemetry.", "brokeredClass: read", "properties:", "site:", "minimum: 0.000001", "typedFloats:", "empty: {}"} {
		if !strings.Contains(s, want) {
			t.Fatalf("rendered agent.yaml missing %q\n---\n%s", want, s)
		}
	}
	for _, never := range []string{"url:", "secretRef:", "headers:", "auth" + ":", "1e-06"} {
		if strings.Contains(s, never) {
			t.Fatalf("rendered brokered agent.yaml leaked %q\n---\n%s", never, s)
		}
	}
}

func TestRenderAgentYAMLFormatsJSONNumberBrokeredSchemaValuesAsNumbers(t *testing.T) {
	cfg := sampleConfig()
	cfg.Tools = nil
	cfg.BrokeredTools = []config.BrokeredTool{{
		Name:          "numeric-tool",
		Description:   "Read numeric data.",
		BrokeredClass: config.BrokeredClassRead,
		Parameters: map[string]any{
			jsonSchemaTypeKey: jsonSchemaTypeObject,
			jsonSchemaPropertiesKey: map[string]any{
				"small": map[string]any{jsonSchemaTypeKey: jsonSchemaTypeNumber, jsonSchemaMinimumKey: json.Number("1e-7")},
			},
		},
	}}

	out, err := Render(effective.FromConfig(cfg, testInstructions))
	if err != nil {
		t.Fatalf("render error: %v", err)
	}
	if !strings.Contains(string(out), "minimum: 0.0000001") || strings.Contains(string(out), "1e-7") {
		t.Fatalf("rendered agent.yaml did not preserve json.Number as fixed YAML number\n---\n%s", out)
	}
}

func TestRenderAgentYAMLPreservesNegativeZeroBrokeredSchemaFloats(t *testing.T) {
	cfg := sampleConfig()
	cfg.Tools = nil
	tool := config.BrokeredTool{
		Name:          "negative-zero-tool",
		Description:   "Preserve negative zero.",
		BrokeredClass: config.BrokeredClassRead,
		Parameters: map[string]any{
			jsonSchemaTypeKey: jsonSchemaTypeObject,
			jsonSchemaPropertiesKey: map[string]any{
				"offset":        map[string]any{jsonSchemaTypeKey: jsonSchemaTypeNumber, jsonSchemaDefaultKey: math.Copysign(0, -1)},
				"jsonOffset":    map[string]any{jsonSchemaTypeKey: jsonSchemaTypeNumber, jsonSchemaDefaultKey: json.Number("-0e0")},
				"largeInteger":  map[string]any{jsonSchemaTypeKey: "integer", jsonSchemaDefaultKey: json.Number("9007199254740995.0")},
				"typedNested":   map[string][]float64{"enum": {math.Copysign(0, -1)}},
				"typedNil":      map[string]any{jsonSchemaDefaultKey: map[string]string(nil)},
				"typedNilSlice": map[string]any{jsonSchemaDefaultKey: []string(nil)},
			},
		},
	}
	digest, err := config.BrokeredToolSchemaDigest(tool)
	if err != nil {
		t.Fatalf("digest error: %v", err)
	}
	tool.SchemaDigest = digest
	cfg.BrokeredTools = []config.BrokeredTool{tool}

	out, err := Render(effective.FromConfig(cfg, testInstructions))
	if err != nil {
		t.Fatalf("render error: %v", err)
	}
	if strings.Count(string(out), "default: -0.0") != 2 {
		t.Fatalf("rendered agent.yaml did not preserve negative zero as a float\n---\n%s", out)
	}
	if !strings.Contains(string(out), "default: 9007199254740995") || strings.Contains(string(out), "9007199254740995.0") {
		t.Fatalf("rendered agent.yaml did not preserve a large integral decimal as an integer\n---\n%s", out)
	}
	if strings.Count(string(out), yamlNegativeZero) != 3 {
		t.Fatalf("rendered agent.yaml did not normalize negative zero in typed nested containers\n---\n%s", out)
	}
	if strings.Count(string(out), "default: null") != 2 {
		t.Fatalf("rendered agent.yaml did not preserve typed nil containers as null\n---\n%s", out)
	}
}
