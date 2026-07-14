package effective

import (
	"encoding/json"
	"math/big"
	"testing"
	"time"

	"github.com/sozercan/agentkit/pkg/agentkit/config"
	"github.com/sozercan/agentkit/pkg/agentkit/runtimes"
	"github.com/sozercan/agentkit/pkg/utils"
)

const (
	testAPIKeyEnvName = "OPENAI_API_KEY" //nolint:gosec // G101: env var NAME, not a credential
	mutatedValue      = "MUTATED"
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
			APIKeyEnv: testAPIKeyEnvName,
		},
		Tools: []config.Tool{
			{
				Name:    "fetch",
				Command: []string{"uvx", "mcp-server-fetch"},
				Env:     []string{"FETCH_TIMEOUT"},
				Headers: []config.ToolHeader{{Name: "X-Trace", ValueEnv: "TRACE_HEADER"}},
				Auth:    &config.Auth{Type: config.AuthTypeBearer, TokenEnv: "TOOL_TOKEN"},
			},
		},
		Env:    []config.EnvVar{{Name: "REQUIRED_FOO", Required: true}},
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

	cfg.Metadata.Labels["team"] = mutatedValue
	cfg.Tools[0].Command[0] = "mutated"
	cfg.Tools[0].Env[0] = mutatedValue
	cfg.Tools[0].Headers[0].Name = mutatedValue
	cfg.Tools[0].Auth.TokenEnv = mutatedValue
	cfg.Env[0].Name = mutatedValue

	if agent.Metadata.Labels["team"] != "agentkit" {
		t.Fatalf("label was not copied: %#v", agent.Metadata.Labels)
	}
	if got := agent.Tools[0].Command[0]; got != "uvx" {
		t.Fatalf("command was not copied: %q", got)
	}
	if got := agent.Tools[0].Env[0]; got != "FETCH_TIMEOUT" {
		t.Fatalf("tool env was not copied: %q", got)
	}
	if got := agent.Tools[0].Headers[0].Name; got != "X-Trace" {
		t.Fatalf("tool headers were not copied: %q", got)
	}
	if got := agent.Tools[0].Auth.TokenEnv; got != "TOOL_TOKEN" {
		t.Fatalf("tool auth was not copied: %q", got)
	}
	if got := agent.Env[0].Name; got != "REQUIRED_FOO" {
		t.Fatalf("agent env was not copied: %q", got)
	}
}

const (
	testSiteField        = "site"
	testSchemaTypeKey    = "type"
	testSchemaTypeString = "string"
)

func TestFromConfigCopiesBrokeredTools(t *testing.T) {
	type schemaValue struct {
		Type  string
		Enum  []string
		Value any
	}
	cfg := baseConfig()
	pointerNumber := json.Number("1.0")
	nilMap := map[string]string(nil)
	nilSlice := []string(nil)
	emptySlice := []string{}
	structPointer := &schemaValue{Type: testSchemaTypeString, Enum: []string{"a", "b"}, Value: int64(9007199254740993)}
	typedStructs := []schemaValue{{Type: testSchemaTypeString, Enum: []string{"a", "b"}, Value: int64(9007199254740993)}}
	typedPointers := []*schemaValue{{Type: testSchemaTypeString, Enum: []string{"a", "b"}, Value: int64(9007199254740993)}}
	timestamp := time.Date(2026, time.July, 11, 12, 0, 0, 0, time.UTC)
	bigInteger := big.NewInt(123)
	largePointer := &map[string]any{"value": int64(9007199254740993)}
	cfg.BrokeredTools = []config.BrokeredTool{{
		Name:          "check-network-telemetry",
		Description:   "Read telemetry.",
		BrokeredClass: config.BrokeredClassRead,
		Parameters: map[string]any{
			testSchemaTypeKey: "object",
			"properties": map[string]any{
				testSiteField:   map[string]any{testSchemaTypeKey: testSchemaTypeString},
				"typed":         map[string]string{testSchemaTypeKey: testSchemaTypeString},
				"generic":       map[string][]string{"enum": {"a", "b"}},
				"nilArray":      [1]map[string]string{nil},
				"nilPointer":    &nilMap,
				"nilSlice":      &nilSlice,
				"emptySlice":    &emptySlice,
				"pointer":       &pointerNumber,
				"struct":        structPointer,
				"typedStructs":  typedStructs,
				"typedPointers": typedPointers,
				"timestamp":     &timestamp,
				"bigInteger":    bigInteger,
				"largePointer":  largePointer,
				"tuple":         []map[string]any{{testSchemaTypeKey: testSchemaTypeString}},
				"empty":         map[string]any{},
			},
			"required": []string{testSiteField},
		},
	}}

	agent := FromConfig(cfg, "prompt")
	cfg.BrokeredTools[0].Parameters[testSchemaTypeKey] = mutatedValue
	mutatedProperties, ok := cfg.BrokeredTools[0].Parameters["properties"].(map[string]any)
	if !ok {
		t.Fatalf("brokered tool properties had unexpected type: %#v", cfg.BrokeredTools[0].Parameters["properties"])
	}
	mutatedSite, ok := mutatedProperties[testSiteField].(map[string]any)
	if !ok {
		t.Fatalf("brokered tool site property had unexpected type: %#v", mutatedProperties["site"])
	}
	mutatedSite[testSchemaTypeKey] = mutatedValue
	mutatedTyped, ok := mutatedProperties["typed"].(map[string]string)
	if !ok {
		t.Fatalf("brokered typed property had unexpected type: %#v", mutatedProperties["typed"])
	}
	mutatedTyped[testSchemaTypeKey] = mutatedValue
	mutatedRequired, ok := cfg.BrokeredTools[0].Parameters["required"].([]string)
	if !ok {
		t.Fatalf("brokered tool required slice had unexpected type: %#v", cfg.BrokeredTools[0].Parameters["required"])
	}
	mutatedRequired[0] = mutatedValue
	mutatedGeneric, ok := mutatedProperties["generic"].(map[string][]string)
	if !ok {
		t.Fatalf("brokered generic property had unexpected type: %#v", mutatedProperties["generic"])
	}
	mutatedGeneric["enum"][0] = mutatedValue
	mutatedTuple, ok := mutatedProperties["tuple"].([]map[string]any)
	if !ok {
		t.Fatalf("brokered tuple property had unexpected type: %#v", mutatedProperties["tuple"])
	}
	mutatedTuple[0][testSchemaTypeKey] = mutatedValue
	pointerNumber = json.Number("2.0")
	structPointer.Enum[0] = mutatedValue
	typedStructs[0].Enum[0] = mutatedValue
	typedPointers[0].Enum[0] = mutatedValue
	timestamp = time.Time{}
	bigInteger.SetInt64(456)
	(*largePointer)["value"] = int64(1)

	if got := agent.BrokeredTools[0].Parameters[testSchemaTypeKey]; got != "object" {
		t.Fatalf("brokered tool parameters were not copied: %q", got)
	}
	properties, ok := agent.BrokeredTools[0].Parameters["properties"].(map[string]any)
	if !ok {
		t.Fatalf("brokered tool properties had unexpected type: %#v", agent.BrokeredTools[0].Parameters["properties"])
	}
	site, ok := properties[testSiteField].(map[string]any)
	if !ok {
		t.Fatalf("brokered tool site property had unexpected type: %#v", properties["site"])
	}
	if got := site[testSchemaTypeKey]; got != "string" {
		t.Fatalf("nested brokered tool parameters were not copied: %q", got)
	}
	typed, ok := properties["typed"].(map[string]string)
	if !ok || typed[testSchemaTypeKey] != testSchemaTypeString {
		t.Fatalf("typed brokered tool schema map was not copied: %#v", properties["typed"])
	}
	empty, ok := properties["empty"].(map[string]any)
	if !ok || empty == nil || len(empty) != 0 {
		t.Fatalf("empty brokered tool schema map was not preserved: %#v", properties["empty"])
	}
	generic, ok := properties["generic"].(map[string][]string)
	if !ok || generic["enum"][0] != "a" {
		t.Fatalf("generic typed brokered schema map was not copied: %#v", properties["generic"])
	}
	tuple, ok := properties["tuple"].([]map[string]any)
	if !ok || tuple[0][testSchemaTypeKey] != testSchemaTypeString {
		t.Fatalf("typed brokered schema slice was not copied: %#v", properties["tuple"])
	}
	pointer, ok := properties["pointer"].(json.Number)
	if !ok || pointer.String() != "1.0" {
		t.Fatalf("pointer-valued brokered schema value was not deep-copied: %#v", properties["pointer"])
	}
	if properties["nilPointer"] != nil {
		t.Fatalf("pointer to nil typed map was not preserved: %#v", properties["nilPointer"])
	}
	nilArray, ok := properties["nilArray"].([1]map[string]string)
	if !ok || nilArray[0] != nil {
		t.Fatalf("nil typed map in array was not preserved: %#v", properties["nilArray"])
	}
	if properties["nilSlice"] != nil {
		t.Fatalf("pointer to nil typed slice was not preserved: %#v", properties["nilSlice"])
	}
	emptySliceCopy, ok := properties["emptySlice"].([]any)
	if !ok || emptySliceCopy == nil || len(emptySliceCopy) != 0 {
		t.Fatalf("pointer to non-nil empty typed slice was not preserved: %#v", properties["emptySlice"])
	}
	structCopy, ok := properties["struct"].(map[string]any)
	structEnum, enumOK := structCopy["Enum"].([]any)
	structValue, valueOK := structCopy["Value"].(json.Number)
	if !ok || !enumOK || !valueOK || structCopy["Type"] != testSchemaTypeString || structEnum[0] != "a" || structValue.String() != "9007199254740993" {
		t.Fatalf("pointer to struct with slice field was not deep-copied: %#v", properties["struct"])
	}
	typedStructCopies, ok := properties["typedStructs"].([]schemaValue)
	if !ok || len(typedStructCopies) != 1 {
		t.Fatalf("typed struct slice had unexpected shape: %#v", properties["typedStructs"])
	}
	typedStructValue, typedStructValueOK := typedStructCopies[0].Value.(json.Number)
	if !typedStructValueOK || typedStructCopies[0].Enum[0] != "a" || typedStructValue.String() != "9007199254740993" {
		t.Fatalf("typed struct slice was not deep-copied: %#v", properties["typedStructs"])
	}
	typedPointerCopies, ok := properties["typedPointers"].([]*schemaValue)
	if !ok || len(typedPointerCopies) != 1 || typedPointerCopies[0] == nil {
		t.Fatalf("typed pointer slice had unexpected shape: %#v", properties["typedPointers"])
	}
	typedPointerValue, typedPointerValueOK := typedPointerCopies[0].Value.(json.Number)
	if !typedPointerValueOK || typedPointerCopies[0] == typedPointers[0] || typedPointerCopies[0].Enum[0] != "a" || typedPointerValue.String() != "9007199254740993" {
		t.Fatalf("typed pointer slice was not deep-copied: %#v", properties["typedPointers"])
	}
	timestampCopy, ok := properties["timestamp"].(string)
	if !ok || timestampCopy != "2026-07-11T12:00:00Z" {
		t.Fatalf("pointer to opaque struct was not preserved: %#v", properties["timestamp"])
	}
	bigIntegerCopy, ok := properties["bigInteger"].(json.Number)
	if !ok || bigIntegerCopy.String() != "123" {
		t.Fatalf("pointer to mutable opaque struct was not deep-copied: %#v", properties["bigInteger"])
	}
	largeCopy, ok := properties["largePointer"].(map[string]any)
	largeValue, valueOK := largeCopy["value"].(json.Number)
	if !ok || !valueOK || largeValue.String() != "9007199254740993" {
		t.Fatalf("pointer-held large integer lost precision during copy: %#v", properties["largePointer"])
	}
	required, ok := agent.BrokeredTools[0].Parameters["required"].([]string)
	if !ok || len(required) != 1 || required[0] != testSiteField {
		t.Fatalf("brokered tool required slice was not copied: %#v", agent.BrokeredTools[0].Parameters["required"])
	}
}
