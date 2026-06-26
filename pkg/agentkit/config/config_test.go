package config

import (
	"strings"
	"testing"

	"github.com/sozercan/agentkit/pkg/agentkit/runtimes"
)

// TestKindProbeRejectsKindlessFile is the regression guard for the AIKit
// silent-misparse bug (plan §5.1/§16.1): a file without `kind: Agent` must be a
// loud load-time error, never a silently-empty config.
func TestKindProbeRejectsKindlessFile(t *testing.T) {
	// An AIKit-style inference file: no kind, has backends/models.
	in := []byte("apiVersion: v1alpha1\nbackends:\n  - llama-cpp\n")
	_, err := NewFromBytes(in)
	if err == nil {
		t.Fatal("expected error for kind-less file, got nil (the silent-misparse bug)")
	}
	if !strings.Contains(err.Error(), "kind") {
		t.Fatalf("expected a kind-related error, got: %v", err)
	}
}

func TestKindProbeRejectsWrongKind(t *testing.T) {
	in := []byte("apiVersion: v1alpha1\nkind: Crew\nmetadata:\n  name: x\n")
	_, err := NewFromBytes(in)
	if err == nil || !strings.Contains(err.Error(), "Crew") {
		t.Fatalf("expected unsupported-kind error mentioning Crew, got: %v", err)
	}
}

func TestStrictParseRejectsUnknownField(t *testing.T) {
	// `runtimes:` (typo of `runtime:`) must be a load-time error under strict mode.
	in := []byte("apiVersion: v1alpha1\nkind: Agent\nmetadata:\n  name: x\nruntimes: pydantic-ai\n")
	_, err := NewFromBytes(in)
	if err == nil {
		t.Fatal("expected strict-mode error for unknown field `runtimes`, got nil")
	}
}

func TestInstructionsAcceptsBareString(t *testing.T) {
	in := []byte(`apiVersion: v1alpha1
kind: Agent
metadata:
  name: hello
model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
  name: gpt-4o-mini
  apiKeyEnv: OPENAI_API_KEY
instructions: |
  Be helpful.
expose:
  openai: true
`)
	cfg, err := NewFromBytes(in)
	if err != nil {
		t.Fatalf("unexpected parse error: %v", err)
	}
	if got := strings.TrimSpace(cfg.Instructions.Inline); got != "Be helpful." {
		t.Fatalf("instructions inline = %q, want %q", got, "Be helpful.")
	}
	if cfg.Instructions.File != "" {
		t.Fatalf("expected empty File, got %q", cfg.Instructions.File)
	}
	if err := cfg.Validate(); err != nil {
		t.Fatalf("valid four-keys config failed validation: %v", err)
	}
}

func TestInstructionsAcceptsFileMapping(t *testing.T) {
	in := []byte(`apiVersion: v1alpha1
kind: Agent
metadata:
  name: hello
model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
  name: gpt-4o-mini
  apiKeyEnv: OPENAI_API_KEY
instructions:
  file: ./prompt.md
expose:
  openai: true
`)
	cfg, err := NewFromBytes(in)
	if err != nil {
		t.Fatalf("unexpected parse error: %v", err)
	}
	if cfg.Instructions.File != "./prompt.md" {
		t.Fatalf("instructions file = %q, want ./prompt.md", cfg.Instructions.File)
	}
}

func TestValidateReportsMultipleErrors(t *testing.T) {
	// Missing model.name, instructions, and expose.openai=false → at least 3 errors,
	// all reported at once (errors.Join, plan §16.2 #3).
	in := []byte(`apiVersion: v1alpha1
kind: Agent
metadata:
  name: broken
model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
expose:
  openai: false
`)
	cfg, err := NewFromBytes(in)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	verr := cfg.Validate()
	if verr == nil {
		t.Fatal("expected validation errors, got nil")
	}
	msg := verr.Error()
	for _, want := range []string{"model.name", "instructions", "expose.openai"} {
		if !strings.Contains(msg, want) {
			t.Errorf("validation error missing %q; full: %s", want, msg)
		}
	}
}

func TestValidateRejectsSecretLiteralInApiKeyEnv(t *testing.T) {
	in := []byte(`apiVersion: v1alpha1
kind: Agent
metadata:
  name: leaky
model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
  name: gpt-4o-mini
  apiKeyEnv: sk-secret-value-here
instructions: hi
expose:
  openai: true
`)
	cfg, err := NewFromBytes(in)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	if verr := cfg.Validate(); verr == nil || !strings.Contains(verr.Error(), "apiKeyEnv") {
		t.Fatalf("expected secret-literal rejection on apiKeyEnv, got: %v", verr)
	}
}

func TestValidateRejectsEmptyToolCommandEntry(t *testing.T) {
	in := []byte(`apiVersion: v1alpha1
kind: Agent
metadata:
  name: emptycmd
model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
  name: gpt-4o-mini
  apiKeyEnv: OPENAI_API_KEY
instructions: hi
tools:
  - name: fetch
    command: [""]
expose:
  openai: true
`)
	cfg, err := NewFromBytes(in)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	if verr := cfg.Validate(); verr == nil || !strings.Contains(verr.Error(), "command[0]") {
		t.Fatalf("expected empty command entry rejection, got: %v", verr)
	}
}

func TestValidateRejectsEmptyToolEnvName(t *testing.T) {
	in := []byte(`apiVersion: v1alpha1
kind: Agent
metadata:
  name: emptyenv
model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
  name: gpt-4o-mini
  apiKeyEnv: OPENAI_API_KEY
instructions: hi
tools:
  - name: fetch
    command: ["uvx", "mcp-server-fetch"]
    env: [""]
expose:
  openai: true
`)
	cfg, err := NewFromBytes(in)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	if verr := cfg.Validate(); verr == nil || !strings.Contains(verr.Error(), "env entry is empty") {
		t.Fatalf("expected empty env name rejection, got: %v", verr)
	}
}

func TestValidateRejectsImageTool(t *testing.T) {
	in := []byte(`apiVersion: v1alpha1
kind: Agent
metadata:
  name: imgtool
model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
  name: gpt-4o-mini
  apiKeyEnv: OPENAI_API_KEY
instructions: hi
tools:
  - name: fetch
    image: ghcr.io/example/server-fetch:latest
expose:
  openai: true
`)
	cfg, err := NewFromBytes(in)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	if verr := cfg.Validate(); verr == nil || !strings.Contains(verr.Error(), "image-based") {
		t.Fatalf("expected v0 rejection of image-based tool, got: %v", verr)
	}
}

// agentBaseYAML returns a minimal valid four-keys agentkitfile with the given
// `runtime:` line spliced in (empty string omits it).
func agentBaseYAML(runtimeLine string) []byte {
	rt := ""
	if runtimeLine != "" {
		rt = "runtime: " + runtimeLine + "\n"
	}
	return []byte(`apiVersion: v1alpha1
kind: Agent
metadata:
  name: rt-agent
` + rt + `model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
  name: gpt-4o-mini
  apiKeyEnv: OPENAI_API_KEY
instructions: hi
expose:
  openai: true
`)
}

// TestValidateAcceptsRegisteredRuntimes proves the widened runtime gate (plan §8):
// the canonical MAF name, its "maf" alias, the default runtime, and an omitted
// runtime all validate.
func TestValidateAcceptsRegisteredRuntimes(t *testing.T) {
	for _, rt := range []string{"", "pydantic-ai", "microsoft-agent-framework", "maf"} {
		cfg, err := NewFromBytes(agentBaseYAML(rt))
		if err != nil {
			t.Fatalf("runtime %q: parse error: %v", rt, err)
		}
		if verr := cfg.Validate(); verr != nil {
			t.Errorf("runtime %q: expected valid, got: %v", rt, verr)
		}
	}
}

// TestValidateRejectsUnknownRuntime keeps the deterministic gate: an unregistered
// runtime is a clear error that lists the supported set.
func TestValidateRejectsUnknownRuntime(t *testing.T) {
	cfg, err := NewFromBytes(agentBaseYAML("langchain"))
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	verr := cfg.Validate()
	if verr == nil || !strings.Contains(verr.Error(), "runtime") {
		t.Fatalf("expected unknown-runtime rejection, got: %v", verr)
	}
	if !strings.Contains(verr.Error(), "microsoft-agent-framework") {
		t.Errorf("error should list supported runtimes; got: %v", verr)
	}
}

func TestEnvDeclarationsValidateAndParse(t *testing.T) {
	in := []byte(`apiVersion: v1alpha1
kind: Agent
metadata:
  name: env-agent
model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
  name: gpt-4o-mini
  apiKeyEnv: OPENAI_API_KEY
instructions: hi
env:
  - name: REQUIRED_FOO
    required: true
  - name: OPTIONAL_BAR
expose:
  openai: true
`)
	cfg, err := NewFromBytes(in)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	if len(cfg.Env) != 2 {
		t.Fatalf("Env len = %d, want 2", len(cfg.Env))
	}
	if cfg.Env[0].Name != "REQUIRED_FOO" || !cfg.Env[0].Required {
		t.Fatalf("first env = %+v, want required REQUIRED_FOO", cfg.Env[0])
	}
	if cfg.Env[1].Name != "OPTIONAL_BAR" || cfg.Env[1].Required {
		t.Fatalf("second env = %+v, want optional OPTIONAL_BAR", cfg.Env[1])
	}
	if verr := cfg.Validate(); verr != nil {
		t.Fatalf("valid env declarations failed validation: %v", verr)
	}
}

func TestEnvDeclarationsRejectInvalidNamesDuplicatesAndValues(t *testing.T) {
	in := []byte(`apiVersion: v1alpha1
kind: Agent
metadata:
  name: env-agent
model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
  name: gpt-4o-mini
  apiKeyEnv: OPENAI_API_KEY
instructions: hi
env:
  - name: required-foo
  - name: REQUIRED_FOO
  - name: REQUIRED_FOO
expose:
  openai: true
`)
	cfg, err := NewFromBytes(in)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	verr := cfg.Validate()
	if verr == nil {
		t.Fatal("expected env validation errors, got nil")
	}
	msg := verr.Error()
	for _, want := range []string{"env[0].name", "[A-Z0-9_]+", "duplicate env var name"} {
		if !strings.Contains(msg, want) {
			t.Errorf("validation error missing %q; full: %s", want, msg)
		}
	}

	withValue := []byte(`apiVersion: v1alpha1
kind: Agent
metadata:
  name: env-agent
model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
  name: gpt-4o-mini
instructions: hi
env:
  - name: REQUIRED_FOO
    value: do-not-allow-literals
expose:
  openai: true
`)
	if _, err := NewFromBytes(withValue); err == nil || !strings.Contains(err.Error(), "value") {
		t.Fatalf("expected strict parse rejection for env.value literal, got: %v", err)
	}
}

func TestValidateRejectsStdioToolWhenRuntimeLacksCapability(t *testing.T) {
	old := append([]runtimes.RuntimeSpec(nil), runtimes.Runtimes...)
	t.Cleanup(func() { runtimes.Runtimes = old })
	for i := range runtimes.Runtimes {
		if runtimes.Runtimes[i].Name == runtimes.PydanticAI {
			runtimes.Runtimes[i].Capabilities = nil
		}
	}

	in := []byte(`apiVersion: v1alpha1
kind: Agent
metadata:
  name: caps-agent
runtime: pydantic-ai
model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
  name: gpt-4o-mini
instructions: hi
tools:
  - name: fetch
    command: ["uvx", "mcp-server-fetch"]
expose:
  openai: true
`)
	cfg, err := NewFromBytes(in)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	verr := cfg.Validate()
	if verr == nil {
		t.Fatal("expected runtime capability validation error, got nil")
	}
	msg := verr.Error()
	for _, want := range []string{"runtime \"pydantic-ai\"", "stdio-mcp"} {
		if !strings.Contains(msg, want) {
			t.Errorf("validation error missing %q; full: %s", want, msg)
		}
	}
}
