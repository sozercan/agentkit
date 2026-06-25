package config

import (
	"strings"
	"testing"
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
// the canonical MAF name, its "maf" alias, LangGraph, the default runtime, and
// an omitted runtime all validate.
func TestValidateAcceptsRegisteredRuntimes(t *testing.T) {
	for _, rt := range []string{"", "pydantic-ai", "microsoft-agent-framework", "maf", "langgraph"} {
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
	for _, want := range []string{"microsoft-agent-framework", "langgraph"} {
		if !strings.Contains(verr.Error(), want) {
			t.Errorf("error should list supported runtime %q; got: %v", want, verr)
		}
	}
}
