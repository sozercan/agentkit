package render

import (
	"bytes"
	"strings"
	"testing"
)

const testRuntimeName = "fibey"

func TestOrkaAgentRuntimeExternalEndpoint(t *testing.T) {
	got, err := OrkaAgentRuntime(OrkaAgentRuntimeOptions{
		Name:             "fibey-agentkit",
		ExternalEndpoint: "http://fibey-agentkit.default.svc.cluster.local:8080",
	})
	if err != nil {
		t.Fatalf("OrkaAgentRuntime() error = %v", err)
	}
	want := `apiVersion: core.orka.ai/v1alpha1
kind: AgentRuntime
metadata:
  name: fibey-agentkit
spec:
  contractVersion: orka.harness.v1
  deployment:
    mode: external-endpoint
    endpoint: http://fibey-agentkit.default.svc.cluster.local:8080
  clientAuth:
    bearerTokenSecretRef:
      name: fibey-agentkit-harness-token
      key: token
  capabilities:
    toolExecutionModes:
    - observed
    supportsCancel: true
    supportsRuntimeSessions: true
`
	if string(got) != want {
		t.Fatalf("manifest mismatch\n got:\n%s\nwant:\n%s", got, want)
	}
}

func TestRunCLIRendersOrkaAgentRuntime(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := RunCLI([]string{
		"--target", TargetOrkaAgentRuntime,
		"--external-endpoint", "http://fibey-agentkit.default.svc.cluster.local:8080",
		"--name", "fibey-agentkit",
		"--auth-secret-name", "fibey-auth",
	}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("RunCLI() code = %d stderr=%s", code, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "kind: AgentRuntime") || !strings.Contains(out, "name: fibey-auth") {
		t.Fatalf("stdout = %s", out)
	}
}

func TestOrkaAgentRuntimeValidation(t *testing.T) {
	if _, err := OrkaAgentRuntime(OrkaAgentRuntimeOptions{Name: testRuntimeName}); err == nil {
		t.Fatal("expected missing external endpoint error")
	}
	if _, err := OrkaAgentRuntime(OrkaAgentRuntimeOptions{Name: testRuntimeName, Image: "ghcr.io/acme/fibey:latest"}); err == nil || !strings.Contains(err.Error(), "external endpoints only") {
		t.Fatalf("expected image unsupported error, got %v", err)
	}
	var stdout, stderr bytes.Buffer
	code := RunCLI([]string{"--target", "other", "--name", testRuntimeName, "--external-endpoint", "http://example.invalid"}, &stdout, &stderr)
	if code == 0 || !strings.Contains(stderr.String(), "unsupported --target") {
		t.Fatalf("RunCLI() code=%d stderr=%q", code, stderr.String())
	}
}
