package build

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestRuntimeAdapterBuildTargetsHonorPlatform(t *testing.T) {
	tests := []struct {
		target     string
		dockerfile string
		image      string
	}{
		{target: "build-serve", dockerfile: "runtimes/pydantic-ai/Dockerfile", image: "agentkit-serve:platform-test"},
		{target: "build-serve-maf", dockerfile: "runtimes/microsoft-agent-framework/Dockerfile", image: "agentkit-serve-maf:platform-test"},
		{target: "build-serve-langgraph", dockerfile: "runtimes/langgraph/Dockerfile", image: "agentkit-serve-langgraph:platform-test"},
	}

	for _, tt := range tests {
		t.Run(tt.target, func(t *testing.T) {
			cmd := makeAdapterDryRunCommand(tt.target)
			cmd.Dir = filepath.Join("..", "..")
			out, err := cmd.CombinedOutput()
			if err != nil {
				t.Fatalf("make dry run failed: %v\n%s", err, out)
			}
			command := string(out)
			for _, want := range []string{
				"docker buildx build",
				"-f " + tt.dockerfile,
				"-t " + tt.image,
				"--platform linux/arm64",
				"--load",
			} {
				if !strings.Contains(command, want) {
					t.Fatalf("%s command = %q, want substring %q", tt.target, command, want)
				}
			}
		})
	}
}

func makeAdapterDryRunCommand(target string) *exec.Cmd {
	switch target {
	case "build-serve":
		return exec.Command("make", "--no-print-directory", "-n", "build-serve", "PLATFORM=linux/arm64", "TAG=platform-test")
	case "build-serve-maf":
		return exec.Command("make", "--no-print-directory", "-n", "build-serve-maf", "PLATFORM=linux/arm64", "TAG=platform-test")
	case "build-serve-langgraph":
		return exec.Command("make", "--no-print-directory", "-n", "build-serve-langgraph", "PLATFORM=linux/arm64", "TAG=platform-test")
	default:
		panic("unsupported adapter build target: " + target)
	}
}

func TestRunTestAgentIsHostReachableAndCapturesCurlToken(t *testing.T) {
	repoRoot, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatalf("resolve repository root: %v", err)
	}
	tempDir := t.TempDir()
	binDir := filepath.Join(tempDir, "bin")
	if err := os.Mkdir(binDir, 0o755); err != nil {
		t.Fatalf("create fake bin directory: %v", err)
	}
	commandLog := filepath.Join(tempDir, "commands.log")

	writeCommandStub(t, binDir, "docker", `
{
  printf 'docker'
  for arg in "$@"; do printf '\t%s' "$arg"; done
  printf '\n'
} >>"${COMMAND_LOG}"
`)

	const modelKey = "model-key-must-not-appear"
	const localToken = "command-capture-token"
	cmd := exec.Command(
		"make",
		"--no-print-directory",
		"run-test-agent",
		"PLATFORM=linux/arm64",
		"TAG=command-capture",
		"LOCAL_AUTH_TOKEN="+localToken,
	)
	cmd.Dir = repoRoot
	cmd.Env = replaceEnvironment(os.Environ(), map[string]string{
		"COMMAND_LOG":    commandLog,
		"MAKEFLAGS":      "",
		"OPENAI_API_KEY": modelKey,
		"PATH":           binDir + string(os.PathListSeparator) + os.Getenv("PATH"),
	})
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("run-test-agent command capture failed: %v\n%s", err, out)
	}
	logBytes, err := os.ReadFile(commandLog)
	if err != nil {
		t.Fatalf("read command log: %v", err)
	}
	command := string(logBytes)
	for _, want := range []string{
		"\t-p\t127.0.0.1:8080:8080\t",
		"\t-e\tAGENTKIT_BIND=0.0.0.0\t",
		"\t-e\tAGENTKIT_AUTH_TOKEN=" + localToken + "\t",
		"\t-e\tOPENAI_API_KEY\t",
		"\thello-agent:command-capture\n",
	} {
		if !strings.Contains(command, want) {
			t.Fatalf("captured docker command = %q, want substring %q", command, want)
		}
	}
	combined := string(out) + command
	if strings.Contains(combined, modelKey) {
		t.Fatalf("run-test-agent output leaked model key: %q", combined)
	}
	for _, want := range []string{
		"Authorization: Bearer " + localToken,
		"http://127.0.0.1:8080/v1/models",
	} {
		if !strings.Contains(string(out), want) {
			t.Fatalf("run-test-agent output = %q, want substring %q", out, want)
		}
	}
}

func TestLiveCopilotScriptForwardsDetectedPlatformToAdapterBuild(t *testing.T) {
	repoRoot, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatalf("resolve repository root: %v", err)
	}
	tempDir := t.TempDir()
	binDir := filepath.Join(tempDir, "bin")
	if err := os.Mkdir(binDir, 0o755); err != nil {
		t.Fatalf("create fake bin directory: %v", err)
	}
	cacheDir := filepath.Join(tempDir, "vekil-cache")
	if err := os.Mkdir(cacheDir, 0o755); err != nil {
		t.Fatalf("create fake Vekil cache: %v", err)
	}
	commandLog := filepath.Join(tempDir, "commands.log")

	writeCommandStub(t, binDir, "docker", `
{
  printf 'docker'
  for arg in "$@"; do printf '\t%s' "$arg"; done
  printf '\n'
} >>"${COMMAND_LOG}"
if [ "${1:-}" = info ]; then
  printf 'arm64\n'
fi
`)
	writeCommandStub(t, binDir, "make", `
{
  printf 'make'
  for arg in "$@"; do printf '\t%s' "$arg"; done
  printf '\n'
} >>"${COMMAND_LOG}"
`)
	writeCommandStub(t, binDir, "curl", `
case "$*" in
  */v1/models*)
    printf '{"data":[{"id":"claude-haiku-4.5"}]}'
    ;;
  */v1/chat/completions*)
    printf '{"model":"claude-haiku-4.5","choices":[{"message":{"content":"DONE42"}}]}'
    ;;
esac
`)
	writeCommandStub(t, binDir, "jq", `
case "$*" in
  *'.data[].id'*) printf 'claude-haiku-4.5\n' ;;
  *'{model,'*) printf '{"model":"claude-haiku-4.5","content":"DONE42"}\n' ;;
esac
`)
	writeCommandStub(t, binDir, "go", "")

	cmd := exec.Command("bash", "scripts/live-copilot-agent-e2e.sh")
	cmd.Dir = repoRoot
	cmd.Env = replaceEnvironment(os.Environ(), map[string]string{
		"COMMAND_LOG":          commandLog,
		"COPILOT_GITHUB_TOKEN": "",
		"PATH":                 binDir + string(os.PathListSeparator) + os.Getenv("PATH"),
		"PLATFORM":             "",
		"RUNNER_TEMP":          tempDir,
		"TAG":                  "command-capture",
		"VEKIL_CACHE_DIR":      cacheDir,
	})
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("live script command capture failed: %v\n%s", err, out)
	}
	logBytes, err := os.ReadFile(commandLog)
	if err != nil {
		t.Fatalf("read command log: %v", err)
	}
	want := "make\tbuild-serve-maf\tTAG=command-capture\tPLATFORM=linux/arm64\n"
	if !strings.Contains(string(logBytes), want) {
		t.Fatalf("captured commands = %q, want %q", logBytes, want)
	}
}

func writeCommandStub(t *testing.T, dir, name, body string) {
	t.Helper()
	path := filepath.Join(dir, name)
	contents := "#!/bin/sh\nset -eu\n" + body
	if err := os.WriteFile(path, []byte(contents), 0o600); err != nil {
		t.Fatalf("write %s stub: %v", name, err)
	}
	if err := os.Chmod(path, 0o700); err != nil {
		t.Fatalf("make %s stub executable: %v", name, err)
	}
}

func replaceEnvironment(base []string, replacements map[string]string) []string {
	out := make([]string, 0, len(base)+len(replacements))
	for _, entry := range base {
		key, _, ok := strings.Cut(entry, "=")
		if ok {
			if _, replace := replacements[key]; replace {
				continue
			}
		}
		out = append(out, entry)
	}
	for key, value := range replacements {
		out = append(out, key+"="+value)
	}
	return out
}
