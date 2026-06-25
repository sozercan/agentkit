package agent

import (
	"fmt"
	"testing"

	specs "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/sozercan/agentkit/pkg/agentkit/abi"
	"github.com/sozercan/agentkit/pkg/agentkit/config"
	"github.com/sozercan/agentkit/pkg/agentkit/effective"
	"github.com/sozercan/agentkit/pkg/agentkit/runtimes"
	"github.com/sozercan/agentkit/pkg/utils"
)

const (
	testTeamLabel = "com.example/team"
	testTeamValue = "agentkit"
)

func imageAgent(runtime string, port int) effective.Agent {
	cfg := &config.AgentConfig{
		Metadata: config.Metadata{
			Name:   "acme-support",
			Labels: map[string]string{testTeamLabel: testTeamValue},
		},
		Runtime: runtime,
		Model: config.Model{
			Provider: utils.ProviderOpenAICompatible,
			BaseURL:  "https://api.openai.com/v1",
			Name:     "gpt-4o-mini",
		},
		Instructions: config.Source{Inline: "ignored after resolution"},
		Expose:       config.Expose{OpenAI: true, Port: port},
	}
	return effective.FromConfig(cfg, "resolved prompt")
}

func TestNewImageConfigUsesEffectiveAgentContract(t *testing.T) {
	agent := imageAgent(runtimes.MAFAlias, 0)
	img := NewImageConfig(agent, &specs.Platform{Architecture: utils.PlatformAMD64})

	if img.Config.User != "1000:1000" {
		t.Fatalf("User = %q, want non-root 1000:1000", img.Config.User)
	}
	if got := fmt.Sprint(img.Config.Cmd); got != fmt.Sprint([]string{"--config", abi.Path}) {
		t.Fatalf("Cmd = %v, want --config %s", img.Config.Cmd, abi.Path)
	}
	if _, ok := img.Config.ExposedPorts[fmt.Sprintf("%d/tcp", utils.DefaultPort)]; !ok {
		t.Fatalf("ExposedPorts = %#v, want default port %d", img.Config.ExposedPorts, utils.DefaultPort)
	}
	if got := img.Config.Labels[utils.LabelPrefix+".runtime"]; got != runtimes.MAF {
		t.Fatalf("runtime label = %q, want canonical %q", got, runtimes.MAF)
	}
	if got := img.Config.Labels[utils.LabelPrefix+".abi"]; got != abi.Version {
		t.Fatalf("abi label = %q, want %q", got, abi.Version)
	}
	if got := img.Config.Labels[testTeamLabel]; got != testTeamValue {
		t.Fatalf("custom label = %q, want %s", got, testTeamValue)
	}
}

func TestNewImageConfigPreservesEffectivePort(t *testing.T) {
	agent := imageAgent("", 9090)
	img := NewImageConfig(agent, &specs.Platform{Architecture: utils.PlatformAMD64})

	if _, ok := img.Config.ExposedPorts["9090/tcp"]; !ok {
		t.Fatalf("ExposedPorts = %#v, want explicit 9090/tcp", img.Config.ExposedPorts)
	}
}
