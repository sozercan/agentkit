// Package agent converts an effective Agent into a BuildKit LLB graph plus
// an OCI image config. v0 uses the runtime adapter image as the LLB base and
// merges exactly one layer on top: the resolved /agent/agent.yaml. It does NOT
// rootfs-copy any tool image (plan §5.2 ⚠): v0 tools are stdio commands recorded
// in agent.yaml and spawned by agentkit-serve at runtime.
package agent

import (
	"os"

	"github.com/moby/buildkit/client/llb"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/sozercan/agentkit/pkg/agentkit/abi"
	"github.com/sozercan/agentkit/pkg/agentkit/effective"
	"github.com/sozercan/agentkit/pkg/utils"
)

// Agentkit2LLB converts an effective Agent to an LLB state and image config. The
// adapter image (agentkit-serve) is the base; the only delta merged on top is
// the baked agent.yaml.
func Agentkit2LLB(agent effective.Agent, adapterRef string, platform *specs.Platform) (llb.State, *specs.Image, error) {
	if platform == nil {
		return llb.State{}, nil, os.ErrInvalid
	}

	agentYAML, err := abi.Render(agent)
	if err != nil {
		return llb.State{}, nil, err
	}

	base := llb.Image(adapterRef, llb.Platform(*platform))

	writeAgentConfig := utils.Phase{
		Name: "Writing /agent/agent.yaml",
		Run: func(s llb.State) (llb.State, error) {
			return s.File(
				llb.Mkdir("/agent", 0o755, llb.WithParents(true)).
					Mkfile(abi.Path, 0o644, agentYAML),
			), nil
		},
	}

	merge, _, err := utils.ApplyPhases(base, base, writeAgentConfig)
	if err != nil {
		return llb.State{}, nil, err
	}

	return merge, NewImageConfig(agent, platform), nil
}
