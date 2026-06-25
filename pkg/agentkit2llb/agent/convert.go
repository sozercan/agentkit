// Package agent converts an effective Agent into a BuildKit LLB graph plus
// an OCI image config. v0 uses the runtime adapter image as the LLB base and
// writes the resolved /agent/agent.yaml on top. It does NOT
// rootfs-copy any tool image (plan §5.2 ⚠): v0 tools are stdio commands recorded
// in agent.yaml and spawned by agentkit-serve at runtime.
package agent

import (
	"os"

	"github.com/moby/buildkit/client/llb"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/sozercan/agentkit/pkg/agentkit/abi"
	"github.com/sozercan/agentkit/pkg/agentkit/effective"
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

	state := llb.Image(adapterRef, llb.Platform(*platform)).File(
		llb.Mkdir("/agent", 0o755, llb.WithParents(true)).
			Mkfile(abi.Path, 0o644, agentYAML),
	)

	return state, NewImageConfig(agent, platform), nil
}
