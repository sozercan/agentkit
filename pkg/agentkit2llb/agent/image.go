package agent

import (
	"fmt"

	"github.com/moby/buildkit/util/system"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/sozercan/agentkit/pkg/agentkit/abi"
	"github.com/sozercan/agentkit/pkg/agentkit/effective"
	"github.com/sozercan/agentkit/pkg/utils"
)

// NewImageConfig builds the OCI image config for the agent image. It deliberately
// does NOT inherit AIKit's root user: per plan §10 the agent runs non-root, binds
// loopback by default, and exposes the serve port.
func NewImageConfig(agent effective.Agent, platform *specs.Platform) *specs.Image {
	img := &specs.Image{
		Platform: specs.Platform{
			Architecture: platform.Architecture,
			OS:           utils.PlatformLinux,
		},
	}
	img.RootFS.Type = "layers"

	img.Config.User = "1000:1000" // NON-ROOT (plan §10)
	img.Config.WorkingDir = "/"
	img.Config.Entrypoint = []string{utils.ServeBinary}
	img.Config.Cmd = []string{"--config", abi.Path}

	img.Config.Env = []string{
		"PATH=" + utils.AgentKitRoot + "/bin:" + system.DefaultPathEnv(utils.PlatformLinux),
		"AGENTKIT_BIND=127.0.0.1", // loopback default; 0.0.0.0 requires AGENTKIT_AUTH_TOKEN
		"PYTHONUNBUFFERED=1",
	}

	img.Config.ExposedPorts = map[string]struct{}{
		fmt.Sprintf("%d/tcp", agent.Expose.Port): {},
	}

	img.Config.Labels = map[string]string{
		utils.LabelPrefix + ".runtime":   agent.Runtime,
		utils.LabelPrefix + ".name":      agent.Metadata.Name,
		utils.LabelPrefix + ".abi":       abi.Version,
		"org.opencontainers.image.title": agent.Metadata.Name,
	}
	for k, v := range agent.Metadata.Labels {
		img.Config.Labels[k] = v
	}

	return img
}
