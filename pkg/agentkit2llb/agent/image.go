package agent

import (
	"fmt"

	"github.com/moby/buildkit/util/system"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/sozercan/agentkit/pkg/agentkit/config"
	"github.com/sozercan/agentkit/pkg/utils"
)

// NewImageConfig builds the OCI image config for the agent image. It deliberately
// does NOT inherit AIKit's root user: per plan §10 the agent runs non-root, binds
// loopback by default, and exposes the serve port.
func NewImageConfig(cfg *config.AgentConfig, platform *specs.Platform) *specs.Image {
	port := cfg.Expose.Port
	if port == 0 {
		port = utils.DefaultPort
	}

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
	img.Config.Cmd = []string{"--config", utils.AgentConfigPath}

	img.Config.Env = []string{
		"PATH=" + utils.AgentKitRoot + "/bin:" + system.DefaultPathEnv(utils.PlatformLinux),
		"AGENTKIT_BIND=127.0.0.1", // loopback default; 0.0.0.0 requires AGENTKIT_AUTH_TOKEN
		"PYTHONUNBUFFERED=1",
	}

	img.Config.ExposedPorts = map[string]struct{}{
		fmt.Sprintf("%d/tcp", port): {},
	}

	img.Config.Labels = map[string]string{
		utils.LabelPrefix + ".runtime":   runtimeLabel(cfg),
		utils.LabelPrefix + ".name":      cfg.Metadata.Name,
		utils.LabelPrefix + ".abi":       abiVersion,
		"org.opencontainers.image.title": cfg.Metadata.Name,
	}
	for k, v := range cfg.Metadata.Labels {
		img.Config.Labels[k] = v
	}

	return img
}

// runtimeLabel returns the canonical runtime name for the image label,
// defaulting to the v0 runtime when unset and resolving any alias (e.g. "maf")
// to its canonical form so the label is stable across spellings.
func runtimeLabel(cfg *config.AgentConfig) string {
	if cfg.Runtime != "" {
		return utils.CanonicalRuntime(cfg.Runtime)
	}
	return utils.RuntimePydanticAI
}
