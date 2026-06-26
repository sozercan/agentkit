// Package effective turns a validated authored AgentConfig plus resolved build
// inputs into the build-ready Agent value consumed by ABI and image writers.
package effective

import (
	"github.com/sozercan/agentkit/pkg/agentkit/config"
	"github.com/sozercan/agentkit/pkg/agentkit/runtimes"
	"github.com/sozercan/agentkit/pkg/utils"
)

// Agent is the effective, build-ready agent description.
//
// It is derived after config validation and instruction resolution, so callers no
// longer need to remember defaulting rules for runtime names or serve ports, nor
// whether instructions are authored as inline/file sources.
type Agent struct {
	Metadata     config.Metadata
	Runtime      string
	Model        config.Model
	Instructions string
	Tools        []config.Tool
	Env          []config.EnvVar
	Expose       config.Expose
}

// FromConfig returns the effective Agent for a validated authored config and a
// fully-resolved instruction string.
func FromConfig(cfg *config.AgentConfig, instructions string) Agent {
	runtime := cfg.Runtime
	if runtime == "" {
		runtime = runtimes.DefaultRuntime()
	} else {
		runtime = runtimes.CanonicalRuntime(runtime)
	}

	expose := cfg.Expose
	if expose.Port == 0 {
		expose.Port = utils.DefaultPort
	}

	return Agent{
		Metadata: config.Metadata{
			Name:   cfg.Metadata.Name,
			Labels: copyLabels(cfg.Metadata.Labels),
		},
		Runtime:      runtime,
		Model:        cfg.Model,
		Instructions: instructions,
		Tools:        copyTools(cfg.Tools),
		Env:          copyEnvVars(cfg.Env),
		Expose:       expose,
	}
}

func copyLabels(in map[string]string) map[string]string {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]string, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func copyTools(in []config.Tool) []config.Tool {
	if len(in) == 0 {
		return nil
	}
	out := make([]config.Tool, len(in))
	for i, tool := range in {
		out[i] = tool
		out[i].Command = append([]string(nil), tool.Command...)
		out[i].Env = append([]string(nil), tool.Env...)
	}
	return out
}

func copyEnvVars(in []config.EnvVar) []config.EnvVar {
	if len(in) == 0 {
		return nil
	}
	out := make([]config.EnvVar, len(in))
	copy(out, in)
	return out
}
