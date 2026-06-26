// Package abi renders the frozen agent.yaml writer contract shared by the Go
// frontend and Python runtime readers.
package abi

import (
	"github.com/goccy/go-yaml"
	"github.com/sozercan/agentkit/pkg/agentkit/effective"
)

// Version is the schema version of the baked agent.yaml (docs/agent-abi.md).
// It MUST equal agentkit-serve's config.ABI_VERSION ("v0").
const Version = "v0"

// Path is where the rendered agent.yaml is baked into the agent image.
const Path = "/agent/agent.yaml"

// The following types are the WRITER half of the frozen agent.yaml ABI
// (docs/agent-abi.md). agentkit-serve reads this file with pydantic
// extra="forbid", so these structs MUST emit EXACTLY the keys documented there.
// Field order here is the emit order.

type abiMetadata struct {
	Name string `yaml:"name"`
}

type abiModel struct {
	Provider  string `yaml:"provider"`
	BaseURL   string `yaml:"baseURL"`
	Name      string `yaml:"name"`
	APIKeyEnv string `yaml:"apiKeyEnv,omitempty"`
}

type abiToolHeader struct {
	Name     string `yaml:"name"`
	Value    string `yaml:"value,omitempty"`
	ValueEnv string `yaml:"valueEnv,omitempty"`
}

type abiAuth struct {
	Type     string `yaml:"type"`
	TokenEnv string `yaml:"tokenEnv,omitempty"`
	Audience string `yaml:"audience,omitempty"`
}

type abiTool struct {
	Name      string          `yaml:"name"`
	Type      string          `yaml:"type,omitempty"`
	Transport string          `yaml:"transport,omitempty"`
	Command   []string        `yaml:"command,omitempty"`
	URLEnv    string          `yaml:"urlEnv,omitempty"`
	Headers   []abiToolHeader `yaml:"headers,omitempty"`
	Auth      *abiAuth        `yaml:"auth,omitempty"`
	Approval  string          `yaml:"approval,omitempty"`
	Env       []string        `yaml:"env,omitempty"`
}

type abiEnvVar struct {
	Name     string `yaml:"name"`
	Required bool   `yaml:"required,omitempty"`
}

type abiContextProvider struct {
	Name         string   `yaml:"name,omitempty"`
	Type         string   `yaml:"type"`
	Source       string   `yaml:"source,omitempty"`
	Path         string   `yaml:"path,omitempty"`
	ToolRef      string   `yaml:"toolRef,omitempty"`
	Index        string   `yaml:"index,omitempty"`
	EndpointEnv  string   `yaml:"endpointEnv,omitempty"`
	IndexEnv     string   `yaml:"indexEnv,omitempty"`
	StoreNameEnv string   `yaml:"storeNameEnv,omitempty"`
	Auth         *abiAuth `yaml:"auth,omitempty"`
}

type abiContext struct {
	Providers []abiContextProvider `yaml:"providers,omitempty"`
}

type abiObservability struct {
	OTel struct {
		EndpointEnv string `yaml:"endpointEnv,omitempty"`
	} `yaml:"otel,omitempty"`
	Logs struct {
		LevelEnv string `yaml:"levelEnv,omitempty"`
	} `yaml:"logs,omitempty"`
}

type abiExpose struct {
	OpenAI bool `yaml:"openai"`
	Port   int  `yaml:"port"`
}

type abiAgent struct {
	ABIVersion    string            `yaml:"abiVersion"`
	Metadata      abiMetadata       `yaml:"metadata"`
	Model         abiModel          `yaml:"model"`
	Instructions  string            `yaml:"instructions"`
	Tools         []abiTool         `yaml:"tools"`
	Env           []abiEnvVar       `yaml:"env,omitempty"`
	Context       *abiContext       `yaml:"context,omitempty"`
	Observability *abiObservability `yaml:"observability,omitempty"`
	Expose        abiExpose         `yaml:"expose"`
}

// Render produces the baked /agent/agent.yaml from an effective Agent. The
// output is byte-compatible with agentkit-serve's strict (extra=forbid) reader.
func Render(agent effective.Agent) ([]byte, error) {
	out := abiAgent{
		ABIVersion: Version,
		Metadata:   abiMetadata{Name: agent.Metadata.Name},
		Model: abiModel{
			Provider:  agent.Model.Provider,
			BaseURL:   agent.Model.BaseURL,
			Name:      agent.Model.Name,
			APIKeyEnv: agent.Model.APIKeyEnv,
		},
		Instructions: agent.Instructions,
		Tools:        make([]abiTool, 0, len(agent.Tools)),
		Env:          make([]abiEnvVar, 0, len(agent.Env)),
		Expose:       abiExpose{OpenAI: agent.Expose.OpenAI, Port: agent.Expose.Port},
	}
	for _, t := range agent.Tools {
		tool := abiTool{
			Name:      t.Name,
			Type:      t.Type,
			Transport: t.Transport,
			Command:   t.Command,
			URLEnv:    t.URLEnv,
			Headers:   make([]abiToolHeader, 0, len(t.Headers)),
			Approval:  t.Approval,
			Env:       t.Env,
		}
		for _, h := range t.Headers {
			tool.Headers = append(tool.Headers, abiToolHeader{Name: h.Name, Value: h.Value, ValueEnv: h.ValueEnv})
		}
		if len(tool.Headers) == 0 {
			tool.Headers = nil
		}
		if t.Auth != nil {
			tool.Auth = &abiAuth{Type: t.Auth.Type, TokenEnv: t.Auth.TokenEnv, Audience: t.Auth.Audience}
		}
		out.Tools = append(out.Tools, tool)
	}
	for _, e := range agent.Env {
		out.Env = append(out.Env, abiEnvVar{Name: e.Name, Required: e.Required})
	}
	if len(out.Env) == 0 {
		out.Env = nil
	}
	if len(agent.Context.Providers) > 0 {
		ctx := &abiContext{Providers: make([]abiContextProvider, 0, len(agent.Context.Providers))}
		for _, provider := range agent.Context.Providers {
			p := abiContextProvider{
				Name:         provider.Name,
				Type:         provider.Type,
				Source:       provider.Source,
				Path:         provider.Path,
				ToolRef:      provider.ToolRef,
				Index:        provider.Index,
				EndpointEnv:  provider.EndpointEnv,
				IndexEnv:     provider.IndexEnv,
				StoreNameEnv: provider.StoreNameEnv,
			}
			if provider.Auth != nil {
				p.Auth = &abiAuth{Type: provider.Auth.Type, TokenEnv: provider.Auth.TokenEnv, Audience: provider.Auth.Audience}
			}
			ctx.Providers = append(ctx.Providers, p)
		}
		out.Context = ctx
	}
	if agent.Observability.OTel.EndpointEnv != "" || agent.Observability.Logs.LevelEnv != "" {
		obs := &abiObservability{}
		obs.OTel.EndpointEnv = agent.Observability.OTel.EndpointEnv
		obs.Logs.LevelEnv = agent.Observability.Logs.LevelEnv
		out.Observability = obs
	}

	return yaml.Marshal(out)
}
