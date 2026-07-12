// Package abi renders the frozen agent.yaml writer contract shared by the Go
// frontend and Python runtime readers.
package abi

import (
	"encoding/json"
	"math"
	"strconv"
	"strings"

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

type yamlNumber string

func (n yamlNumber) MarshalYAML() ([]byte, error) {
	return []byte(n), nil
}

func yamlFloat(value float64, bitSize int) yamlNumber {
	rendered := strconv.FormatFloat(value, 'f', -1, bitSize)
	if value == 0 && math.Signbit(value) {
		rendered = "-0.0"
	}
	return yamlNumber(rendered)
}

type abiMetadata struct {
	Name string `yaml:"name"`
}

type abiModel struct {
	Provider  string   `yaml:"provider"`
	BaseURL   string   `yaml:"baseURL"`
	Name      string   `yaml:"name"`
	APIKeyEnv string   `yaml:"apiKeyEnv,omitempty"`
	Auth      *abiAuth `yaml:"auth,omitempty"`
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

type abiBrokeredTool struct {
	Name          string         `yaml:"name"`
	Description   string         `yaml:"description"`
	BrokeredClass string         `yaml:"brokeredClass"`
	Parameters    map[string]any `yaml:"parameters"`
	SchemaDigest  string         `yaml:"schemaDigest,omitempty"`
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
	BrokeredTools []abiBrokeredTool `yaml:"brokeredTools,omitempty"`
	Env           []abiEnvVar       `yaml:"env,omitempty"`
	Context       *abiContext       `yaml:"context,omitempty"`
	Observability *abiObservability `yaml:"observability,omitempty"`
	Expose        abiExpose         `yaml:"expose"`
}

func expandJSONNumber(value string) string {
	lower := strings.ToLower(value)
	parts := strings.Split(lower, "e")
	if len(parts) != 2 {
		return value
	}
	exponent, err := strconv.Atoi(parts[1])
	if err != nil {
		return value
	}
	mantissa := parts[0]
	sign := ""
	if strings.HasPrefix(mantissa, "-") || strings.HasPrefix(mantissa, "+") {
		if mantissa[0] == '-' {
			sign = "-"
		}
		mantissa = mantissa[1:]
	}
	fracLen := 0
	if dot := strings.IndexByte(mantissa, '.'); dot >= 0 {
		fracLen = len(mantissa) - dot - 1
		mantissa = mantissa[:dot] + mantissa[dot+1:]
	}
	mantissa = strings.TrimLeft(mantissa, "0")
	if mantissa == "" {
		return "0"
	}
	decimalPos := len(mantissa) - fracLen + exponent
	var out string
	switch {
	case decimalPos <= 0:
		out = "0." + strings.Repeat("0", -decimalPos) + mantissa
	case decimalPos >= len(mantissa):
		out = mantissa + strings.Repeat("0", decimalPos-len(mantissa))
	default:
		out = mantissa[:decimalPos] + "." + mantissa[decimalPos:]
	}
	if strings.Contains(out, ".") {
		out = strings.TrimRight(out, "0")
		out = strings.TrimRight(out, ".")
	}
	if out == "" || out == "-" {
		return "0"
	}
	return sign + out
}

func copyMap(in map[string]any) map[string]any {
	if in == nil {
		return nil
	}
	out := make(map[string]any, len(in))
	for k, v := range in {
		out[k] = copyAny(v)
	}
	return out
}

func copyAny(v any) any {
	switch typed := v.(type) {
	case map[string]any:
		return copyMap(typed)
	case map[string]string:
		out := make(map[string]string, len(typed))
		for key, value := range typed {
			out[key] = value
		}
		return out
	case map[string]int:
		out := make(map[string]int, len(typed))
		for key, value := range typed {
			out[key] = value
		}
		return out
	case map[string]float64:
		out := make(map[string]any, len(typed))
		for key, value := range typed {
			out[key] = yamlFloat(value, 64)
		}
		return out
	case map[string]bool:
		out := make(map[string]bool, len(typed))
		for key, value := range typed {
			out[key] = value
		}
		return out
	case []any:
		out := make([]any, len(typed))
		for i, item := range typed {
			out[i] = copyAny(item)
		}
		return out
	case []string:
		return append([]string(nil), typed...)
	case []int:
		return append([]int(nil), typed...)
	case []float64:
		out := make([]any, len(typed))
		for i, item := range typed {
			out[i] = yamlFloat(item, 64)
		}
		return out
	case []bool:
		return append([]bool(nil), typed...)
	case float32:
		return yamlFloat(float64(typed), 32)
	case float64:
		return yamlFloat(typed, 64)
	case json.Number:
		return yamlNumber(expandJSONNumber(typed.String()))
	default:
		return typed
	}
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
	if agent.Model.Auth != nil {
		out.Model.Auth = &abiAuth{Type: agent.Model.Auth.Type, TokenEnv: agent.Model.Auth.TokenEnv, Audience: agent.Model.Auth.Audience}
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
	for _, t := range agent.BrokeredTools {
		out.BrokeredTools = append(out.BrokeredTools, abiBrokeredTool{
			Name:          t.Name,
			Description:   t.Description,
			BrokeredClass: t.BrokeredClass,
			Parameters:    copyMap(t.Parameters),
			SchemaDigest:  t.SchemaDigest,
		})
	}
	if len(out.BrokeredTools) == 0 {
		out.BrokeredTools = nil
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
