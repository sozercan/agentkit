// Package effective turns a validated authored AgentConfig plus resolved build
// inputs into the build-ready Agent value consumed by ABI and image writers.
package effective

import (
	"bytes"
	"encoding/json"
	"reflect"

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
	Metadata      config.Metadata
	Runtime       string
	Model         config.Model
	Instructions  string
	Tools         []config.Tool
	BrokeredTools []config.BrokeredTool
	Env           []config.EnvVar
	Context       config.Context
	Observability config.Observability
	Expose        config.Expose
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
		Runtime:       runtime,
		Model:         cfg.Model,
		Instructions:  instructions,
		Tools:         copyTools(cfg.Tools),
		BrokeredTools: copyBrokeredTools(cfg.BrokeredTools),
		Env:           copyEnvVars(cfg.Env),
		Context:       copyContext(cfg.Context),
		Observability: cfg.Observability,
		Expose:        expose,
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
		out[i].Headers = append([]config.ToolHeader(nil), tool.Headers...)
		if tool.Auth != nil {
			auth := *tool.Auth
			out[i].Auth = &auth
		}
		out[i].Env = append([]string(nil), tool.Env...)
	}
	return out
}

func copyBrokeredTools(in []config.BrokeredTool) []config.BrokeredTool {
	if len(in) == 0 {
		return nil
	}
	out := make([]config.BrokeredTool, len(in))
	for i, tool := range in {
		out[i] = tool
		out[i].Parameters = copyMap(tool.Parameters)
	}
	return out
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
	if v == nil {
		return nil
	}
	switch typed := v.(type) {
	case map[string]any:
		return copyMap(typed)
	case map[string]string:
		if typed == nil {
			return typed
		}
		out := make(map[string]string, len(typed))
		for key, value := range typed {
			out[key] = value
		}
		return out
	case map[string]int:
		if typed == nil {
			return typed
		}
		out := make(map[string]int, len(typed))
		for key, value := range typed {
			out[key] = value
		}
		return out
	case map[string]float64:
		if typed == nil {
			return typed
		}
		out := make(map[string]float64, len(typed))
		for key, value := range typed {
			out[key] = value
		}
		return out
	case map[string]bool:
		if typed == nil {
			return typed
		}
		out := make(map[string]bool, len(typed))
		for key, value := range typed {
			out[key] = value
		}
		return out
	case []any:
		if typed == nil {
			return typed
		}
		out := make([]any, len(typed))
		for i, item := range typed {
			out[i] = copyAny(item)
		}
		return out
	case []string:
		if typed == nil {
			return typed
		}
		out := make([]string, len(typed))
		copy(out, typed)
		return out
	case []int:
		if typed == nil {
			return typed
		}
		out := make([]int, len(typed))
		copy(out, typed)
		return out
	case []float64:
		if typed == nil {
			return typed
		}
		out := make([]float64, len(typed))
		copy(out, typed)
		return out
	case []bool:
		if typed == nil {
			return typed
		}
		out := make([]bool, len(typed))
		copy(out, typed)
		return out
	default:
		return copyReflectValue(v)
	}
}

func copyReflectValue(v any) any {
	value := reflect.ValueOf(v)
	switch value.Kind() {
	case reflect.Interface:
		if value.IsNil() {
			return nil
		}
		return copyAny(value.Elem().Interface())
	case reflect.Pointer:
		if value.IsNil() {
			return reflect.Zero(value.Type()).Interface()
		}
		if normalized, ok := copyJSONNormalized(value.Interface()); ok {
			return normalized
		}
		out := reflect.New(value.Type().Elem())
		copied := copyAny(value.Elem().Interface())
		out.Elem().Set(copiedReflectValue(copied, value.Type().Elem(), value.Elem()))
		return out.Interface()
	case reflect.Map:
		if value.IsNil() {
			return reflect.Zero(value.Type()).Interface()
		}
		out := reflect.MakeMapWithSize(value.Type(), value.Len())
		iter := value.MapRange()
		for iter.Next() {
			copied := copyAny(iter.Value().Interface())
			out.SetMapIndex(iter.Key(), copiedReflectValue(copied, value.Type().Elem(), iter.Value()))
		}
		return out.Interface()
	case reflect.Slice:
		if value.IsNil() {
			return reflect.Zero(value.Type()).Interface()
		}
		out := reflect.MakeSlice(value.Type(), value.Len(), value.Len())
		for i := 0; i < value.Len(); i++ {
			copied := copyAny(value.Index(i).Interface())
			out.Index(i).Set(copiedReflectValue(copied, value.Type().Elem(), value.Index(i)))
		}
		return out.Interface()
	case reflect.Array:
		out := reflect.New(value.Type()).Elem()
		for i := 0; i < value.Len(); i++ {
			copied := copyAny(value.Index(i).Interface())
			out.Index(i).Set(copiedReflectValue(copied, value.Type().Elem(), value.Index(i)))
		}
		return out.Interface()
	case reflect.Struct:
		if normalized, ok := copyJSONNormalized(value.Interface()); ok {
			return normalized
		}
		return value.Interface()
	default:
		return v
	}
}

func copyJSONNormalized(value any) (any, bool) {
	encoded, err := json.Marshal(value)
	if err != nil {
		return nil, false
	}
	decoder := json.NewDecoder(bytes.NewReader(encoded))
	decoder.UseNumber()
	var out any
	if err := decoder.Decode(&out); err != nil {
		return nil, false
	}
	return out, true
}

func copiedReflectValue(copied any, targetType reflect.Type, fallback reflect.Value) reflect.Value {
	if copied == nil {
		return reflect.Zero(targetType)
	}
	value := reflect.ValueOf(copied)
	if value.Type().AssignableTo(targetType) {
		return value
	}
	if value.Type().ConvertibleTo(targetType) {
		return value.Convert(targetType)
	}
	if rehydrated, ok := jsonNormalizedToType(copied, targetType); ok {
		return rehydrated
	}
	return fallback
}

func jsonNormalizedToType(value any, targetType reflect.Type) (reflect.Value, bool) {
	encoded, err := json.Marshal(value)
	if err != nil {
		return reflect.Value{}, false
	}
	target := reflect.New(targetType)
	decoder := json.NewDecoder(bytes.NewReader(encoded))
	decoder.UseNumber()
	if err := decoder.Decode(target.Interface()); err != nil {
		return reflect.Value{}, false
	}
	return target.Elem(), true
}

func copyEnvVars(in []config.EnvVar) []config.EnvVar {
	if len(in) == 0 {
		return nil
	}
	out := make([]config.EnvVar, len(in))
	copy(out, in)
	return out
}

func copyContext(in config.Context) config.Context {
	if len(in.Providers) == 0 {
		return config.Context{}
	}
	out := config.Context{Providers: make([]config.ContextProvider, len(in.Providers))}
	for i, provider := range in.Providers {
		out.Providers[i] = provider
		if provider.Auth != nil {
			auth := *provider.Auth
			out.Providers[i].Auth = &auth
		}
	}
	return out
}
