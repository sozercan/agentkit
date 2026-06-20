package config

import (
	"errors"
	"fmt"

	"github.com/goccy/go-yaml"
	"github.com/sozercan/agentkit/pkg/utils"
)

// probe is the minimal {apiVersion, kind} discriminator read before full
// unmarshal. Reading kind explicitly is the fix for AIKit's silent-misparse bug
// (plan §5.1): dispatch on kind, never on an implicit type collision.
type probe struct {
	APIVersion string `yaml:"apiVersion"`
	Kind       string `yaml:"kind"`
}

// NewFromBytes parses an agentkitfile. It first reads a {apiVersion, kind}
// probe (non-strict, so the probe tolerates other fields), dispatches on kind,
// then strict-unmarshals the full AgentConfig so unknown/misspelled fields are
// load-time errors with line:column positions.
//
// Only kind: Agent is supported. A kind-less file is rejected with a clear
// message rather than silently misparsing.
func NewFromBytes(b []byte) (*AgentConfig, error) {
	var p probe
	if err := yaml.Unmarshal(b, &p); err != nil {
		return nil, fmt.Errorf("reading apiVersion/kind probe: %w", formatYAMLError(b, err))
	}

	if p.Kind == "" {
		return nil, errors.New("kind is not defined; expected `kind: Agent`")
	}
	if p.Kind != utils.KindAgent {
		return nil, fmt.Errorf("kind %q is not supported; expected `kind: Agent`", p.Kind)
	}

	cfg := &AgentConfig{}
	if err := yaml.UnmarshalWithOptions(b, cfg, yaml.Strict()); err != nil {
		return nil, fmt.Errorf("parsing agentkitfile: %w", formatYAMLError(b, err))
	}
	return cfg, nil
}

// formatYAMLError enriches a goccy decode error with line:column context.
func formatYAMLError(_ []byte, err error) error {
	if err == nil {
		return nil
	}
	// FormatError renders the offending line and a caret; colored=false for logs.
	return errors.New(yaml.FormatError(err, false, true))
}
