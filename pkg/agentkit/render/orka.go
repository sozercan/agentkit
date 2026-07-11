// Package render emits deployment manifests for orchestrators that consume
// AgentKit-built images.
package render

import (
	"bytes"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/url"
	"strings"
	"unicode"

	"github.com/goccy/go-yaml"
)

const (
	TargetOrkaAgentRuntime = "orka-agentruntime"
	OrkaHarnessVersion     = "orka.harness.v1"
	DefaultAuthSecretKey   = "token"
)

// OrkaAgentRuntimeOptions controls rendering of an Orka AgentRuntime manifest.
type OrkaAgentRuntimeOptions struct {
	Name             string
	Image            string
	ExternalEndpoint string
	AuthSecretName   string
	AuthSecretKey    string
}

type metadata struct {
	Name string `yaml:"name"`
}

type agentRuntimeManifest struct {
	APIVersion string           `yaml:"apiVersion"`
	Kind       string           `yaml:"kind"`
	Metadata   metadata         `yaml:"metadata"`
	Spec       agentRuntimeSpec `yaml:"spec"`
}

type agentRuntimeSpec struct {
	ContractVersion string                 `yaml:"contractVersion"`
	Deployment      agentRuntimeDeployment `yaml:"deployment"`
	ClientAuth      agentRuntimeClientAuth `yaml:"clientAuth"`
	Capabilities    agentRuntimeCaps       `yaml:"capabilities"`
}

type agentRuntimeDeployment struct {
	Mode     string `yaml:"mode"`
	Endpoint string `yaml:"endpoint"`
}

type agentRuntimeClientAuth struct {
	BearerTokenSecretRef secretKeySelector `yaml:"bearerTokenSecretRef"`
}

type secretKeySelector struct {
	Name string `yaml:"name"`
	Key  string `yaml:"key"`
}

type agentRuntimeCaps struct {
	ToolExecutionModes      []string `yaml:"toolExecutionModes"`
	SupportsCancel          bool     `yaml:"supportsCancel"`
	SupportsRuntimeSessions bool     `yaml:"supportsRuntimeSessions"`
}

const invalidExternalEndpointMessage = "--external-endpoint must be an absolute http(s) URL with a host and no userinfo, query, fragment, or whitespace"

func validateExternalEndpoint(endpoint string) error {
	if (!strings.HasPrefix(endpoint, "http://") && !strings.HasPrefix(endpoint, "https://")) ||
		strings.IndexFunc(endpoint, unicode.IsSpace) >= 0 || strings.ContainsAny(endpoint, "@?#") {
		return errors.New(invalidExternalEndpointMessage)
	}
	parsed, err := url.Parse(endpoint)
	if err != nil || !parsed.IsAbs() || parsed.Opaque != "" || parsed.Hostname() == "" || parsed.User != nil ||
		parsed.RawQuery != "" || parsed.ForceQuery || parsed.Fragment != "" || parsed.RawFragment != "" ||
		(parsed.Scheme != "http" && parsed.Scheme != "https") {
		return errors.New(invalidExternalEndpointMessage)
	}
	return nil
}

// OrkaAgentRuntime renders a core.orka.ai AgentRuntime manifest for an
// AgentKit image that exposes the observed-mode orka.harness.v1 protocol.
func OrkaAgentRuntime(opts OrkaAgentRuntimeOptions) ([]byte, error) {
	name := strings.TrimSpace(opts.Name)
	if name == "" {
		return nil, errors.New("--name is required")
	}
	if strings.TrimSpace(opts.Image) != "" {
		return nil, errors.New("the current Orka AgentRuntime CRD supports external endpoints only; deploy the image with AGENTKIT_PROTOCOL=orka and AGENTKIT_AUTH_TOKEN from the bearer token Secret, then pass --external-endpoint")
	}
	endpoint := opts.ExternalEndpoint
	if endpoint == "" {
		return nil, errors.New("--external-endpoint is required for the current Orka AgentRuntime CRD")
	}
	if err := validateExternalEndpoint(endpoint); err != nil {
		return nil, err
	}
	authSecretName := strings.TrimSpace(opts.AuthSecretName)
	if authSecretName == "" {
		authSecretName = name + "-harness-token"
	}
	authSecretKey := strings.TrimSpace(opts.AuthSecretKey)
	if authSecretKey == "" {
		authSecretKey = DefaultAuthSecretKey
	}

	manifest := agentRuntimeManifest{
		APIVersion: "core.orka.ai/v1alpha1",
		Kind:       "AgentRuntime",
		Metadata:   metadata{Name: name},
		Spec: agentRuntimeSpec{
			ContractVersion: OrkaHarnessVersion,
			Deployment: agentRuntimeDeployment{
				Mode:     "external-endpoint",
				Endpoint: endpoint,
			},
			ClientAuth: agentRuntimeClientAuth{
				BearerTokenSecretRef: secretKeySelector{Name: authSecretName, Key: authSecretKey},
			},
			Capabilities: agentRuntimeCaps{
				ToolExecutionModes:      []string{"observed"},
				SupportsCancel:          true,
				SupportsRuntimeSessions: true,
			},
		},
	}
	out, err := yaml.Marshal(manifest)
	if err != nil {
		return nil, err
	}
	return append(bytes.TrimSpace(out), '\n'), nil
}

// RunCLI implements `agentkit render` for the frontend binary. The BuildKit
// frontend path still runs when no subcommand is provided.
func RunCLI(args []string, stdout io.Writer, stderr io.Writer) int {
	fs := flag.NewFlagSet("agentkit render", flag.ContinueOnError)
	fs.SetOutput(stderr)
	target := fs.String("target", "", "render target (orka-agentruntime)")
	image := fs.String("image", "", "agent image reference; not supported by current Orka external-endpoint CRD")
	name := fs.String("name", "", "AgentRuntime metadata.name")
	externalEndpoint := fs.String("external-endpoint", "", "external Orka harness endpoint")
	authSecretName := fs.String("auth-secret-name", "", "Secret name containing the Orka harness bearer token")
	authSecretKey := fs.String("auth-secret-key", DefaultAuthSecretKey, "Secret key containing the Orka harness bearer token")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *target != TargetOrkaAgentRuntime {
		fmt.Fprintf(stderr, "agentkit render: unsupported --target %q (expected %q)\n", *target, TargetOrkaAgentRuntime)
		return 2
	}
	out, err := OrkaAgentRuntime(OrkaAgentRuntimeOptions{
		Name:             *name,
		Image:            *image,
		ExternalEndpoint: *externalEndpoint,
		AuthSecretName:   *authSecretName,
		AuthSecretKey:    *authSecretKey,
	})
	if err != nil {
		fmt.Fprintf(stderr, "agentkit render: %v\n", err)
		return 2
	}
	_, _ = stdout.Write(out)
	return 0
}
