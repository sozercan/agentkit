// Package utils holds small, dependency-light helpers shared across the
// AgentKit frontend: shell-command constructors, URL helpers, and the
// canonical BuildKit Diff/Merge idiom used by the LLB converter.
package utils

const (
	// APIv1alpha1 is the only supported apiVersion for v0.
	APIv1alpha1 = "v1alpha1"

	// KindAgent is the config discriminator for an agent build (see §5.1).
	KindAgent = "Agent"

	// ProviderOpenAICompatible is the only model provider supported in v0.
	ProviderOpenAICompatible = "openai-compatible"

	// OutputKindImage is the default (and only v0) output kind.
	OutputKindImage = "image"

	// Platform constants.
	PlatformLinux = "linux"
	PlatformAMD64 = "amd64"
	PlatformARM64 = "arm64"

	// AgentConfigPath is where the resolved agent.yaml is baked in the image.
	// It is the ABI boundary between the frontend (writer) and agentkit-serve (reader).
	AgentConfigPath = "/agent/agent.yaml"

	// AgentKitRoot is where the runtime adapter (agentkit-serve) is staged.
	AgentKitRoot = "/opt/agentkit"

	// ServeBinary is the agentkit-serve entrypoint inside the adapter payload.
	ServeBinary = AgentKitRoot + "/bin/agentkit-serve"

	// DefaultPort is the default serve port.
	DefaultPort = 8080

	// LabelPrefix namespaces all AgentKit OCI labels (ai.<ns>.agentkit.*).
	LabelPrefix = "ai.sozercan.agentkit"
)
