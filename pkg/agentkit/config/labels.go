package config

import (
	"strings"

	"github.com/sozercan/agentkit/pkg/utils"
)

const (
	nativeImageLabelNamespace   = utils.LabelPrefix
	portableImageLabelNamespace = "ai.agentkit"
	orkaImageLabelNamespace     = "ai.orka"

	// ImageLabelNativeRuntime identifies the canonical AgentKit runtime.
	ImageLabelNativeRuntime = nativeImageLabelNamespace + ".runtime"
	// ImageLabelNativeName identifies the authored AgentKit agent name.
	ImageLabelNativeName = nativeImageLabelNamespace + ".name"
	// ImageLabelNativeABI identifies the baked Agent YAML ABI version.
	ImageLabelNativeABI = nativeImageLabelNamespace + ".abi"

	// ImageLabelPortableABI is the cross-orchestrator Agent YAML ABI label.
	ImageLabelPortableABI = portableImageLabelNamespace + ".abi"
	// ImageLabelPortableRuntime is the cross-orchestrator runtime identity label.
	ImageLabelPortableRuntime = portableImageLabelNamespace + ".runtime"
	// ImageLabelPortableProtocols lists the protocols exposed by AgentKit images.
	ImageLabelPortableProtocols = portableImageLabelNamespace + ".protocols"
	// ImageLabelPortableCapabilities lists the selected runtime's capabilities.
	ImageLabelPortableCapabilities = portableImageLabelNamespace + ".capabilities"

	// ImageLabelOrkaHarnessVersion identifies the supported Orka harness contract.
	ImageLabelOrkaHarnessVersion = orkaImageLabelNamespace + ".harness.version"

	// ImageLabelOCITitle is the standard OCI title generated from metadata.name.
	ImageLabelOCITitle = "org.opencontainers.image.title"
)

var reservedMetadataLabelNamespaces = [...]string{
	nativeImageLabelNamespace,
	portableImageLabelNamespace,
	orkaImageLabelNamespace,
}

var reservedMetadataLabelKeys = [...]string{
	ImageLabelOCITitle,
}

func reservedMetadataLabelNamespace(key string) (string, bool) {
	for _, namespace := range reservedMetadataLabelNamespaces {
		if key == namespace || strings.HasPrefix(key, namespace+".") {
			return namespace, true
		}
	}
	return "", false
}

func isReservedMetadataLabelKey(key string) bool {
	for _, reserved := range reservedMetadataLabelKeys {
		if key == reserved {
			return true
		}
	}
	return false
}
