// Package runtimes declares the runtime Adapter catalog used by validation, routing,
// and image labeling.
//
// Runtime identity lives here so BOTH pkg/agentkit/config (the validator) and
// pkg/build (the adapter registry) can agree on "what runtimes exist" without a
// config→build import cycle. This package owns the WHOLE declaration — canonical
// name, aliases, and the default adapter image ref — as a single source of truth;
// pkg/build merely DERIVES its route/registry maps from it. Adding a runtime is
// therefore one RuntimeSpec literal here (plus the adapter image itself); no
// second registry to keep in sync.

package runtimes

const (
	// PydanticAI is the default runtime adapter (the v0 runtime).
	PydanticAI = "pydantic-ai"

	// MAF is the Microsoft Agent Framework runtime adapter (runtime #2).
	MAF = "microsoft-agent-framework"
	// MAFAlias is a short, convenient alias for MAF that users may write in
	// `runtime:`; it resolves to MAF (see CanonicalRuntime).
	MAFAlias = "maf"
)

// RuntimeSpec is the complete declaration of one runtime adapter.
type RuntimeSpec struct {
	// Name is the canonical runtime identifier used in agentkitfile `runtime:`,
	// routes, and image labels.
	Name string
	// Aliases are alternate user-writable spellings that resolve to Name.
	Aliases []string
	// DefaultAdapterRef is the OCI ref of the serve adapter image used as the LLB
	// base when no `--build-arg adapter=` override is supplied.
	DefaultAdapterRef string
}

// Runtimes is the canonical, ordered list of runtime adapters AgentKit ships.
// THE single source of truth — pkg/build derives its registry from this; the
// validator and router consult the helpers below. Runtime #3 = one entry here.
var Runtimes = []RuntimeSpec{
	{
		Name:              PydanticAI,
		DefaultAdapterRef: "ghcr.io/sozercan/agentkit/serve-pydantic-ai:latest",
	},
	{
		Name:              MAF,
		Aliases:           []string{MAFAlias}, // "maf" → "microsoft-agent-framework"
		DefaultAdapterRef: "ghcr.io/sozercan/agentkit/serve-maf:latest",
	},
}

// DefaultRuntime is the runtime used when an agentkitfile does not name one. It is
// the first entry in Runtimes (pydantic-ai, the v0 runtime).
func DefaultRuntime() string {
	return Runtimes[0].Name
}

// CanonicalRuntime resolves a user-supplied runtime name (which may be an alias)
// to its canonical form. An empty string stays empty (the caller defaults it to
// the v0 runtime). An unknown name is returned unchanged so the validator can
// report it verbatim.
func CanonicalRuntime(name string) string {
	if name == "" {
		return ""
	}
	for _, rt := range Runtimes {
		if rt.Name == name {
			return name
		}
		for _, alias := range rt.Aliases {
			if alias == name {
				return rt.Name
			}
		}
	}
	return name
}

// IsKnownRuntime reports whether name (after alias resolution) is a recognized
// runtime. The empty string is NOT known here — callers treat empty as "use the
// default" before this check.
func IsKnownRuntime(name string) bool {
	if name == "" {
		return false
	}
	canonical := CanonicalRuntime(name)
	for _, rt := range Runtimes {
		if rt.Name == canonical {
			return true
		}
	}
	return false
}

// KnownRuntimes returns the canonical runtime names, for building human-readable
// error messages ("supported: ..."). Order follows Runtimes.
func KnownRuntimes() []string {
	out := make([]string, 0, len(Runtimes))
	for _, rt := range Runtimes {
		out = append(out, rt.Name)
	}
	return out
}

// RuntimeByName returns the RuntimeSpec for a name (canonical or alias), and
// whether it was found.
func RuntimeByName(name string) (RuntimeSpec, bool) {
	canonical := CanonicalRuntime(name)
	for _, rt := range Runtimes {
		if rt.Name == canonical {
			return rt, true
		}
	}
	return RuntimeSpec{}, false
}
