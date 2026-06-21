package utils

// Runtime identity lives here, in the dependency-light utils package, so BOTH
// pkg/agentkit/config (the validator) and pkg/build (the adapter registry) can
// agree on "what runtimes exist" without a config→build import cycle (plan §8 /
// Open Q4). utils owns the canonical NAMES and aliases; build owns the WIRING
// (adapter image refs, routes). A lockstep test keeps the two in sync.

// runtimeAliases maps user-writable aliases to their canonical runtime name.
// The canonical names themselves are always accepted (see CanonicalRuntime).
var runtimeAliases = map[string]string{
	RuntimeMAFAlias: RuntimeMAF, // "maf" → "microsoft-agent-framework"
}

// canonicalRuntimes is the set of canonical runtime names AgentKit recognizes.
// Adding a runtime here (plus its adapter registration in pkg/build) is the whole
// frontend surface of a new runtime.
var canonicalRuntimes = map[string]struct{}{
	RuntimePydanticAI: {},
	RuntimeMAF:        {},
}

// CanonicalRuntime resolves a user-supplied runtime name (which may be an alias)
// to its canonical form. An empty string stays empty (the caller defaults it to
// the v0 runtime). An unknown name is returned unchanged so the validator can
// report it verbatim.
func CanonicalRuntime(name string) string {
	if name == "" {
		return ""
	}
	if canonical, ok := runtimeAliases[name]; ok {
		return canonical
	}
	return name
}

// IsKnownRuntime reports whether name (after alias resolution) is a recognized
// runtime. The empty string is NOT known here — callers treat empty as "use the
// default" before this check.
func IsKnownRuntime(name string) bool {
	_, ok := canonicalRuntimes[CanonicalRuntime(name)]
	return ok
}

// KnownRuntimes returns the canonical runtime names, for building human-readable
// error messages ("supported: ..."). Order is not guaranteed; callers sort.
func KnownRuntimes() []string {
	out := make([]string, 0, len(canonicalRuntimes))
	for name := range canonicalRuntimes {
		out = append(out, name)
	}
	return out
}
