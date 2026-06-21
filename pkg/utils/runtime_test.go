package utils

import (
	"sort"
	"testing"
)

// nonexistentRuntime is a name that is deliberately not registered, reused across
// the negative cases below.
const nonexistentRuntime = "nonexistent"

func TestCanonicalRuntime(t *testing.T) {
	cases := map[string]string{
		"":                 "",                 // empty stays empty (caller defaults it)
		RuntimePydanticAI:  RuntimePydanticAI,  // canonical → itself
		RuntimeMAF:         RuntimeMAF,         // canonical → itself
		RuntimeMAFAlias:    RuntimeMAF,         // alias → canonical
		nonexistentRuntime: nonexistentRuntime, // unknown returned verbatim
	}
	for in, want := range cases {
		if got := CanonicalRuntime(in); got != want {
			t.Errorf("CanonicalRuntime(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestIsKnownRuntime(t *testing.T) {
	for _, name := range []string{RuntimePydanticAI, RuntimeMAF, RuntimeMAFAlias} {
		if !IsKnownRuntime(name) {
			t.Errorf("IsKnownRuntime(%q) = false, want true", name)
		}
	}
	for _, name := range []string{"", nonexistentRuntime, "MAF", "pydantic"} {
		if IsKnownRuntime(name) {
			t.Errorf("IsKnownRuntime(%q) = true, want false", name)
		}
	}
}

func TestKnownRuntimesContainsBoth(t *testing.T) {
	got := KnownRuntimes()
	sort.Strings(got)
	want := []string{RuntimeMAF, RuntimePydanticAI}
	if len(got) != len(want) {
		t.Fatalf("KnownRuntimes() = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("KnownRuntimes() = %v, want %v", got, want)
		}
	}
}
