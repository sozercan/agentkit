package runtimes

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
		PydanticAI:         PydanticAI,         // canonical → itself
		MAF:                MAF,                // canonical → itself
		MAFAlias:           MAF,                // alias → canonical
		LangGraph:          LangGraph,          // canonical → itself
		nonexistentRuntime: nonexistentRuntime, // unknown returned verbatim
	}
	for in, want := range cases {
		if got := CanonicalRuntime(in); got != want {
			t.Errorf("CanonicalRuntime(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestIsKnownRuntime(t *testing.T) {
	for _, name := range []string{PydanticAI, MAF, MAFAlias, LangGraph} {
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

func TestKnownRuntimesContainsAll(t *testing.T) {
	got := KnownRuntimes()
	sort.Strings(got)
	want := []string{LangGraph, MAF, PydanticAI}
	if len(got) != len(want) {
		t.Fatalf("KnownRuntimes() = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("KnownRuntimes() = %v, want %v", got, want)
		}
	}
}

// TestRuntimesDeclarationInvariants locks the single-source-of-truth contract:
// every RuntimeSpec resolves by its own name and by each alias, carries a default
// adapter ref, and is reachable via RuntimeByName. Adding runtime #3 is one entry
// in runtimes.Runtimes — this test guarantees that entry is internally consistent.
func TestRuntimesDeclarationInvariants(t *testing.T) {
	if len(Runtimes) == 0 {
		t.Fatal("Runtimes is empty")
	}
	if DefaultRuntime() != Runtimes[0].Name {
		t.Errorf("DefaultRuntime()=%q, want first spec %q", DefaultRuntime(), Runtimes[0].Name)
	}
	for _, rt := range Runtimes {
		if rt.Name == "" {
			t.Error("a RuntimeSpec has an empty Name")
		}
		if rt.DefaultAdapterRef == "" {
			t.Errorf("runtime %q has no DefaultAdapterRef", rt.Name)
		}
		// Canonical name resolves to itself and is known.
		if CanonicalRuntime(rt.Name) != rt.Name || !IsKnownRuntime(rt.Name) {
			t.Errorf("runtime %q does not resolve to itself", rt.Name)
		}
		// Every alias resolves to the canonical name and is known.
		for _, alias := range rt.Aliases {
			if CanonicalRuntime(alias) != rt.Name {
				t.Errorf("alias %q does not resolve to %q", alias, rt.Name)
			}
			if !IsKnownRuntime(alias) {
				t.Errorf("alias %q is not known", alias)
			}
		}
		// RuntimeByName finds it by canonical name and by alias.
		if got, ok := RuntimeByName(rt.Name); !ok || got.Name != rt.Name {
			t.Errorf("RuntimeByName(%q) failed", rt.Name)
		}
		for _, alias := range rt.Aliases {
			if got, ok := RuntimeByName(alias); !ok || got.Name != rt.Name {
				t.Errorf("RuntimeByName(alias %q) failed", alias)
			}
		}
	}
	// An unknown name resolves verbatim and is not found.
	if _, ok := RuntimeByName(nonexistentRuntime); ok {
		t.Errorf("RuntimeByName(%q) should not be found", nonexistentRuntime)
	}
}
