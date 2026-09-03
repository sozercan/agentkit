package build

import "testing"

const (
	wantImageRoute     = "pydantic-ai/image"
	runtimePydca       = "pydantic-ai"
	runtimeMAFName     = "microsoft-agent-framework"
	runtimeMAFAls      = "maf"
	wantMAFRoute       = "microsoft-agent-framework/image"
	runtimeLangGraph   = "langgraph"
	wantLangGraphRoute = "langgraph/image"
)

func TestLookupRouteEmptyTargetDefaults(t *testing.T) {
	matched, _, rc, ok := lookupRoute("", "")
	if !ok {
		t.Fatal("empty target with empty runtime should resolve to the default image route")
	}
	if matched != wantImageRoute {
		t.Fatalf("matched = %q, want %s", matched, wantImageRoute)
	}
	if rc == nil || rc.Name != runtimePydca {
		t.Fatalf("rc = %+v, want pydantic-ai", rc)
	}
}

func TestLookupRouteBareRuntime(t *testing.T) {
	matched, _, _, ok := lookupRoute(runtimePydca, runtimePydca)
	if !ok || matched != wantImageRoute {
		t.Fatalf("bare runtime target: matched=%q ok=%v, want %s", matched, ok, wantImageRoute)
	}
}

func TestLookupRouteExact(t *testing.T) {
	matched, _, _, ok := lookupRoute("pydantic-ai/image", "")
	if !ok || matched != wantImageRoute {
		t.Fatalf("exact target: matched=%q ok=%v", matched, ok)
	}
}

func TestLookupRouteUnknownRuntime(t *testing.T) {
	if _, _, _, ok := lookupRoute("", "nonexistent"); ok {
		t.Fatal("unknown runtime should not resolve")
	}
}

func TestLookupRouteRejectsConflictingRuntimeTarget(t *testing.T) {
	matched, _, rc, ok := lookupRoute(wantLangGraphRoute, runtimePydca)
	if ok {
		t.Fatalf("conflicting target resolved: matched=%q rc=%+v, want no route", matched, rc)
	}
}

// TestLookupRouteLangGraph proves the LangGraph runtime resolves through the
// same data-derived router as pydantic-ai and MAF.
func TestLookupRouteLangGraph(t *testing.T) {
	// empty target + LangGraph runtime → LangGraph image route.
	matched, _, rc, ok := lookupRoute("", runtimeLangGraph)
	if !ok || matched != wantLangGraphRoute {
		t.Fatalf("LangGraph empty target: matched=%q ok=%v, want %s", matched, ok, wantLangGraphRoute)
	}
	if rc == nil || rc.Name != runtimeLangGraph {
		t.Fatalf("rc = %+v, want langgraph", rc)
	}
	// exact target match.
	if m, _, _, okExact := lookupRoute(wantLangGraphRoute, runtimeLangGraph); !okExact || m != wantLangGraphRoute {
		t.Fatalf("LangGraph exact target: matched=%q ok=%v", m, okExact)
	}
	// bare runtime target.
	if m, _, _, okBare := lookupRoute(runtimeLangGraph, runtimeLangGraph); !okBare || m != wantLangGraphRoute {
		t.Fatalf("LangGraph bare target: matched=%q ok=%v", m, okBare)
	}
}

// TestLookupRouteMAF proves the second runtime resolves through the SAME flat
// router with zero handler changes (plan §8 — "the router already handles the
// second runtime").
func TestLookupRouteMAF(t *testing.T) {
	// empty target + MAF runtime → MAF image route.
	matched, _, rc, ok := lookupRoute("", runtimeMAFName)
	if !ok || matched != wantMAFRoute {
		t.Fatalf("MAF empty target: matched=%q ok=%v, want %s", matched, ok, wantMAFRoute)
	}
	if rc == nil || rc.Name != runtimeMAFName {
		t.Fatalf("rc = %+v, want microsoft-agent-framework", rc)
	}
	// exact target match.
	if m, _, _, okExact := lookupRoute(wantMAFRoute, runtimeMAFName); !okExact || m != wantMAFRoute {
		t.Fatalf("MAF exact target: matched=%q ok=%v", m, okExact)
	}
}

// TestLookupRouteMAFAlias proves the "maf" alias resolves to the canonical MAF
// route (alias handling lives in runtimes.CanonicalRuntime, consumed by lookupRoute).
func TestLookupRouteMAFAlias(t *testing.T) {
	matched, _, rc, ok := lookupRoute("", runtimeMAFAls)
	if !ok || matched != wantMAFRoute {
		t.Fatalf("maf alias: matched=%q ok=%v, want %s", matched, ok, wantMAFRoute)
	}
	if rc == nil || rc.Name != runtimeMAFName {
		t.Fatalf("rc = %+v, want microsoft-agent-framework", rc)
	}
}

// TestLookupRouteAliasAsTarget proves the alias also resolves when it is the
// BUILD TARGET (not just the runtime arg): both bare ("maf") and the route form
// ("maf/image") canonicalize and route to the MAF image route, matching
// pydantic-ai's bare-runtime-as-target behavior. (rc follows the `runtime` arg —
// the authoritative runtime source in build.go — so these cases pass the MAF
// runtime too, exercising the target-canonicalization path.)
func TestLookupRouteAliasAsTarget(t *testing.T) {
	cases := []struct{ target, runtime string }{
		{runtimeMAFAls, runtimeMAFAls},            // bare alias as target + runtime
		{runtimeMAFAls + "/image", runtimeMAFAls}, // alias/image target + alias runtime
		{runtimeMAFAls, runtimeMAFName},           // alias target, canonical runtime
		{runtimeMAFName, runtimeMAFAls},           // canonical target, alias runtime
	}
	for _, tc := range cases {
		matched, _, rc, ok := lookupRoute(tc.target, tc.runtime)
		if !ok || matched != wantMAFRoute {
			t.Errorf("lookupRoute(%q, %q): matched=%q ok=%v, want %s", tc.target, tc.runtime, matched, ok, wantMAFRoute)
		}
		if rc == nil || rc.Name != runtimeMAFName {
			t.Errorf("lookupRoute(%q, %q): rc=%+v, want microsoft-agent-framework", tc.target, tc.runtime, rc)
		}
	}
}

func TestLookupRouteRejectsConflictingAliasAndPrefixedTargets(t *testing.T) {
	cases := []struct {
		target  string
		runtime string
	}{
		{runtimeMAFAls, ""},
		{runtimeMAFAls + "/image", runtimePydca},
		{runtimeMAFAls + "/image/debug", runtimePydca},
		{runtimeMAFName + "/image/debug", runtimeLangGraph},
	}
	for _, tc := range cases {
		matched, _, rc, ok := lookupRoute(tc.target, tc.runtime)
		if ok {
			t.Errorf("lookupRoute(%q, %q) resolved: matched=%q rc=%+v, want no route", tc.target, tc.runtime, matched, rc)
		}
	}
}

func TestLookupRouteMatchingRuntimePreservesOutputSuffix(t *testing.T) {
	cases := []struct {
		target      string
		runtime     string
		wantRoute   string
		wantRuntime string
	}{
		{wantImageRoute + "/debug", runtimePydca, wantImageRoute, runtimePydca},
		{runtimeMAFAls + "/image/debug", runtimeMAFName, wantMAFRoute, runtimeMAFName},
		{wantMAFRoute + "/debug/more", runtimeMAFAls, wantMAFRoute, runtimeMAFName},
		{wantLangGraphRoute + "/debug", runtimeLangGraph, wantLangGraphRoute, runtimeLangGraph},
	}
	for _, tc := range cases {
		matched, _, rc, ok := lookupRoute(tc.target, tc.runtime)
		if !ok || matched != tc.wantRoute {
			t.Errorf("lookupRoute(%q, %q): matched=%q ok=%v, want %q", tc.target, tc.runtime, matched, ok, tc.wantRoute)
		}
		if rc == nil || rc.Name != tc.wantRuntime {
			t.Errorf("lookupRoute(%q, %q): rc=%+v, want runtime %q", tc.target, tc.runtime, rc, tc.wantRuntime)
		}
	}
}

// TestIsRegisteredRuntime locks the validator's seam: every canonical runtime and
// the alias are registered; an unknown name is not.
func TestIsRegisteredRuntime(t *testing.T) {
	for _, name := range []string{runtimePydca, runtimeMAFName, runtimeMAFAls, runtimeLangGraph} {
		if !IsRegisteredRuntime(name) {
			t.Errorf("IsRegisteredRuntime(%q) = false, want true", name)
		}
	}
	if IsRegisteredRuntime("nonexistent") {
		t.Error("IsRegisteredRuntime(\"nonexistent\") = true, want false")
	}
	if IsRegisteredRuntime("") {
		t.Error("IsRegisteredRuntime(\"\") = true, want false (empty is 'use default', handled upstream)")
	}
}

func TestAdapterRefOverride(t *testing.T) {
	rc := &RuntimeConfig{Name: runtimePydca, defaultAdapterRef: "default:latest"}
	if got := rc.AdapterRef(map[string]string{}); got != "default:latest" {
		t.Fatalf("no override: got %q", got)
	}
	if got := rc.AdapterRef(map[string]string{"build-arg:adapter": "local:test"}); got != "local:test" {
		t.Fatalf("override: got %q, want local:test", got)
	}
}
