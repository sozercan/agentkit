package build

import "testing"

const wantImageRoute = "pydantic-ai/image"

func TestLookupRouteEmptyTargetDefaults(t *testing.T) {
	matched, _, rc, ok := lookupRoute("", "")
	if !ok {
		t.Fatal("empty target with empty runtime should resolve to the default image route")
	}
	if matched != wantImageRoute {
		t.Fatalf("matched = %q, want %s", matched, wantImageRoute)
	}
	if rc == nil || rc.Name != "pydantic-ai" {
		t.Fatalf("rc = %+v, want pydantic-ai", rc)
	}
}

func TestLookupRouteBareRuntime(t *testing.T) {
	matched, _, _, ok := lookupRoute("pydantic-ai", "pydantic-ai")
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

func TestAdapterRefOverride(t *testing.T) {
	rc := &RuntimeConfig{Name: "pydantic-ai", defaultAdapterRef: "default:latest"}
	if got := rc.AdapterRef(map[string]string{}); got != "default:latest" {
		t.Fatalf("no override: got %q", got)
	}
	if got := rc.AdapterRef(map[string]string{"build-arg:adapter": "local:test"}); got != "local:test" {
		t.Fatalf("override: got %q, want local:test", got)
	}
}
