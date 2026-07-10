package build_test

import (
	"testing"

	build "github.com/sozercan/agentkit/pkg/build"
)

// Downstream users historically constructed Route with an unkeyed one-field
// literal. Keep that source-compatible layout even though Build has an internal
// context-aware dispatch path.
func TestRouteUnkeyedLiteralSourceCompatibility(t *testing.T) {
	route := build.Route{nil}
	if route.Handler != nil {
		t.Fatal("zero handler unexpectedly became non-nil")
	}
}
