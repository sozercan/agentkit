package build

import (
	"context"
	"sort"
	"strings"

	"github.com/moby/buildkit/frontend/gateway/client"
	"github.com/sozercan/agentkit/pkg/agentkit/config"
	"github.com/sozercan/agentkit/pkg/utils"
)

// RouteHandler builds a result for a resolved <runtime>/<outputkind> route.
type RouteHandler func(ctx context.Context, c client.Client, cfg *config.AgentConfig, rc *RuntimeConfig) (*client.Result, error)

// Route is one entry in the flat router (plan §7.1). Output kinds are
// re-packagings of the one agent layer, so v0 has a single handler (image);
// adding agentpack/compose later is a new route, not a Build() rewrite.
type Route struct {
	Handler RouteHandler
}

// RuntimeConfig describes a swappable runtime adapter (the DistroConfig analog,
// plan §7.1). v0 registers exactly one (pydantic-ai). A second runtime is one
// init() registration with zero edits to Build().
type RuntimeConfig struct {
	// Name is the runtime identifier used in agentkitfile `runtime:` and routes.
	Name string
	// defaultAdapterRef is the OCI ref of the serve adapter image used as the
	// LLB base when no `adapter` build-arg override is supplied.
	defaultAdapterRef string
}

// AdapterRef returns the adapter image ref to use as the LLB base, honoring the
// `--build-arg adapter=` override (the local dev-loop seam, plan §15.6) and
// falling back to the runtime's default.
func (rc *RuntimeConfig) AdapterRef(opts map[string]string) string {
	if ref := getBuildArg(opts, "adapter"); ref != "" {
		return ref
	}
	return rc.defaultAdapterRef
}

// routes maps "<runtime>/<outputkind>" to a handler. runtimes maps a runtime
// name to its config. Both are populated in init() — dispatch as data.
var (
	routes   = map[string]Route{}
	runtimes = map[string]*RuntimeConfig{}
)

// registerRuntime wires a runtime adapter and its image output route.
func registerRuntime(rc *RuntimeConfig) {
	runtimes[rc.Name] = rc
	routes[rc.Name+"/"+utils.OutputKindImage] = Route{Handler: HandleAgent}
}

func init() {
	registerRuntime(&RuntimeConfig{
		Name:              utils.RuntimePydanticAI,
		defaultAdapterRef: "ghcr.io/sozercan/agentkit/serve-pydantic-ai:latest",
	})
}

// defaultRuntime returns the runtime to use when the config does not name one.
func defaultRuntime() string {
	return utils.RuntimePydanticAI
}

// lookupRoute resolves a build target plus the effective runtime to a route and
// its RuntimeConfig.
//
// Target semantics (Dalec lookupTarget, plan §15.5.4 — empty target is NOT an
// error here; unlike Dalec we have a single sensible default output): an empty
// target resolves to "<runtime>/image". A bare runtime ("pydantic-ai") resolves
// to "<runtime>/image". Otherwise exact match, then longest-prefix match.
func lookupRoute(target, runtime string) (matched string, route Route, rc *RuntimeConfig, ok bool) {
	if runtime == "" {
		runtime = defaultRuntime()
	}
	rc, rcOK := runtimes[runtime]
	if !rcOK {
		return "", Route{}, nil, false
	}

	// Empty or bare-runtime target → the runtime's image route.
	if target == "" || target == runtime {
		r, rOK := routes[runtime+"/"+utils.OutputKindImage]
		return runtime + "/" + utils.OutputKindImage, r, rc, rOK
	}

	// Exact match.
	if r, rOK := routes[target]; rOK {
		return target, r, rc, true
	}

	// Longest-prefix match on "<key>/...".
	var candidates []string
	for k := range routes {
		if strings.HasPrefix(target, k+"/") {
			candidates = append(candidates, k)
		}
	}
	if len(candidates) > 0 {
		sort.Strings(candidates)
		k := candidates[len(candidates)-1]
		return k, routes[k], rc, true
	}

	return "", Route{}, nil, false
}
