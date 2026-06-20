package utils

import (
	"github.com/moby/buildkit/client/llb"
)

// Phase is a composable, named mutation of a working LLB state. Each AgentKit
// build concern (stage the runtime adapter, stage an MCP server, write the
// agent config, write the entrypoint) is expressed as a Phase and chained with
// ApplyPhases, which isolates each concern's filesystem delta and merges it
// onto a lean base — AIKit's savedState->Diff->Merge idiom, centralized once
// (plan §16.2 #4) instead of copy-pasted at every call site.
type Phase struct {
	// Name is a human-readable label surfaced in BuildKit progress output.
	Name string
	// Run mutates the (fat) working state s and returns the new working state.
	// Whatever it adds to s relative to the input is captured as an isolated
	// diff and grafted onto base; build tooling that runs but leaves no
	// artifact in s never lands in the final image.
	Run func(s llb.State) (llb.State, error)
}

// ApplyPhases threads a working state through each phase, collecting every
// phase's filesystem delta as an independent, cacheable diff merged onto base.
//
//	merge := base
//	for each phase:
//	    saved := s
//	    s = phase.Run(s)            // mutate the fat working state
//	    diff  := llb.Diff(saved, s) // isolate only what this phase changed
//	    merge := llb.Merge(merge, diff)
//
// The returned merge state is lean (only the phase deltas land on base); the
// returned working state s carries the full toolchain for any subsequent phase
// that needs it. Callers ship merge, never s.
func ApplyPhases(base, working llb.State, phases ...Phase) (merge, s llb.State, err error) {
	merge = base
	s = working
	for _, p := range phases {
		saved := s
		s, err = p.Run(s)
		if err != nil {
			return llb.State{}, llb.State{}, err
		}
		var diff llb.State
		if p.Name != "" {
			diff = llb.Diff(saved, s, llb.WithCustomName(p.Name))
		} else {
			diff = llb.Diff(saved, s)
		}
		merge = llb.Merge([]llb.State{merge, diff})
	}
	return merge, s, nil
}
