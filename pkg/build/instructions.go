package build

import (
	"context"

	"github.com/moby/buildkit/client/llb"
	"github.com/moby/buildkit/frontend/dockerui"
	"github.com/moby/buildkit/frontend/gateway/client"
	"github.com/pkg/errors"
	"github.com/sozercan/agentkit/pkg/agentkit/config"
)

// contextFileReader is the small seam between authored instruction sources and
// the BuildKit context. Tests use an in-memory Adapter; production uses the
// BuildKit gateway client Adapter below.
type contextFileReader interface {
	ReadFile(ctx context.Context, path string) ([]byte, error)
}

// buildkitContextReader reads files from the build context through BuildKit.
type buildkitContextReader struct {
	client client.Client
}

func (r buildkitContextReader) ReadFile(ctx context.Context, path string) ([]byte, error) {
	localSt := llb.Local(localNameContext,
		llb.IncludePatterns([]string{path}),
		llb.SessionID(r.client.BuildOpts().SessionID),
		llb.SharedKeyHint("agentkit-instructions"),
		dockerui.WithInternalName("load instructions "+path),
	)
	def, err := localSt.Marshal(ctx)
	if err != nil {
		return nil, errors.Wrap(err, "failed to marshal instructions source")
	}
	res, err := r.client.Solve(ctx, client.SolveRequest{Definition: def.ToPB()})
	if err != nil {
		return nil, errors.Wrap(err, "failed to resolve instructions source")
	}
	ref, err := res.SingleRef()
	if err != nil {
		return nil, err
	}
	dt, err := ref.ReadFile(ctx, client.ReadRequest{Filename: path})
	if err != nil {
		return nil, errors.Wrap(err, "failed to read context file")
	}
	return dt, nil
}

// resolveInstructions returns the fully-resolved system prompt: inline content
// as-is, or file contents read from the build context.
func resolveInstructions(ctx context.Context, c client.Client, cfg *config.AgentConfig) (string, error) {
	return resolveInstructionSource(ctx, buildkitContextReader{client: c}, cfg.Instructions)
}

func resolveInstructionSource(ctx context.Context, reader contextFileReader, source config.Source) (string, error) {
	if source.File == "" {
		return source.Inline, nil
	}

	dt, err := reader.ReadFile(ctx, source.File)
	if err != nil {
		return "", errors.Wrapf(err, "failed to read instructions file %s", source.File)
	}
	return string(dt), nil
}
