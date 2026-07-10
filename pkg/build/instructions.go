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
// the BuildKit context. Tests use an in-memory adapter; production binds it to
// either a resolved remote reference or the local context session input.
type contextFileReader interface {
	ReadFile(ctx context.Context, path string) ([]byte, error)
}

// localContextReader reads files from BuildKit's local context session input.
type localContextReader struct {
	client client.Client
}

func (r localContextReader) ReadFile(ctx context.Context, path string) ([]byte, error) {
	state := llb.Local(localNameContext,
		llb.IncludePatterns([]string{path}),
		llb.SessionID(r.client.BuildOpts().SessionID),
		llb.SharedKeyHint("agentkit-instructions"),
		dockerui.WithInternalName("load instructions "+path),
	)
	ref, err := solveStateReference(ctx, r.client, &state, "local instructions source")
	if err != nil {
		return nil, err
	}
	return readReferenceFile(ctx, ref, path, "local build context")
}

// referenceContextReader reuses one solved remote context reference, ensuring
// relative instruction files come from the same Git or HTTP archive snapshot as
// the Agentkitfile.
type referenceContextReader struct {
	ref         client.Reference
	description string
}

func (r referenceContextReader) ReadFile(ctx context.Context, path string) ([]byte, error) {
	return readRemoteReferenceFile(ctx, r.ref, client.ReadRequest{Filename: path}, r.description)
}

type unsupportedContextReader struct {
	err error
}

func (r unsupportedContextReader) ReadFile(context.Context, string) ([]byte, error) {
	if r.err == nil {
		return nil, errors.New("instructions.file is not supported by this build context")
	}
	return nil, r.err
}

// resolveInstructionSource returns the fully-resolved system prompt: inline
// content as-is, or file contents read from the supplied build-context reader.
func resolveInstructionSource(ctx context.Context, reader contextFileReader, source config.Source) (string, error) {
	if source.File == "" {
		return source.Inline, nil
	}
	if reader == nil {
		return "", errors.Errorf("failed to read instructions file %s: build context reader is nil", source.File)
	}

	dt, err := reader.ReadFile(ctx, source.File)
	if err != nil {
		return "", errors.Wrapf(err, "failed to read instructions file %s", source.File)
	}
	return string(dt), nil
}
