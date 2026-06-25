package build

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/sozercan/agentkit/pkg/agentkit/config"
)

const (
	inlinePrompt = "Be helpful."
	filePath     = "prompts/system.md"
	filePrompt   = "Use tools carefully."
	missingPath  = "missing.md"
)

type fakeContextReader struct {
	files map[string][]byte
	calls []string
	err   error
}

func (r *fakeContextReader) ReadFile(_ context.Context, path string) ([]byte, error) {
	r.calls = append(r.calls, path)
	if r.err != nil {
		return nil, r.err
	}
	return r.files[path], nil
}

func TestResolveInstructionSourceInlineDoesNotReadContext(t *testing.T) {
	reader := &fakeContextReader{}
	got, err := resolveInstructionSource(context.Background(), reader, config.Source{Inline: inlinePrompt})
	if err != nil {
		t.Fatalf("resolve inline: %v", err)
	}
	if got != inlinePrompt {
		t.Fatalf("got %q", got)
	}
	if len(reader.calls) != 0 {
		t.Fatalf("inline source should not read context, calls=%v", reader.calls)
	}
}

func TestResolveInstructionSourceFileReadsContextPath(t *testing.T) {
	reader := &fakeContextReader{files: map[string][]byte{filePath: []byte(filePrompt)}}
	got, err := resolveInstructionSource(context.Background(), reader, config.Source{File: filePath})
	if err != nil {
		t.Fatalf("resolve file: %v", err)
	}
	if got != filePrompt {
		t.Fatalf("got %q", got)
	}
	if len(reader.calls) != 1 || reader.calls[0] != filePath {
		t.Fatalf("calls=%v, want [%s]", reader.calls, filePath)
	}
}

func TestResolveInstructionSourceFileErrorIncludesPath(t *testing.T) {
	reader := &fakeContextReader{err: errors.New("boom")}
	_, err := resolveInstructionSource(context.Background(), reader, config.Source{File: missingPath})
	if err == nil {
		t.Fatal("expected error")
	}
	msg := err.Error()
	if !strings.Contains(msg, missingPath) || !strings.Contains(msg, "boom") {
		t.Fatalf("error %q does not include path and cause", msg)
	}
}
