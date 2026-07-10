package build

import (
	"archive/tar"
	"bytes"
	"context"
	"errors"
	"fmt"
	"net/url"
	"strings"
	"sync"
	"testing"

	"github.com/moby/buildkit/client/llb"
	"github.com/moby/buildkit/frontend/gateway/client"
	"github.com/moby/buildkit/solver/pb"
	fstypes "github.com/tonistiigi/fsutil/types"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

const (
	remoteContextURL              = "https://example.com/agent.git#main"
	localDockerfileSourcePrefix   = "local://dockerfile"
	dockerImageSourcePrefix       = "docker-image://"
	downloadedHTTPContextFilename = "context"
	redactionTestPrefix           = "redaction"
)

type memoryReference struct {
	mu      sync.Mutex
	files   map[string][]byte
	reads   []string
	readErr error
}

func (r *memoryReference) ToState() (llb.State, error) {
	return llb.Scratch(), nil
}

func (r *memoryReference) Evaluate(context.Context) error {
	return nil
}

func (r *memoryReference) ReadFile(_ context.Context, req client.ReadRequest) ([]byte, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	name := strings.TrimPrefix(req.Filename, "/")
	r.reads = append(r.reads, name)
	if r.readErr != nil {
		return nil, r.readErr
	}
	dt, ok := r.files[name]
	if !ok {
		return nil, fmt.Errorf("file %q not found", req.Filename)
	}
	if req.Range == nil {
		return append([]byte(nil), dt...), nil
	}
	start := req.Range.Offset
	if start < 0 || start > len(dt) {
		return nil, fmt.Errorf("invalid offset %d for %q", start, req.Filename)
	}
	end := len(dt)
	if req.Range.Length > 0 && start+req.Range.Length < end {
		end = start + req.Range.Length
	}
	return append([]byte(nil), dt[start:end]...), nil
}

func (r *memoryReference) StatFile(context.Context, client.StatRequest) (*fstypes.Stat, error) {
	return nil, fmt.Errorf("StatFile not implemented")
}

func (r *memoryReference) ReadDir(context.Context, client.ReadDirRequest) ([]*fstypes.Stat, error) {
	return nil, fmt.Errorf("ReadDir not implemented")
}

func (r *memoryReference) readPaths() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]string(nil), r.reads...)
}

type expectedSolve struct {
	sourcePrefix         string
	requireAttemptUnpack bool
	ref                  client.Reference
	err                  error
}

type fakeBuildClient struct {
	client.Client

	opts client.BuildOpts

	mu             sync.Mutex
	expected       []expectedSolve
	sources        [][]string
	operationNames [][]string
}

func (c *fakeBuildClient) BuildOpts() client.BuildOpts {
	return c.opts
}

func (c *fakeBuildClient) Solve(_ context.Context, req client.SolveRequest) (*client.Result, error) {
	sources, err := definitionSources(req.Definition)
	if err != nil {
		return nil, err
	}
	names := definitionOperationNames(req.Definition)

	c.mu.Lock()
	defer c.mu.Unlock()
	c.sources = append(c.sources, sources)
	c.operationNames = append(c.operationNames, names)
	index := len(c.sources) - 1
	if index >= len(c.expected) {
		return nil, fmt.Errorf("unexpected solve %d with sources %v", index+1, sources)
	}
	expected := c.expected[index]
	if !hasSourcePrefix(sources, expected.sourcePrefix) {
		return nil, fmt.Errorf("solve %d sources %v do not include %q", index+1, sources, expected.sourcePrefix)
	}
	if expected.requireAttemptUnpack && !definitionAttemptsUnpack(req.Definition) {
		return nil, fmt.Errorf("solve %d does not unpack the HTTP archive", index+1)
	}
	if expected.err != nil {
		return nil, expected.err
	}
	result := client.NewResult()
	result.SetRef(expected.ref)
	return result, nil
}

func (c *fakeBuildClient) allOperationNames() []string {
	c.mu.Lock()
	defer c.mu.Unlock()
	var out []string
	for _, names := range c.operationNames {
		out = append(out, names...)
	}
	return out
}

func (c *fakeBuildClient) solveSources() [][]string {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make([][]string, len(c.sources))
	for i := range c.sources {
		out[i] = append([]string(nil), c.sources[i]...)
	}
	return out
}

func definitionSources(def *pb.Definition) ([]string, error) {
	if def == nil {
		return nil, nil
	}
	var sources []string
	for _, dt := range def.Def {
		var op pb.Op
		if err := op.Unmarshal(dt); err != nil {
			return nil, fmt.Errorf("unmarshal solve op: %w", err)
		}
		if source := op.GetSource(); source != nil {
			sources = append(sources, source.Identifier)
		}
	}
	return sources, nil
}

func definitionOperationNames(def *pb.Definition) []string {
	if def == nil {
		return nil
	}
	var names []string
	for _, metadata := range def.Metadata {
		if metadata.Description == nil {
			continue
		}
		if name := metadata.Description["llb.customname"]; name != "" {
			names = append(names, name)
		}
	}
	return names
}

func definitionAttemptsUnpack(def *pb.Definition) bool {
	if def == nil {
		return false
	}
	for _, dt := range def.Def {
		var op pb.Op
		if err := op.Unmarshal(dt); err != nil {
			continue
		}
		file := op.GetFile()
		if file == nil {
			continue
		}
		for _, action := range file.Actions {
			if copyAction := action.GetCopy(); copyAction != nil && copyAction.AttemptUnpackDockerCompatibility {
				return true
			}
		}
	}
	return false
}

func hasSourcePrefix(sources []string, prefix string) bool {
	for _, source := range sources {
		if strings.HasPrefix(source, prefix) {
			return true
		}
	}
	return false
}

func fileBackedAgentkitfile(path string) []byte {
	return []byte(fmt.Sprintf(`apiVersion: v1alpha1
kind: Agent
metadata:
  name: reliability-test
model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
  name: gpt-4o-mini
instructions:
  file: %s
expose:
  openai: true
`, path))
}

func inlineAgentkitfile() []byte {
	return []byte(`apiVersion: v1alpha1
kind: Agent
metadata:
  name: reliability-test
model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
  name: gpt-4o-mini
instructions: Be reliable.
expose:
  openai: true
`)
}

func inlineAgentkitfileWithRuntime(runtime string) []byte {
	return []byte(strings.Replace(
		string(inlineAgentkitfile()),
		"kind: Agent\n",
		"kind: Agent\nruntime: "+runtime+"\n",
		1,
	))
}

func TestBuildRejectsTargetRuntimeMismatchBeforeImageSolve(t *testing.T) {
	tests := []struct {
		name              string
		agentkitfile      []byte
		target            string
		buildArgValue     string
		wantTargetRuntime string
	}{
		{
			name:              "config runtime",
			agentkitfile:      inlineAgentkitfileWithRuntime(runtimePydca),
			target:            wantLangGraphRoute,
			wantTargetRuntime: runtimeLangGraph,
		},
		{
			name:              "build arg overrides config runtime",
			agentkitfile:      inlineAgentkitfileWithRuntime(runtimeLangGraph),
			target:            wantLangGraphRoute,
			buildArgValue:     runtimePydca,
			wantTargetRuntime: runtimeLangGraph,
		},
		{
			name:              "default runtime rejects alias-prefixed route",
			agentkitfile:      inlineAgentkitfile(),
			target:            runtimeMAFAls + "/image/debug",
			wantTargetRuntime: runtimeMAFName,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			configRef := &memoryReference{files: map[string][]byte{defaultAgentkitfileName: tt.agentkitfile}}
			opts := map[string]string{keyTarget: tt.target}
			if tt.buildArgValue != "" {
				opts["build-arg:runtime"] = tt.buildArgValue
			}
			c := &fakeBuildClient{
				opts: client.BuildOpts{Opts: opts},
				expected: []expectedSolve{
					{sourcePrefix: localDockerfileSourcePrefix, ref: configRef},
				},
			}

			_, err := Build(context.Background(), c)
			if err == nil {
				t.Fatal("Build() error = nil, want runtime mismatch")
			}
			for _, want := range []string{
				"no route",
				fmt.Sprintf("target runtime %q", tt.wantTargetRuntime),
				`effective runtime "pydantic-ai"`,
			} {
				if !strings.Contains(err.Error(), want) {
					t.Fatalf("Build() error = %q, want substring %q", err, want)
				}
			}
			if got := len(c.solveSources()); got != 1 {
				t.Fatalf("Build() solve count = %d, want 1 config solve and no image solve", got)
			}
		})
	}
}

func tarContext(t *testing.T, files map[string][]byte) []byte {
	t.Helper()
	var buf bytes.Buffer
	tw := tar.NewWriter(&buf)
	for name, data := range files {
		if err := tw.WriteHeader(&tar.Header{Name: name, Mode: 0o644, Size: int64(len(data))}); err != nil {
			t.Fatalf("write tar header: %v", err)
		}
		if _, err := tw.Write(data); err != nil {
			t.Fatalf("write tar file: %v", err)
		}
	}
	if err := tw.Close(); err != nil {
		t.Fatalf("close tar: %v", err)
	}
	return buf.Bytes()
}

type redactionTestMaterial struct {
	username      string
	userInfo      string
	queryValue    string
	fragmentValue string
}

func newRedactionTestMaterial() redactionTestMaterial {
	return redactionTestMaterial{
		username:      strings.Join([]string{redactionTestPrefix, "user"}, "-"),
		userInfo:      strings.Join([]string{redactionTestPrefix, "userinfo"}, "-"),
		queryValue:    strings.Join([]string{redactionTestPrefix, "query"}, "-"),
		fragmentValue: strings.Join([]string{redactionTestPrefix, "fragment"}, "-"),
	}
}

func redactionTestURL(t *testing.T, baseURL, queryKey string, includeFragment bool) string {
	t.Helper()
	parsed, err := url.Parse(baseURL)
	if err != nil {
		t.Fatalf("parse redaction test URL: %v", err)
	}
	material := newRedactionTestMaterial()
	parsed.User = url.UserPassword(material.username, material.userInfo)
	if queryKey != "" {
		query := parsed.Query()
		query.Set(queryKey, material.queryValue)
		parsed.RawQuery = query.Encode()
	}
	if includeFragment {
		parsed.Fragment = material.fragmentValue
	}
	return parsed.String()
}

func TestBuildGitContextInstructionsFileUsesSameRemoteContext(t *testing.T) {
	remote := &memoryReference{files: map[string][]byte{
		defaultAgentkitfileName: fileBackedAgentkitfile(filePath),
		filePath:                []byte(filePrompt),
	}}
	final := &memoryReference{}
	c := &fakeBuildClient{
		opts: client.BuildOpts{Opts: map[string]string{localNameContext: remoteContextURL}},
		expected: []expectedSolve{
			{sourcePrefix: "git://example.com/agent.git", ref: remote},
			{sourcePrefix: dockerImageSourcePrefix, ref: final},
		},
	}

	if _, err := Build(context.Background(), c); err != nil {
		t.Fatalf("Build() error = %v", err)
	}
	if got := remote.readPaths(); !containsString(got, filePath) {
		t.Fatalf("remote context reads = %v, want instructions path %q", got, filePath)
	}
	for _, solveSources := range c.solveSources() {
		if hasSourcePrefix(solveSources, "local://context") {
			t.Fatalf("remote build unexpectedly created local context source: %v", solveSources)
		}
	}
}

func TestBuildHTTPArchiveInstructionsFileUsesSameRemoteContext(t *testing.T) {
	archive := tarContext(t, map[string][]byte{
		defaultAgentkitfileName: fileBackedAgentkitfile(filePath),
		filePath:                []byte(filePrompt),
	})
	rawHTTP := &memoryReference{files: map[string][]byte{downloadedHTTPContextFilename: archive}}
	resolvedHTTP := &memoryReference{files: map[string][]byte{
		defaultAgentkitfileName: fileBackedAgentkitfile(filePath),
		filePath:                []byte(filePrompt),
	}}
	final := &memoryReference{}
	contextURL := "https://example.com/agent-context.tar"
	c := &fakeBuildClient{
		opts: client.BuildOpts{Opts: map[string]string{localNameContext: contextURL}},
		expected: []expectedSolve{
			{sourcePrefix: contextURL, ref: rawHTTP},
			{sourcePrefix: contextURL, requireAttemptUnpack: true, ref: resolvedHTTP},
			{sourcePrefix: dockerImageSourcePrefix, ref: final},
		},
	}

	if _, err := Build(context.Background(), c); err != nil {
		t.Fatalf("Build() error = %v", err)
	}
	if got := resolvedHTTP.readPaths(); !containsString(got, filePath) {
		t.Fatalf("resolved HTTP context reads = %v, want instructions path %q", got, filePath)
	}
	for _, solveSources := range c.solveSources() {
		if hasSourcePrefix(solveSources, "local://context") {
			t.Fatalf("remote build unexpectedly created local context source: %v", solveSources)
		}
	}
}

func TestBuildHTTPArchiveRedactsContextURLFromOperationNames(t *testing.T) {
	contextURL := redactionTestURL(t, "https://example.com/contexts/agent.tar", "sig", true)
	archive := tarContext(t, map[string][]byte{
		defaultAgentkitfileName: fileBackedAgentkitfile(filePath),
		filePath:                []byte(filePrompt),
	})
	rawHTTP := &memoryReference{files: map[string][]byte{downloadedHTTPContextFilename: archive}}
	resolvedHTTP := &memoryReference{files: map[string][]byte{
		defaultAgentkitfileName: fileBackedAgentkitfile(filePath),
		filePath:                []byte(filePrompt),
	}}
	final := &memoryReference{}
	c := &fakeBuildClient{
		opts: client.BuildOpts{Opts: map[string]string{localNameContext: contextURL}},
		expected: []expectedSolve{
			{sourcePrefix: contextURL, ref: rawHTTP},
			{sourcePrefix: contextURL, requireAttemptUnpack: true, ref: resolvedHTTP},
			{sourcePrefix: dockerImageSourcePrefix, ref: final},
		},
	}

	if _, err := Build(context.Background(), c); err != nil {
		t.Fatalf("Build() error = %v", err)
	}
	assertSafeContextText(t, strings.Join(c.allOperationNames(), "\n"), "example.com/contexts/agent.tar")
	if sources := fmt.Sprint(c.solveSources()); !strings.Contains(sources, contextURL) {
		t.Fatalf("source resolution = %s, want raw context URL preserved", sources)
	}
}

func TestBuildGitContextRedactsContextURLFromOperationNames(t *testing.T) {
	contextURL := redactionTestURL(t, "https://example.com/repos/agent.git", "ref", false)
	remote := &memoryReference{files: map[string][]byte{
		defaultAgentkitfileName: fileBackedAgentkitfile(filePath),
		filePath:                []byte(filePrompt),
	}}
	final := &memoryReference{}
	c := &fakeBuildClient{
		opts: client.BuildOpts{Opts: map[string]string{localNameContext: contextURL}},
		expected: []expectedSolve{
			{sourcePrefix: "git://example.com/repos/agent.git#" + newRedactionTestMaterial().queryValue, ref: remote},
			{sourcePrefix: dockerImageSourcePrefix, ref: final},
		},
	}

	if _, err := Build(context.Background(), c); err != nil {
		t.Fatalf("Build() error = %v", err)
	}
	assertSafeContextText(t, strings.Join(c.allOperationNames(), "\n"), "example.com/repos/agent.git")
}

func TestBuildHTTPContextRedactsContextURLFromErrors(t *testing.T) {
	contextURL := redactionTestURL(t, "https://example.com/contexts/agent.yaml", "sig", true)
	c := &fakeBuildClient{
		opts: client.BuildOpts{Opts: map[string]string{localNameContext: contextURL}},
		expected: []expectedSolve{
			{
				sourcePrefix: contextURL,
				err:          fmt.Errorf("backend failed to fetch %s", contextURL),
			},
		},
	}

	_, err := Build(context.Background(), c)
	if err == nil {
		t.Fatal("Build() error = nil, want context resolution error")
	}
	assertSafeContextText(t, err.Error(), "example.com/contexts/agent.yaml")
}

func assertSafeContextText(t *testing.T, text, usefulContext string) {
	t.Helper()
	if !strings.Contains(text, usefulContext) {
		t.Fatalf("text = %q, want useful context %q", text, usefulContext)
	}
	material := newRedactionTestMaterial()
	for _, forbidden := range []string{
		material.username,
		material.userInfo,
		"?sig=",
		"?ref=",
		material.queryValue,
		"#" + material.fragmentValue,
		material.fragmentValue,
	} {
		if strings.Contains(text, forbidden) {
			t.Fatalf("text = %q, leaked %q", text, forbidden)
		}
	}
}

func TestBuildHostlessHTTPContextRedactsQueryAndFragmentFromErrors(t *testing.T) {
	material := newRedactionTestMaterial()
	query := url.Values{"sig": []string{material.queryValue}}
	contextURL := "https://?" + query.Encode() + "#" + url.PathEscape(material.fragmentValue)
	c := &fakeBuildClient{opts: client.BuildOpts{Opts: map[string]string{localNameContext: contextURL}}}

	_, err := Build(context.Background(), c)
	if err == nil {
		t.Fatal("Build() error = nil, want invalid HTTP context error")
	}
	assertSafeContextText(t, err.Error(), "HTTP(S) remote context")
}

func TestBuildNetworkPathContextRedactsCredentialsFromErrors(t *testing.T) {
	contextName := redactionTestURL(t, "//example.com/context", "sig", true)
	c := &fakeBuildClient{opts: client.BuildOpts{Opts: map[string]string{localNameContext: contextName}}}

	_, err := Build(context.Background(), c)
	if err == nil {
		t.Fatal("Build() error = nil, want unsupported context error")
	}
	assertSafeContextText(t, err.Error(), "example.com/context")
}

func TestBuildMalformedContextUsesGenericDisplayInsteadOfRawCredentials(t *testing.T) {
	material := newRedactionTestMaterial()
	for _, contextName := range []string{
		fmt.Sprintf("///%s:%s@example.com/context", material.username, material.userInfo),
		fmt.Sprintf("%s:%s@example.com/context", material.username, material.userInfo),
	} {
		t.Run(contextName, func(t *testing.T) {
			c := &fakeBuildClient{opts: client.BuildOpts{Opts: map[string]string{localNameContext: contextName}}}

			_, err := Build(context.Background(), c)
			if err == nil {
				t.Fatal("Build() error = nil, want unsupported context error")
			}
			assertSafeContextText(t, err.Error(), "remote context")
		})
	}
}

func TestBuildRemoteSolveErrorPreservesCancellationWithoutURLLeak(t *testing.T) {
	contextURL := redactionTestURL(t, "https://example.com/contexts/agent.yaml", "sig", true)
	c := &fakeBuildClient{
		opts: client.BuildOpts{Opts: map[string]string{localNameContext: contextURL}},
		expected: []expectedSolve{
			{sourcePrefix: contextURL, err: context.Canceled},
		},
	}

	_, err := Build(context.Background(), c)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Build() error = %v, want context.Canceled", err)
	}
	assertSafeContextText(t, err.Error(), "example.com/contexts/agent.yaml")
}

func TestBuildRemoteSolveErrorPreservesGRPCStatusWithoutURLLeak(t *testing.T) {
	contextURL := redactionTestURL(t, "https://example.com/contexts/agent.yaml", "sig", true)
	c := &fakeBuildClient{
		opts: client.BuildOpts{Opts: map[string]string{localNameContext: contextURL}},
		expected: []expectedSolve{
			{
				sourcePrefix: contextURL,
				err:          status.Error(codes.Unauthenticated, "backend rejected "+contextURL),
			},
		},
	}

	_, err := Build(context.Background(), c)
	if got := status.Code(err); got != codes.Unauthenticated {
		t.Fatalf("Build() status code = %s, want %s (error: %v)", got, codes.Unauthenticated, err)
	}
	assertSafeContextText(t, err.Error(), "example.com/contexts/agent.yaml")
}

func TestBuildRemoteReadErrorPreservesDeadlineWithoutURLLeak(t *testing.T) {
	contextURL := redactionTestURL(t, "https://example.com/repos/agent.git", "ref", false)
	remote := &memoryReference{readErr: context.DeadlineExceeded}
	c := &fakeBuildClient{
		opts: client.BuildOpts{Opts: map[string]string{localNameContext: contextURL}},
		expected: []expectedSolve{
			{sourcePrefix: "git://example.com/repos/agent.git#" + newRedactionTestMaterial().queryValue, ref: remote},
		},
	}

	_, err := Build(context.Background(), c)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("Build() error = %v, want context.DeadlineExceeded", err)
	}
	assertSafeContextText(t, err.Error(), "example.com/repos/agent.git")
}

func TestBuildLocalContextInstructionsFileUsesLocalContext(t *testing.T) {
	configRef := &memoryReference{files: map[string][]byte{
		defaultAgentkitfileName: fileBackedAgentkitfile(filePath),
	}}
	contextRef := &memoryReference{files: map[string][]byte{filePath: []byte(filePrompt)}}
	final := &memoryReference{}
	c := &fakeBuildClient{
		opts: client.BuildOpts{Opts: map[string]string{}},
		expected: []expectedSolve{
			{sourcePrefix: localDockerfileSourcePrefix, ref: configRef},
			{sourcePrefix: "local://context", ref: contextRef},
			{sourcePrefix: dockerImageSourcePrefix, ref: final},
		},
	}

	if _, err := Build(context.Background(), c); err != nil {
		t.Fatalf("Build() error = %v", err)
	}
	if got := contextRef.readPaths(); !containsString(got, filePath) {
		t.Fatalf("local context reads = %v, want instructions path %q", got, filePath)
	}
}

func TestBuildSingleFileHTTPContextRejectsInstructionsFileClearly(t *testing.T) {
	contextURL := "https://example.com/agentkitfile.yaml"
	rawHTTP := &memoryReference{files: map[string][]byte{downloadedHTTPContextFilename: fileBackedAgentkitfile(filePath)}}
	c := &fakeBuildClient{
		opts: client.BuildOpts{Opts: map[string]string{localNameContext: contextURL}},
		expected: []expectedSolve{
			{sourcePrefix: contextURL, ref: rawHTTP},
		},
	}

	_, err := Build(context.Background(), c)
	if err == nil {
		t.Fatal("Build() error = nil, want unsupported instructions.file error")
	}
	for _, want := range []string{contextURL, "single-file HTTP context", "instructions.file"} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("Build() error = %q, want substring %q", err, want)
		}
	}
}

func TestBuildUnsupportedRemoteContextReturnsError(t *testing.T) {
	contextName := "ftp://example.com/agent-context.tar"
	c := &fakeBuildClient{opts: client.BuildOpts{Opts: map[string]string{localNameContext: contextName}}}

	_, err := Build(context.Background(), c)
	if err == nil {
		t.Fatal("Build() error = nil, want unsupported context error")
	}
	for _, want := range []string{contextName, "unsupported build context", "Git", "HTTP"} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("Build() error = %q, want substring %q", err, want)
		}
	}
}

func TestBuildMalformedGitContextReturnsError(t *testing.T) {
	contextName := redactionTestURL(t, "https://example.com/agent.git", "ref", true)
	c := &fakeBuildClient{opts: client.BuildOpts{Opts: map[string]string{localNameContext: contextName}}}

	_, err := Build(context.Background(), c)
	if err == nil {
		t.Fatal("Build() error = nil, want malformed context error")
	}
	if !strings.Contains(err.Error(), "ref conflicts") {
		t.Fatalf("Build() error = %q, want ref-conflict context", err)
	}
	assertSafeContextText(t, err.Error(), "example.com/agent.git")
}

func TestBuildNullCacheImportReturnsError(t *testing.T) {
	configRef := &memoryReference{files: map[string][]byte{defaultAgentkitfileName: inlineAgentkitfile()}}
	c := &fakeBuildClient{
		opts: client.BuildOpts{Opts: map[string]string{keyCacheImports: "[null]"}},
		expected: []expectedSolve{
			{sourcePrefix: localDockerfileSourcePrefix, ref: configRef},
		},
	}

	_, err := Build(context.Background(), c)
	if err == nil {
		t.Fatal("Build() error = nil, want invalid cache import error")
	}
	for _, want := range []string{keyCacheImports, "entry 0", "null"} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("Build() error = %q, want substring %q", err, want)
		}
	}
}

func containsString(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}
