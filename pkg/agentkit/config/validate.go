package config

import (
	"errors"
	"fmt"
	"sort"
	"strings"

	"github.com/sozercan/agentkit/pkg/agentkit/runtimes"
	"github.com/sozercan/agentkit/pkg/utils"
)

const (
	ContextTypeSearch       = "search"
	ContextTypeSkills       = "skills"
	ContextTypeMemory       = "memory"
	ContextSourceFilesystem = "filesystem"
	ContextSourceMCP        = "mcp"

	ToolTypeMCP                 = "mcp"
	ToolTransportStdio          = "stdio"
	ToolTransportStreamableHTTP = "streamable-http"
	AuthTypeBearer              = "bearer"
	AuthTypeWorkloadIdentity    = "workload-identity-token"
	ApprovalNever               = "never"
	ApprovalAuto                = "auto"
	ApprovalAlways              = "always"
)

// Validate reports every problem with the config at once via errors.Join (plan
// §16.2 #3 — one report-all validator, not scattered first-error-wins funcs).
//
// It covers both schema validity and the v0 deterministic security gates that
// run BEFORE any LLB (plan §10): no secret literals in YAML, exposure sanity.
// Behavioral evals are out of v0 scope; these static gates are the v0 safety net.
func (c *AgentConfig) Validate() error {
	var errs []error
	add := func(format string, args ...any) {
		errs = append(errs, fmt.Errorf(format, args...))
	}

	// --- apiVersion / kind -------------------------------------------------
	if c.APIVersion == "" {
		add("apiVersion is not defined")
	} else if c.APIVersion != utils.APIv1alpha1 {
		add("apiVersion %q is not supported (expected %q)", c.APIVersion, utils.APIv1alpha1)
	}
	if c.Kind != utils.KindAgent {
		add("kind %q is not supported (expected %q)", c.Kind, utils.KindAgent)
	}

	// --- metadata ----------------------------------------------------------
	if c.Metadata.Name == "" {
		add("metadata.name is required")
	}

	// --- runtime -----------------------------------------------------------
	runtimeName := c.Runtime
	if runtimeName == "" {
		runtimeName = runtimes.DefaultRuntime()
	}
	runtimeSpec, runtimeKnown := runtimes.RuntimeByName(runtimeName)
	if c.Runtime != "" && !runtimeKnown {
		supported := runtimes.KnownRuntimes()
		sort.Strings(supported)
		add("runtime %q is not supported (supported: %s)", c.Runtime, strings.Join(supported, ", "))
	}

	// --- model -------------------------------------------------------------
	if c.Model.Provider == "" {
		add("model.provider is required")
	} else if c.Model.Provider != utils.ProviderOpenAICompatible {
		add("model.provider %q is not supported in v0 (only %q); local models compose with a co-located AIKit container over baseURL, never baked", c.Model.Provider, utils.ProviderOpenAICompatible)
	}
	if c.Model.BaseURL == "" {
		add("model.baseURL is required")
	}
	if c.Model.Name == "" {
		add("model.name is required")
	}
	// noSecretsInImage gate (plan §10): apiKeyEnv must be a NAME, not a literal.
	if c.Model.APIKeyEnv != "" && (!isEnvVarName(c.Model.APIKeyEnv) || looksLikeSecretLiteral(c.Model.APIKeyEnv)) {
		add("model.apiKeyEnv %q must be an env var NAME matching [A-Z0-9_]+; provide the NAME of an env var (e.g. OPENAI_API_KEY) and inject the value with `docker run -e`", c.Model.APIKeyEnv)
	}
	validateAuth(add, "model.auth", c.Model.Auth)
	if c.Model.Auth != nil && c.Model.Auth.Type != AuthTypeWorkloadIdentity {
		add("model.auth.type %q is not supported in v0; use apiKeyEnv or workload-identity-token gated by runtime capability", c.Model.Auth.Type)
	}

	// --- instructions ------------------------------------------------------
	if c.Instructions.IsZero() {
		add("instructions are required (a string, or a {file: ...} source)")
	} else if set := c.Instructions.variantsSet(); len(set) > 1 {
		add("instructions set multiple sources %v; exactly one of inline|file is allowed", set)
	}

	// --- tools -------------------------------------------------------------
	seen := map[string]bool{}
	for i, t := range c.Tools {
		if t.Name == "" {
			add("tools[%d].name is required", i)
		} else if seen[t.Name] {
			add("tools[%d]: duplicate tool name %q", i, t.Name)
		}
		seen[t.Name] = true

		if t.Type != "" && t.Type != ToolTypeMCP {
			add("tools[%d] (%s): type %q is not supported (expected %q)", i, t.Name, t.Type, ToolTypeMCP)
		}

		set := t.variantsSet()
		switch len(set) {
		case 0:
			add("tools[%d] (%s): a tool source is required (command or urlEnv)", i, t.Name)
		case 1:
			if t.Image != "" {
				add("tools[%d] (%s): image-based MCP servers are not supported in v0 (use command or streamable-http urlEnv; arbitrary-OCI staging is v1)", i, t.Name)
			}
		default:
			add("tools[%d] (%s): sets multiple sources %v; exactly one is allowed", i, t.Name, set)
		}

		validateToolTransport(add, i, t)
		validateToolEnv(add, i, t)
		validateToolHeaders(add, i, t)
		validateAuth(add, fmt.Sprintf("tools[%d] (%s).auth", i, t.Name), t.Auth)
		validateApproval(add, i, t)
	}

	// --- env requirements ---------------------------------------------------
	seenEnv := map[string]bool{}
	for i, e := range c.Env {
		if e.Name == "" {
			add("env[%d].name is required; list env var NAMES only", i)
			continue
		}
		if !isEnvVarName(e.Name) || looksLikeSecretLiteral(e.Name) {
			add("env[%d].name %q must be an env var NAME matching [A-Z0-9_]+; list names only, never values", i, e.Name)
		}
		if seenEnv[e.Name] {
			add("env[%d]: duplicate env var name %q", i, e.Name)
		}
		seenEnv[e.Name] = true
	}

	validateContext(add, c.Context)
	validateObservability(add, c.Observability)

	// --- runtime capability gate -------------------------------------------
	if runtimeKnown {
		missing := runtimeSpec.MissingCapabilities(c.requiredCapabilities())
		if len(missing) > 0 {
			add("runtime %q does not support requested capabilities: %s", runtimeSpec.Name, strings.Join(missing, ", "))
		}
	}

	// --- expose ------------------------------------------------------------
	if !c.Expose.OpenAI {
		add("expose.openai must be true in v0 (the OpenAI /v1 façade is the only serving surface)")
	}
	if c.Expose.Port < 0 || c.Expose.Port > 65535 {
		add("expose.port %d is out of range", c.Expose.Port)
	}

	return errors.Join(errs...)
}

func validateContext(add func(string, ...any), ctx Context) {
	seen := map[string]bool{}
	for i, provider := range ctx.Providers {
		path := fmt.Sprintf("context.providers[%d]", i)
		if provider.Name != "" {
			if seen[provider.Name] {
				add("%s: duplicate provider name %q", path, provider.Name)
			}
			seen[provider.Name] = true
		}
		switch provider.Type {
		case ContextTypeSearch:
			validateEnvField(add, path+".endpointEnv", provider.EndpointEnv, true)
			validateEnvField(add, path+".indexEnv", provider.IndexEnv, true)
			validateAuth(add, path+".auth", provider.Auth)
		case ContextTypeSkills:
			switch provider.Source {
			case ContextSourceFilesystem:
				if provider.Path == "" {
					add("%s.path is required for filesystem skills", path)
				}
			case ContextSourceMCP:
				if provider.ToolRef == "" {
					add("%s.toolRef is required for MCP skills", path)
				}
				if provider.Index == "" {
					add("%s.index is required for MCP skills", path)
				}
			default:
				add("%s.source %q is not supported for skills (expected filesystem or mcp)", path, provider.Source)
			}
		case ContextTypeMemory:
			validateEnvField(add, path+".storeNameEnv", provider.StoreNameEnv, true)
			validateEnvField(add, path+".endpointEnv", provider.EndpointEnv, true)
			validateAuth(add, path+".auth", provider.Auth)
		case "":
			add("%s.type is required", path)
		default:
			add("%s.type %q is not supported (expected search, skills, or memory)", path, provider.Type)
		}
	}
}

func validateObservability(add func(string, ...any), obs Observability) {
	validateEnvField(add, "observability.otel.endpointEnv", obs.OTel.EndpointEnv, false)
	validateEnvField(add, "observability.logs.levelEnv", obs.Logs.LevelEnv, false)
}

func validateEnvField(add func(string, ...any), path, value string, required bool) {
	if value == "" {
		if required {
			add("%s is required", path)
		}
		return
	}
	if !isEnvVarName(value) || looksLikeSecretLiteral(value) {
		add("%s %q must be an env var NAME matching [A-Z0-9_]+", path, value)
	}
}

func validateToolTransport(add func(string, ...any), i int, t Tool) {
	if len(t.Command) > 0 {
		if t.Transport != "" && t.Transport != ToolTransportStdio {
			add("tools[%d] (%s): command tools use transport %q or omit transport", i, t.Name, ToolTransportStdio)
		}
		if t.URLEnv != "" || len(t.Headers) > 0 || t.Auth != nil {
			add("tools[%d] (%s): stdio command tools must not set urlEnv, headers, or auth", i, t.Name)
		}
	} else if t.URLEnv != "" {
		if t.Type != ToolTypeMCP {
			add("tools[%d] (%s): remote MCP tools must set type: %s", i, t.Name, ToolTypeMCP)
		}
		if t.Transport != ToolTransportStreamableHTTP {
			add("tools[%d] (%s): remote MCP tools must set transport: %s", i, t.Name, ToolTransportStreamableHTTP)
		}
		if !isEnvVarName(t.URLEnv) || looksLikeSecretLiteral(t.URLEnv) {
			add("tools[%d] (%s): urlEnv %q must be an env var NAME matching [A-Z0-9_]+", i, t.Name, t.URLEnv)
		}
		if len(t.Env) > 0 {
			add("tools[%d] (%s): remote MCP tools must use headers/auth instead of stdio env", i, t.Name)
		}
	} else if t.Transport != "" && t.Transport != ToolTransportStdio {
		add("tools[%d] (%s): transport %q requires urlEnv", i, t.Name, t.Transport)
	}
}

func validateToolEnv(add func(string, ...any), i int, t Tool) {
	for j, part := range t.Command {
		if part == "" {
			add("tools[%d] (%s): command[%d] must be non-empty", i, t.Name, j)
		}
	}
	for _, e := range t.Env {
		if e == "" {
			add("tools[%d] (%s): env entry is empty; list env var NAMES only", i, t.Name)
		} else if !isEnvVarName(e) || looksLikeSecretLiteral(e) {
			add("tools[%d] (%s): env entry %q must be an env var NAME matching [A-Z0-9_]+; list names only, never values", i, t.Name, e)
		}
	}
}

func validateToolHeaders(add func(string, ...any), i int, t Tool) {
	seen := map[string]bool{}
	for j, h := range t.Headers {
		path := fmt.Sprintf("tools[%d] (%s).headers[%d]", i, t.Name, j)
		if h.Name == "" {
			add("%s.name is required", path)
		} else if !isHTTPHeaderName(h.Name) {
			add("%s.name %q is not a valid HTTP header name", path, h.Name)
		} else if seen[strings.ToLower(h.Name)] {
			add("%s: duplicate header name %q", path, h.Name)
		}
		seen[strings.ToLower(h.Name)] = true

		values := 0
		if h.Value != "" {
			values++
		}
		if h.ValueEnv != "" {
			values++
		}
		if values != 1 {
			add("%s must set exactly one of value or valueEnv", path)
		}
		if t.Auth != nil && strings.EqualFold(h.Name, "authorization") {
			add("%s must not set Authorization when auth is also configured; use one auth path", path)
		}
		if h.Value != "" && isCredentialHeaderName(h.Name) {
			add("%s.value must not bake a static credential header; use valueEnv or auth", path)
		}
		if h.Value != "" && hasSecretPrefix(h.Value) {
			add("%s.value looks like a secret value; use valueEnv instead", path)
		}
		if h.ValueEnv != "" && (!isEnvVarName(h.ValueEnv) || looksLikeSecretLiteral(h.ValueEnv)) {
			add("%s.valueEnv %q must be an env var NAME matching [A-Z0-9_]+", path, h.ValueEnv)
		}
	}
}

func validateAuth(add func(string, ...any), path string, auth *Auth) {
	if auth == nil {
		return
	}
	switch auth.Type {
	case AuthTypeBearer:
		if auth.TokenEnv == "" {
			add("%s.tokenEnv is required for bearer auth", path)
		} else if !isEnvVarName(auth.TokenEnv) || looksLikeSecretLiteral(auth.TokenEnv) {
			add("%s.tokenEnv %q must be an env var NAME matching [A-Z0-9_]+", path, auth.TokenEnv)
		}
		if auth.Audience != "" {
			add("%s.audience must be empty for bearer auth", path)
		}
	case AuthTypeWorkloadIdentity:
		if auth.Audience == "" {
			add("%s.audience is required for workload identity token auth", path)
		}
		if auth.TokenEnv != "" {
			add("%s.tokenEnv must be empty for workload identity token auth", path)
		}
	default:
		add("%s.type %q is not supported (expected %q or %q)", path, auth.Type, AuthTypeBearer, AuthTypeWorkloadIdentity)
	}
}

func validateApproval(add func(string, ...any), i int, t Tool) {
	switch t.Approval {
	case "", ApprovalNever, ApprovalAuto, ApprovalAlways:
	default:
		add("tools[%d] (%s): approval %q is not supported (expected never, auto, or always)", i, t.Name, t.Approval)
	}
}

func (c *AgentConfig) requiredCapabilities() []string {
	seen := map[string]bool{}
	var out []string
	add := func(cap string) {
		if !seen[cap] {
			seen[cap] = true
			out = append(out, cap)
		}
	}
	if c.Model.Auth != nil && c.Model.Auth.Type == AuthTypeWorkloadIdentity {
		add(runtimes.CapabilityModelWorkloadIdentityAuth)
	}
	for _, t := range c.Tools {
		if len(t.Command) > 0 {
			add(runtimes.CapabilityStdioMCP)
		}
		if t.URLEnv != "" || t.Transport == ToolTransportStreamableHTTP {
			add(runtimes.CapabilityStreamableHTTPMCP)
		}
		if t.Auth != nil && t.Auth.Type == AuthTypeWorkloadIdentity {
			add(runtimes.CapabilityWorkloadIdentityTokenAuth)
		}
		if t.Approval == ApprovalAuto || t.Approval == ApprovalAlways {
			add(runtimes.CapabilityToolApproval)
		}
	}
	for _, provider := range c.Context.Providers {
		switch provider.Type {
		case ContextTypeSearch:
			add(runtimes.CapabilityContextProviderSearch)
		case ContextTypeSkills:
			add(runtimes.CapabilityContextProviderSkills)
			if provider.Source == ContextSourceFilesystem {
				add(runtimes.CapabilityFilesystemSkills)
			}
			if provider.Source == ContextSourceMCP {
				add(runtimes.CapabilityMCPSkills)
			}
		case ContextTypeMemory:
			add(runtimes.CapabilityContextProviderMemory)
		}
		if provider.Auth != nil && provider.Auth.Type == AuthTypeWorkloadIdentity {
			add(runtimes.CapabilityWorkloadIdentityTokenAuth)
		}
	}
	if c.Observability.OTel.EndpointEnv != "" {
		add(runtimes.CapabilityOTelExport)
	}
	return out
}

func isEnvVarName(v string) bool {
	if v == "" {
		return false
	}
	for _, r := range v {
		isUpper := r >= 'A' && r <= 'Z'
		isDigit := r >= '0' && r <= '9'
		if !isUpper && !isDigit && r != '_' {
			return false
		}
	}
	return true
}

func isHTTPHeaderName(v string) bool {
	if v == "" {
		return false
	}
	for _, r := range v {
		isAlpha := (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z')
		isDigit := r >= '0' && r <= '9'
		switch {
		case isAlpha || isDigit:
			continue
		case strings.ContainsRune("!#$%&'*+-.^_`|~", r):
			continue
		default:
			return false
		}
	}
	return true
}

func isCredentialHeaderName(name string) bool {
	switch strings.ToLower(name) {
	case "authorization", "proxy-authorization", "cookie", "set-cookie", "x-api-key", "api-key", "ocp-apim-subscription-key", "subscription-key", "x-functions-key":
		return true
	default:
		return false
	}
}

func hasSecretPrefix(v string) bool {
	for _, p := range []string{"sk-", "sk_", "ghp_", "github_pat_", "xoxb-", "AKIA"} {
		if strings.HasPrefix(v, p) {
			return true
		}
	}
	return false
}

// looksLikeSecretLiteral heuristically flags a value that appears to be a secret
// rather than an env var NAME. Env var names are uppercase letters, digits, and
// underscores; common secret prefixes (sk-, etc.) and lowercase/punctuation are
// strong signals the user pasted a value where a name belongs.
func looksLikeSecretLiteral(v string) bool {
	if v == "" {
		return false
	}
	// Known secret-value prefixes.
	if hasSecretPrefix(v) {
		return true
	}
	// Env var NAMEs are [A-Z0-9_]+; lowercase/spaces/URLs/punctuation are a strong
	// signal the user pasted a value into an env-name field.
	if isEnvVarName(v) {
		return false
	}
	for _, r := range v {
		isUpper := r >= 'A' && r <= 'Z'
		isDigit := r >= '0' && r <= '9'
		if !isUpper && !isDigit && r != '_' {
			return true
		}
	}
	return false
}
