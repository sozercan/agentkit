package config

import (
	"errors"
	"fmt"
	"strings"

	"github.com/sozercan/agentkit/pkg/utils"
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

	// --- runtime (v0: pydantic-ai only; empty defaults to it) --------------
	if c.Runtime != "" && c.Runtime != utils.RuntimePydanticAI {
		add("runtime %q is not supported in v0 (only %q)", c.Runtime, utils.RuntimePydanticAI)
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
	if looksLikeSecretLiteral(c.Model.APIKeyEnv) {
		add("model.apiKeyEnv %q looks like a secret value, not an env var NAME; provide the NAME of an env var (e.g. OPENAI_API_KEY) and inject the value with `docker run -e`", c.Model.APIKeyEnv)
	}

	// --- instructions ------------------------------------------------------
	if c.Instructions.IsZero() {
		add("instructions are required (a string, or a {file: ...} source)")
	} else if set := c.Instructions.variantsSet(); len(set) > 1 {
		add("instructions set multiple sources %v; exactly one of inline|file is allowed", set)
	}

	// --- tools (v0: stdio command MCP servers only) ------------------------
	seen := map[string]bool{}
	for i, t := range c.Tools {
		if t.Name == "" {
			add("tools[%d].name is required", i)
		} else if seen[t.Name] {
			add("tools[%d]: duplicate tool name %q", i, t.Name)
		}
		seen[t.Name] = true

		set := t.variantsSet()
		switch len(set) {
		case 0:
			add("tools[%d] (%s): a tool source is required (v0: command)", i, t.Name)
		case 1:
			if t.Image != "" {
				add("tools[%d] (%s): image-based MCP servers are not supported in v0 (use a stdio command; arbitrary-OCI staging is v1)", i, t.Name)
			}
		default:
			add("tools[%d] (%s): sets multiple sources %v; exactly one is allowed", i, t.Name, set)
		}

		for _, e := range t.Env {
			if looksLikeSecretLiteral(e) {
				add("tools[%d] (%s): env entry %q looks like a secret value; list env var NAMES only", i, t.Name, e)
			}
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

// looksLikeSecretLiteral heuristically flags a value that appears to be a secret
// rather than an env var NAME. Env var names are uppercase letters, digits, and
// underscores; common secret prefixes (sk-, etc.) and lowercase/punctuation are
// strong signals the user pasted a value where a name belongs.
func looksLikeSecretLiteral(v string) bool {
	if v == "" {
		return false
	}
	// Known secret-value prefixes.
	for _, p := range []string{"sk-", "sk_", "ghp_", "github_pat_", "xoxb-", "AKIA"} {
		if strings.HasPrefix(v, p) {
			return true
		}
	}
	// An env var NAME is [A-Z0-9_]+; anything else (lowercase, spaces, ://,
	// punctuation) is not a valid name and is treated as a misplaced value.
	for _, r := range v {
		isUpper := r >= 'A' && r <= 'Z'
		isDigit := r >= '0' && r <= '9'
		if !isUpper && !isDigit && r != '_' {
			return true
		}
	}
	return false
}
