package config

import (
	"crypto/sha256"
	"fmt"
	"strings"
	"testing"
	"unicode"
	"unicode/utf8"
)

func TestLowerBrokeredTextUnicode15(t *testing.T) {
	var scalarValues strings.Builder
	for r := rune(0); r <= unicode.MaxRune; r++ {
		if utf8.ValidRune(r) {
			scalarValues.WriteRune(r)
		}
	}
	// Recorded from Go 1.26.1 strings.ToLower, unicode.Version 15.0.0,
	// over every valid Unicode scalar in ascending order.
	const want = "137590953b837f1ec8b7c02b3a0425d0789df789a59ad256e7a63416f9fc4c11"
	got := fmt.Sprintf("%x", sha256.Sum256([]byte(lowerBrokeredText(scalarValues.String()))))
	if got != want {
		t.Fatalf("brokered lowercase mapping differs from Unicode 15: got %s, want %s", got, want)
	}
}

func TestValidateBrokeredDescriptionsPreservesUnicode15(t *testing.T) {
	for _, test := range []struct {
		description string
		valid       bool
	}{
		{description: "Count input tokens\u1c89", valid: true},
		{description: "Count input tokens\ua7cb", valid: true},
		{description: "Count input tokens\ua7ce", valid: true},
		{description: "Count input tokens\U00010d50", valid: true},
		{description: "Count input tokens\U00010d65", valid: true},
		{description: "Count input tokens\U00016ea0", valid: true},
		{description: "Count input tokens\U00016eb8", valid: true},
		{description: "Count input Tokens\U00010d50", valid: false},
		{description: "Count input tokens\u00c9", valid: false},
		{description: "Count input tokens\u0130", valid: false},
		{description: "Count input tokens_\U00010d50", valid: false},
		{description: "Count input tokens\U00010d50 abc123", valid: false},
		{description: "token\U00016ea0=abc123", valid: false},
		{description: "Read {auth\U00016ea0}", valid: false},
		{description: "Bas\u0130c dXNlcjpwYXNz", valid: false},
	} {
		t.Run(test.description, func(t *testing.T) {
			cfg := validMinimalConfig()
			cfg.BrokeredTools = []BrokeredTool{{
				Name:          safeLookupToolName,
				Description:   test.description,
				BrokeredClass: BrokeredClassRead,
				Parameters:    map[string]any{jsonSchemaTypeKey: jsonSchemaTypeObject},
			}}
			err := cfg.Validate()
			if test.valid {
				if err != nil {
					t.Fatalf("Unicode 15 description should be accepted: %v", err)
				}
			} else if err == nil || !strings.Contains(err.Error(), "brokeredTools[0].description") {
				t.Fatalf("credential-shaped description should be rejected, got %v", err)
			}
		})
	}
}
