package runtimes

import (
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"testing"

	"github.com/goccy/go-yaml"
)

type catalogEntry struct {
	APIVersion   string   `yaml:"apiVersion"`
	Runtime      string   `yaml:"runtime"`
	Aliases      []string `yaml:"aliases,omitempty"`
	Capabilities []string `yaml:"capabilities,omitempty"`
	Adapter      string   `yaml:"adapter"`
}

func TestRuntimeCatalogFilesMatchRuntimeSpecs(t *testing.T) {
	catalogDir := filepath.Join("..", "..", "..", "runtimes", "catalog")

	seen := map[string]bool{}
	for _, spec := range Runtimes {
		path := filepath.Join(catalogDir, spec.Name+".yaml")
		raw, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("runtime %q has no catalog file %s: %v", spec.Name, path, err)
		}
		var entry catalogEntry
		if err := yaml.Unmarshal(raw, &entry); err != nil {
			t.Fatalf("decode %s: %v", path, err)
		}
		if entry.APIVersion != "v1alpha1" {
			t.Errorf("%s apiVersion = %q, want v1alpha1", path, entry.APIVersion)
		}
		if entry.Runtime != spec.Name {
			t.Errorf("%s runtime = %q, want %q", path, entry.Runtime, spec.Name)
		}
		if entry.Adapter != spec.DefaultAdapterRef {
			t.Errorf("%s adapter = %q, want %q", path, entry.Adapter, spec.DefaultAdapterRef)
		}
		gotAliases := append([]string(nil), entry.Aliases...)
		wantAliases := append([]string(nil), spec.Aliases...)
		sort.Strings(gotAliases)
		sort.Strings(wantAliases)
		if !reflect.DeepEqual(gotAliases, wantAliases) {
			t.Errorf("%s aliases = %v, want %v", path, gotAliases, wantAliases)
		}

		gotCapabilities := append([]string(nil), entry.Capabilities...)
		wantCapabilities := append([]string(nil), spec.Capabilities...)
		sort.Strings(gotCapabilities)
		sort.Strings(wantCapabilities)
		if !reflect.DeepEqual(gotCapabilities, wantCapabilities) {
			t.Errorf("%s capabilities = %v, want %v", path, gotCapabilities, wantCapabilities)
		}
		seen[filepath.Base(path)] = true
	}

	entries, err := filepath.Glob(filepath.Join(catalogDir, "*.yaml"))
	if err != nil {
		t.Fatalf("glob catalog: %v", err)
	}
	for _, path := range entries {
		if !seen[filepath.Base(path)] {
			t.Errorf("catalog file %s has no RuntimeSpec", path)
		}
	}
}
