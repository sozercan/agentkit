package utils

import (
	"fmt"
	"net/url"
	"path"

	"github.com/moby/buildkit/client/llb"
)

// FileNameFromURL returns the base filename of a URL's path.
func FileNameFromURL(urlString string) (string, error) {
	parsedURL, err := url.Parse(urlString)
	if err != nil {
		return "", fmt.Errorf("parsing url %q: %w", urlString, err)
	}
	return path.Base(parsedURL.Path), nil
}

// Sh builds a /bin/sh -c RunOption from a literal command.
func Sh(cmd string) llb.RunOption {
	return llb.Args([]string{"/bin/sh", "-c", cmd})
}

// Shf builds a /bin/sh -c RunOption from a format string.
//
// SECURITY: never interpolate untrusted prompt/instruction text into a command
// via Shf. Untrusted bytes belong in files written with llb.Mkfile, not in argv.
func Shf(cmd string, v ...interface{}) llb.RunOption {
	return llb.Args([]string{"/bin/sh", "-c", fmt.Sprintf(cmd, v...)})
}

// Bashf builds a /bin/bash -c RunOption from a format string.
func Bashf(cmd string, v ...interface{}) llb.RunOption {
	return llb.Args([]string{"/bin/bash", "-c", fmt.Sprintf(cmd, v...)})
}
