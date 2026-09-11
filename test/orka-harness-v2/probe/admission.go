package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"

	v2 "github.com/orka-agents/orka/internal/harness/v2"
)

func rejected(err error, status int, code v2.ErrorCode) error {
	var clientErr *v2.ClientError
	if !errors.As(err, &clientErr) || clientErr.Kind != v2.ClientErrorHTTP || clientErr.StatusCode != status || clientErr.Code != code {
		return fmt.Errorf("expected native HTTP %d/%s rejection, got %v", status, code, err)
	}
	return nil
}

func (s *suite) admission() error {
	for _, name := range []string{"bearer", "capability", "instance", "boot", "epoch", "model", "config-digest"} {
		request, err := s.sessionRequest()
		if err != nil {
			return err
		}
		client := s.client
		status, code := http.StatusGone, v2.ErrorCodeStaleFence
		switch name {
		case "bearer":
			client, err = s.newClient(randomID(), s.cfg.CapabilityKey)
			status, code = http.StatusUnauthorized, v2.ErrorCodeUnauthenticated
		case "capability":
			client, err = s.newClient(s.cfg.ControllerToken, randomID()+randomID())
			status, code = http.StatusForbidden, v2.ErrorCodeForbidden
		case "instance":
			request.Metadata.Fence.RuntimeInstanceID = "wrong-instance"
		case "boot":
			request.Metadata.Fence.SupervisorBootID = "wrong-boot"
		case "epoch":
			request.Metadata.Fence.ControllerEpoch++
		case "model", "config-digest":
			if name == "model" {
				request.Profile.Model = "wrong-model"
			} else {
				request.Profile.AgentConfigurationDigest = sha([]byte("different config bytes"))
			}
			request.Metadata.Fence.RuntimeProfileDigest, err = v2.CanonicalProfileDigest(request.Profile)
		}
		if err != nil {
			return err
		}
		if err := seal(&request, &request.Metadata); err != nil {
			return err
		}
		_, err = client.CreateRuntimeSession(s.ctx, request)
		if err := rejected(err, status, code); err != nil {
			return fmt.Errorf("%s: %w", name, err)
		}
	}
	status, err := s.client.Status(s.ctx)
	if err != nil {
		return err
	}
	children, err := childProcesses()
	if err != nil {
		return err
	}
	if len(status.Sessions) != 0 || len(children) != 0 {
		return errors.New("rejected session admission created a child or session")
	}
	session, err := s.createSession()
	if err != nil {
		return err
	}
	for _, name := range []string{"prompt-correlation", "tool-policy"} {
		request, err := s.promptRequest(session, &scenario{kind: "completion", input: "rejected input must never execute"})
		if err != nil {
			return err
		}
		if name == "prompt-correlation" {
			request.MCPAuthorization.PromptID = "wrong-prompt"
		} else {
			request.MCPAuthorization.ToolPolicy.AllowedToolNames = []string{}
		}
		if err := seal(&request, &request.Metadata); err != nil {
			return err
		}
		path, err := v2.PromptPath(session.id, request.Metadata.PromptID)
		if err != nil {
			return err
		}
		if err := s.rejectRawPrompt(path, request); err != nil {
			return fmt.Errorf("%s: %w", name, err)
		}
	}
	provider, broker, active, err := s.fixture.counts()
	if err != nil {
		return err
	}
	if provider != 0 || broker != 0 || active != 0 {
		return errors.New("admission rejection allowed unauthorized provider/tool work")
	}
	return s.deleteSession(s.ctx, session)
}

// These intentionally invalid requests bypass only the client's preflight,
// using canonical types/digests/signatures to reach the server's validators.
func (s *suite) rejectRawPrompt(path string, request v2.StartPromptRequest) error {
	capability, err := v2.SignOperationCapability([]byte(s.cfg.CapabilityKey), v2.ClaimsForMutation(request.Metadata))
	if err != nil {
		return err
	}
	body, err := json.Marshal(request)
	if err != nil {
		return err
	}
	httpRequest, err := http.NewRequestWithContext(s.ctx, http.MethodPut, s.url+path, bytes.NewReader(body))
	if err != nil {
		return err
	}
	httpRequest.Header.Set("Content-Type", "application/json")
	httpRequest.Header.Set("Accept", v2.NDJSONMediaType+", application/json")
	httpRequest.Header.Set("Authorization", "Bearer "+s.cfg.ControllerToken)
	httpRequest.Header.Set(v2.OperationCapabilityHeader, capability)
	client := &http.Client{
		Transport: v2.NewProxylessTransport(), Timeout: 15 * time.Second,
		CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
	}
	defer client.CloseIdleConnections()
	response, err := client.Do(httpRequest)
	if err != nil {
		return errors.New("invalid prompt probe transport failed")
	}
	defer response.Body.Close()
	var failure v2.ErrorResponse
	if err := json.NewDecoder(io.LimitReader(response.Body, 8192)).Decode(&failure); err != nil {
		return errors.New("invalid prompt did not return a native error envelope")
	}
	if response.StatusCode != http.StatusBadRequest || failure.Code != v2.ErrorCodeInvalidRequest || failure.Validate() != nil {
		return fmt.Errorf("invalid prompt returned HTTP %d/%s, want 400/invalid_request", response.StatusCode, failure.Code)
	}
	return nil
}

func (s *suite) rejectRetiredPrompt(session *runtimeSession) error {
	beforeProvider, beforeBroker, _, err := s.fixture.counts()
	if err != nil {
		return err
	}
	request, err := s.promptRequest(session, &scenario{kind: "completion", input: "retired session must not continue"})
	if err != nil {
		return err
	}
	stream, err := s.client.StartPrompt(s.ctx, session.id, request)
	if stream != nil {
		_ = stream.Close()
	}
	// Retirement removes the session resource; retained cancellation proof is
	// available separately through the canonical cancellation endpoint.
	if err := rejected(err, http.StatusNotFound, v2.ErrorCodeInvalidRequest); err != nil {
		return fmt.Errorf("retired session continuation: %w", err)
	}
	provider, broker, _, err := s.fixture.counts()
	if err != nil {
		return err
	}
	if provider != beforeProvider || broker != beforeBroker {
		return errors.New("retired session continuation started provider/tool work")
	}
	return nil
}
