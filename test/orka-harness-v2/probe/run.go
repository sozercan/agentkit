package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"time"

	v2 "github.com/orka-agents/orka/internal/harness/v2"
	"github.com/orka-agents/orka/internal/harness/v2/conformance"
)

type outcome struct {
	Scenario         string       `json:"scenario"`
	Status           string       `json:"status"`
	Error            string       `json:"error,omitempty"`
	Terminal         v2.EventType `json:"terminal,omitempty"`
	ProviderRequests int          `json:"providerRequests"`
	BrokerCalls      int          `json:"brokerCalls"`
	ToolResults      int          `json:"toolResultsObservedByProvider"`
	DurationMillis   int64        `json:"durationMillis"`
}

type report struct {
	Adapter                  string           `json:"adapter"`
	Mode                     string           `json:"mode"`
	Passed                   bool             `json:"passed"`
	AdapterDigest            string           `json:"adapterDigest"`
	AgentConfigurationDigest string           `json:"agentConfigurationDigest"`
	ProfileDigest            v2.ProfileDigest `json:"profileDigest"`
	Scenarios                []outcome        `json:"scenarios"`
}

type runtimeSession struct {
	id        v2.RuntimeSessionID
	fence     v2.Fence
	workspace v2.WorkspaceSpec
	history   []historyMessage
	uid       int
	pid       int
	directory string
	retired   bool
}

type suite struct {
	ctx      context.Context
	cfg      settings
	url      string
	client   *v2.Client
	fixture  *fixture
	limits   v2.ProtocolLimits
	sessions []*runtimeSession
	usedUIDs map[int]bool
	report   report
}

func exercise(ctx context.Context, cfg settings, runtimeURL, upstream, results string) (runErr error) {
	f, closeFixture, err := startFixture(ctx, cfg, upstream)
	if err != nil {
		return err
	}
	defer closeFixture()
	s := &suite{ctx: ctx, cfg: cfg, url: runtimeURL, fixture: f, usedUIDs: map[int]bool{}, report: report{
		Adapter: cfg.Adapter, Mode: cfg.Mode, AdapterDigest: cfg.Profile.AdapterDigests["agentkit-serve-acp"],
		AgentConfigurationDigest: cfg.Profile.AgentConfigurationDigest, ProfileDigest: cfg.Fence.RuntimeProfileDigest,
	}}
	s.client, err = s.newClient(cfg.ControllerToken, cfg.CapabilityKey)
	if err != nil {
		return err
	}
	defer func() {
		cleanupCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 45*time.Second)
		defer cancel()
		cleanupErr := s.check("session-cleanup", func() error { return s.cleanup(cleanupCtx) })
		runErr = errors.Join(runErr, cleanupErr)
		s.report.Passed = runErr == nil
		data, err := json.MarshalIndent(s.report, "", "  ")
		if err == nil {
			err = os.WriteFile(results, append(data, '\n'), 0o644)
		}
		runErr = errors.Join(runErr, err)
	}()
	if err := s.check("startup-and-authentication", s.startup); err != nil {
		return err
	}
	if err := s.check("admission-rejections", s.admission); err != nil {
		return err
	}
	var session *runtimeSession
	if err := s.check("child-and-session-identity", func() error {
		var err error
		session, err = s.createSession()
		return err
	}); err != nil {
		return err
	}
	for _, kind := range []string{"completion", "continuation", "tool"} {
		if err := s.prompt(session, kind); err != nil {
			return err
		}
	}
	if err := s.check("successful-session-deletion", func() error { return s.deleteSession(s.ctx, session) }); err != nil {
		return err
	}
	kinds := []string{"provider-failure", "tool-failure", "cancel", "deadline"}
	if cfg.Mode == "live" {
		// A real model chooses the blocking tool in the live cancellation check.
		// Offline fixtures supply authoritative deadline and failure scheduling.
		kinds = []string{"cancel"}
	}
	for _, kind := range kinds {
		var session *runtimeSession
		if err := s.check(kind+"-session", func() error {
			var err error
			session, err = s.createSession()
			return err
		}); err != nil {
			return err
		}
		if err := s.prompt(session, kind); err != nil {
			return err
		}
	}
	return nil
}

func (s *suite) newClient(token, secret string) (*v2.Client, error) {
	return v2.NewClient(s.url, v2.WithControllerBearerToken(token), v2.WithOperationCapabilitySecret([]byte(secret)),
		v2.WithControlTimeout(30*time.Second), v2.WithStatusCapabilityBinding(v2.StatusCapabilityBinding{
			RuntimeInstanceID: s.cfg.Fence.RuntimeInstanceID, RuntimeProfileDigest: s.cfg.Fence.RuntimeProfileDigest,
		}))
}

func (s *suite) check(name string, run func() error) error {
	started := time.Now()
	err := run()
	item := outcome{Scenario: name, Status: "passed", DurationMillis: time.Since(started).Milliseconds()}
	if err != nil {
		item.Status, item.Error = "failed", err.Error()
	}
	s.report.Scenarios = append(s.report.Scenarios, item)
	// These records deliberately omit wire payloads, provider output, and keys.
	_ = json.NewEncoder(os.Stdout).Encode(map[string]any{
		"adapter": s.cfg.Adapter, "mode": s.cfg.Mode,
		"scenario": name, "status": item.Status, "error": item.Error,
	})
	if err != nil {
		return fmt.Errorf("adapter=%s scenario=%s: %w", s.cfg.Adapter, name, err)
	}
	return nil
}

func (s *suite) startup() error {
	ctx, cancel := context.WithTimeout(s.ctx, 45*time.Second)
	defer cancel()
	if err := eventually(ctx, func() (bool, error) {
		health, err := s.client.Health(ctx)
		return err == nil && health.Status == v2.HealthStatusOK, nil
	}); err != nil {
		return fmt.Errorf("supervisor did not become healthy: %w", err)
	}
	caps, err := s.client.Capabilities(ctx)
	if err != nil {
		return err
	}
	s.limits = caps.Limits
	if caps.RuntimeProfileDigest != s.cfg.Fence.RuntimeProfileDigest || !reflect.DeepEqual(caps.AdapterDigests, s.cfg.Profile.AdapterDigests) ||
		caps.SupportsAgentSessionConfiguration || caps.Provider.SupportsPermissions || !caps.Provider.SupportsTools || !caps.Provider.SupportsCancel ||
		!reflect.DeepEqual(caps.Provider.ProviderKinds, []string{"agentkit"}) || !reflect.DeepEqual(caps.Provider.Models, []string{s.cfg.Profile.Model}) {
		return errors.New("capabilities differ from the registered immutable AgentKit profile")
	}
	result := conformance.Check(ctx, conformance.Target{
		BaseURL: s.url, ControllerBearerToken: s.cfg.ControllerToken, OperationCapabilitySecret: []byte(s.cfg.CapabilityKey),
		ControlTimeout: 30 * time.Second, ExpectedRuntimeInstanceID: s.cfg.Fence.RuntimeInstanceID,
		ExpectedControllerEpoch: s.cfg.Fence.ControllerEpoch, ExpectedFence: &s.cfg.Fence,
		Profile: s.cfg.Profile, ToolPolicy: s.cfg.MCP.ToolPolicy, ApprovalPolicy: s.cfg.MCP.ApprovalPolicy,
		Limits: caps.Limits, SupportsDrain: caps.SupportsDrain, SupportsPublicationFinalization: caps.SupportsPublicationFinalization,
		WorkspaceGovernance: caps.WorkspaceGovernance,
	})
	if !result.Passed || result.LifecycleProbeExecuted {
		return fmt.Errorf("native conformance authentication probes: %s", result.Message)
	}
	provider, broker, _, err := s.fixture.counts()
	if err != nil {
		return err
	}
	if provider != 0 || broker != 0 || len(result.ObservedStatus.Sessions) != 0 {
		return errors.New("safe probes started provider/tool/session work")
	}
	return verifySupervisorIdentity()
}

func metadata(fence v2.Fence, task v2.TaskUID, prompt v2.PromptID, expiry time.Time) v2.MutationMetadata {
	m := v2.MutationMetadata{
		Fence: fence, TaskUID: task, PromptID: prompt, OperationID: v2.OperationID("op-" + randomID()),
		RequestDigestSchemaVersion: v2.RequestDigestSchemaVersion, ExpiresAt: expiry,
	}
	if task != "" {
		m.TaskAttempt = 1
	}
	return m
}

func seal(request any, m *v2.MutationMetadata) error {
	digest, err := v2.CanonicalRequestDigest(request)
	m.RequestDigest = digest
	return err
}

func (s *suite) sessionRequest() (v2.CreateRuntimeSessionRequest, error) {
	fence := s.cfg.Fence
	fence.RuntimeSessionUID = v2.RuntimeSessionUID("suid-" + randomID())
	fence.RuntimeSessionGeneration = 1
	request := v2.CreateRuntimeSessionRequest{
		Protocol: v2.ProtocolVersion, Metadata: metadata(fence, v2.TaskUID("task-"+randomID()), "", time.Now().UTC().Add(45*time.Second)),
		RuntimeSessionID: v2.RuntimeSessionID("session-" + randomID()), Profile: s.cfg.Profile, MCPConfiguration: s.cfg.MCP,
		Workspace: v2.WorkspaceSpec{Intent: v2.WorkspaceIntentRead, Baseline: v2.WorkspaceBaseline{
			RepositoryIdentity: "agentkit-v2-e2e.invalid/empty", Revision: "empty", TreeDigest: sha(nil),
		}},
	}
	return request, seal(&request, &request.Metadata)
}

func (s *suite) createSession() (*runtimeSession, error) {
	request, err := s.sessionRequest()
	if err != nil {
		return nil, err
	}
	response, err := s.client.CreateRuntimeSession(s.ctx, request)
	if err != nil {
		return nil, err
	}
	session := &runtimeSession{id: request.RuntimeSessionID, fence: request.Metadata.Fence, workspace: request.Workspace}
	s.sessions = append(s.sessions, session)
	if response.Classification.Class != v2.RequestClassificationFresh || response.Session.State != v2.RuntimeSessionStateIdle ||
		!strings.HasPrefix(response.Session.ProviderSessionID, "agentkit-") || response.Session.RuntimeSessionUID != session.fence.RuntimeSessionUID ||
		response.Session.RuntimeProfileDigest != s.cfg.Fence.RuntimeProfileDigest || response.Session.SupervisorBootID != s.cfg.Fence.SupervisorBootID {
		return nil, errors.New("session did not initialize the real AgentKit ACP child with the expected identity")
	}
	if err := s.verifyChild(session); err != nil {
		return nil, err
	}
	return session, nil
}

func (s *suite) promptRequest(session *runtimeSession, scenario *scenario) (v2.StartPromptRequest, error) {
	now := time.Now().UTC()
	ttl := 90 * time.Second
	if scenario.kind == "deadline" {
		ttl = 12 * time.Second
	}
	lease := v2.PromptLease{Generation: 1, IssuedAt: now, ExpiresAt: now.Add(ttl)}
	m := metadata(session.fence, v2.TaskUID("task-"+randomID()), v2.PromptID("prompt-"+randomID()), lease.ExpiresAt)
	auth := v2.PromptMCPAuthorization{
		RuntimeSessionUID: session.fence.RuntimeSessionUID, SessionGeneration: session.fence.RuntimeSessionGeneration,
		TaskUID: m.TaskUID, TaskAttempt: m.TaskAttempt, PromptID: m.PromptID, LeaseGeneration: lease.Generation,
		ToolPolicyDigest: s.cfg.Profile.ToolPolicyDigest, ApprovalPolicyDigest: s.cfg.Profile.ApprovalPolicyDigest,
		MCPConfigurationDigest: s.cfg.Profile.MCPConfigurationDigest, ToolPolicy: s.cfg.MCP.ToolPolicy,
		ApprovalPolicy: s.cfg.MCP.ApprovalPolicy, ExpiresAt: lease.ExpiresAt,
	}
	request := v2.StartPromptRequest{
		Protocol: v2.ProtocolVersion, Metadata: m, Lease: lease, MCPAuthorization: auth,
		Input: v2.PromptInput{Content: []v2.ContentBlock{{Type: v2.ContentBlockText, Text: scenario.input}}},
	}
	return request, seal(&request, &request.Metadata)
}

func (s *suite) prompt(session *runtimeSession, kind string) error {
	scenario := &scenario{
		name: kind, kind: kind, marker: "E2E_" + randomID(), receipt: "RECEIPT_" + randomID(),
		previous: append([]historyMessage{}, session.history...), blocked: make(chan time.Time, 1), disconnected: make(chan struct{}, 1),
	}
	scenario.input = "Reply with exactly this marker: " + scenario.marker + ". Do not call tools."
	if kind == "tool" || kind == "tool-failure" || kind == "cancel" || kind == "deadline" {
		scenario.input = "Call " + modelToolName + " exactly once with value \"" + scenario.marker + "\". Wait for its result, then include its exact receipt and the value in your answer."
	}
	err := s.check(kind, func() error { return s.executePrompt(session, scenario) })
	// Add counts to the saved report even when a scenario failed midway.
	item := &s.report.Scenarios[len(s.report.Scenarios)-1]
	s.fixture.mu.Lock()
	item.ProviderRequests, item.BrokerCalls, item.ToolResults = scenario.requests, scenario.tools, scenario.toolResults
	item.Terminal = scenario.terminal
	s.fixture.mu.Unlock()
	return err
}

type streamResult struct {
	summary v2.PromptStreamSummary
	events  []v2.Event
	err     error
}

func (s *suite) executePrompt(session *runtimeSession, scenario *scenario) error {
	ctx, cancel := context.WithTimeout(s.ctx, 115*time.Second)
	defer cancel()
	request, err := s.promptRequest(session, scenario)
	if err != nil {
		return err
	}
	scenario.metadata = request.Metadata
	if err := s.fixture.selectScenario(scenario); err != nil {
		return err
	}
	done := make(chan streamResult, 1)
	go func() {
		var result streamResult
		result.summary, result.err = s.client.StreamPrompt(ctx, session.id, request, func(event v2.Event) error {
			result.events = append(result.events, event)
			return nil
		})
		done <- result
	}()
	var cancellation *v2.CancelPromptResponse
	if scenario.kind == "cancel" || scenario.kind == "deadline" {
		select {
		case started := <-scenario.blocked:
			if !started.Before(request.Lease.ExpiresAt.Add(-time.Second)) {
				return errors.New("blocking tool started too late to prove cancellation/expiry during execution")
			}
		case early := <-done:
			return fmt.Errorf("prompt settled before the broker observed a blocking tool: %v", early.err)
		case <-ctx.Done():
			return errors.New("broker never observed the requested blocking tool")
		}
		status, err := s.client.Status(ctx)
		if err != nil {
			return err
		}
		if len(status.ActivePrompts) != 1 || status.ActivePrompts[0].PromptID != request.Metadata.PromptID {
			return errors.New("blocked broker request did not have a running native prompt")
		}
		if scenario.kind == "cancel" {
			cancellation, err = s.cancelPrompt(ctx, session, request)
			if err != nil {
				return err
			}
		}
	}
	var result streamResult
	select {
	case result = <-done:
	case <-ctx.Done():
		return errors.New("native prompt stream did not settle within the scenario deadline")
	}
	if result.err != nil {
		return result.err
	}
	_, _, _, err = s.fixture.counts()
	if err != nil {
		return err
	}
	terminal := result.summary.Terminal
	if terminal != nil {
		scenario.terminal = terminal.Type
	}
	want := v2.EventCompleted
	switch scenario.kind {
	case "provider-failure", "tool-failure":
		want = v2.EventFailed
	case "cancel", "deadline":
		want = v2.EventCancelled
	}
	if terminal == nil || terminal.Type != want || !result.summary.Accepted {
		got := v2.EventType("missing")
		if terminal != nil {
			got = terminal.Type
		}
		return fmt.Errorf("native terminal = %s, want %s", got, want)
	}
	if err := validateToolEvents(result.events, scenario.kind); err != nil {
		return err
	}
	s.fixture.mu.Lock()
	providerRequests, toolCalls, toolResults := scenario.requests, scenario.tools, scenario.toolResults
	s.fixture.mu.Unlock()
	if providerRequests < 1 {
		return errors.New("scenario never traversed the real provider proxy")
	}
	if scenario.kind == "tool" && (toolCalls != 1 || toolResults != 1 || providerRequests != 2) {
		return fmt.Errorf("tool round-trip counts = provider:%d broker:%d results:%d, want 2/1/1", providerRequests, toolCalls, toolResults)
	}
	if scenario.kind == "tool-failure" && (toolCalls != 1 || providerRequests != 1) {
		return errors.New("broker failure must settle after one provider request and one tool call")
	}
	if want == v2.EventCompleted {
		var answer strings.Builder
		for _, block := range terminal.Completed.Result.Content {
			answer.WriteString(block.Text)
		}
		if !strings.Contains(answer.String(), scenario.marker) || (scenario.kind == "tool" && !strings.Contains(answer.String(), scenario.receipt)) {
			return errors.New("successful output omitted the unique marker or broker-only receipt")
		}
		if err := s.validateWorkspace(ctx, session, request, *terminal); err != nil {
			return err
		}
		session.history = append(session.history, historyMessage{"user", scenario.input}, historyMessage{"assistant", answer.String()})
		return nil
	}
	if scenario.kind == "deadline" && terminal.Identity.Timestamp.Before(request.Lease.ExpiresAt) {
		return errors.New("deadline prompt settled before its lease expired")
	}
	if scenario.kind == "cancel" || scenario.kind == "deadline" {
		select {
		case <-scenario.disconnected:
		case <-ctx.Done():
			return errors.New("supervisor did not cancel the blocked broker request")
		}
		if toolCalls != 1 {
			return errors.New("cancellation/expiry did not execute exactly one blocking tool")
		}
	}
	if err := s.waitRetired(ctx, session); err != nil {
		return err
	}
	// The canonical cancellation endpoint also returns retained settlement
	// proof for a failed/expired prompt after automatic session retirement.
	if cancellation == nil {
		cancellation, err = s.cancelPrompt(ctx, session, request)
		if err != nil {
			return err
		}
	}
	// Orka's cancellation handler and stream finisher can race to record the
	// same outcome using different timestamps. Both must belong to this prompt.
	settledAt := cancellation.Settlement.SettledAt
	earliest := request.Lease.IssuedAt
	if scenario.kind == "deadline" {
		earliest = request.Lease.ExpiresAt
	}
	if !cancellation.SettlementProven || cancellation.LiveDescendantCount != 0 || cancellation.ForcedTermination ||
		cancellation.Settlement.TerminalEvent != terminal.Type || settledAt.Before(earliest) || settledAt.After(time.Now().UTC()) {
		return fmt.Errorf("native settlement proof: proven=%t descendants=%d forced=%t terminal=%s stream=%s settledAt=%s eventAt=%s",
			cancellation.SettlementProven, cancellation.LiveDescendantCount, cancellation.ForcedTermination,
			cancellation.Settlement.TerminalEvent, terminal.Type, cancellation.Settlement.SettledAt.Format(time.RFC3339Nano), terminal.Identity.Timestamp.Format(time.RFC3339Nano))
	}
	return s.rejectRetiredPrompt(session)
}

func (s *suite) cancelPrompt(ctx context.Context, session *runtimeSession, prompt v2.StartPromptRequest) (*v2.CancelPromptResponse, error) {
	now := time.Now().UTC()
	m := metadata(session.fence, prompt.Metadata.TaskUID, prompt.Metadata.PromptID, now.Add(30*time.Second))
	request := v2.CancelPromptRequest{Protocol: v2.ProtocolVersion, Metadata: m, Reason: v2.CancelReasonUserRequested, SettlementDeadline: now.Add(20 * time.Second)}
	if err := seal(&request, &request.Metadata); err != nil {
		return nil, err
	}
	return s.client.CancelPrompt(ctx, session.id, request)
}

func (s *suite) validateWorkspace(ctx context.Context, session *runtimeSession, prompt v2.StartPromptRequest, terminal v2.Event) error {
	settlement := v2.PromptSettlement{
		TerminalEvent: v2.EventCompleted, Outcome: v2.PromptOutcomeSucceeded,
		StopReason: terminal.Completed.StopReason, SettledAt: terminal.Identity.Timestamp,
	}
	digest, err := v2.CanonicalPromptSettlementDigest(settlement)
	if err != nil {
		return err
	}
	// Successful strict-governed prompts must pass no-change workspace
	// validation before the supervisor makes this same child reusable.
	request := v2.CreateWorkspaceDeltaRequest{
		Protocol: v2.ProtocolVersion, Metadata: metadata(session.fence, prompt.Metadata.TaskUID, prompt.Metadata.PromptID, time.Now().UTC().Add(30*time.Second)),
		DeltaID: v2.WorkspaceDeltaID("delta-" + randomID()), Intent: v2.WorkspaceIntentRead, VerifiedBaseline: session.workspace.Baseline,
		PromptSettlementDigest: digest, Limits: v2.WorkspaceDeltaLimits{MaxBytes: s.limits.MaxWorkspaceDeltaBytes, MaxEntries: 4096},
	}
	if err := seal(&request, &request.Metadata); err != nil {
		return err
	}
	response, err := s.client.CreateWorkspaceDelta(ctx, session.id, request)
	if err != nil {
		return err
	}
	if response.Delta.State != v2.WorkspaceDeltaNoChange || !response.Delta.NoFollowVerified || !response.Delta.PublicationSafe {
		return errors.New("successful prompt failed canonical no-change workspace validation")
	}
	status, err := s.client.Status(ctx)
	if err != nil {
		return err
	}
	if len(status.Sessions) != 1 || status.Sessions[0].RuntimeSessionUID != session.fence.RuntimeSessionUID || status.Sessions[0].State != v2.RuntimeSessionStateIdle {
		return errors.New("workspace validation did not return the same RuntimeSession to idle")
	}
	return verifyProcess(session.pid, session.uid)
}

func validateToolEvents(events []v2.Event, kind string) error {
	var updates []v2.ToolCallUpdate
	for _, event := range events {
		if event.Update != nil && event.Update.ToolCall != nil {
			updates = append(updates, *event.Update.ToolCall)
		}
	}
	wantTool := kind == "tool" || kind == "tool-failure" || kind == "cancel" || kind == "deadline"
	if !wantTool {
		if len(updates) != 0 {
			return errors.New("tool events appeared in a scenario that requested no tools")
		}
		return nil
	}
	wantStatus := v2.ToolCallStatusFailed
	if kind == "tool" {
		wantStatus = v2.ToolCallStatusCompleted
	}
	if len(updates) != 2 || updates[0].Status != v2.ToolCallStatusInProgress || updates[1].Status != wantStatus ||
		updates[0].ToolCallID != updates[1].ToolCallID || updates[0].Title != modelToolName {
		return fmt.Errorf("tool lifecycle must be one in_progress followed by one %s update for the same tool", wantStatus)
	}
	return nil
}

func (s *suite) deleteSession(ctx context.Context, session *runtimeSession) error {
	request := v2.DeleteRuntimeSessionRequest{
		Protocol: v2.ProtocolVersion,
		Metadata: metadata(session.fence, "", "", time.Now().UTC().Add(30*time.Second)), Reason: "E2E complete",
	}
	if err := seal(&request, &request.Metadata); err != nil {
		return err
	}
	response, err := s.client.DeleteRuntimeSession(ctx, session.id, request)
	if err != nil {
		return err
	}
	if response.State != v2.RuntimeSessionStateDeleted {
		return errors.New("native session deletion did not confirm deleted state")
	}
	return s.waitRetired(ctx, session)
}

func (s *suite) waitRetired(ctx context.Context, session *runtimeSession) error {
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	err := eventually(ctx, func() (bool, error) {
		status, err := s.client.Status(ctx)
		if err != nil {
			return false, err
		}
		for _, candidate := range status.Sessions {
			if candidate.RuntimeSessionUID == session.fence.RuntimeSessionUID {
				return false, nil
			}
		}
		if _, err := os.Stat(session.directory); !errors.Is(err, os.ErrNotExist) {
			return false, nil
		}
		children, err := childProcesses()
		if err != nil {
			return false, err
		}
		for _, child := range children {
			if child.uid == session.uid {
				return false, nil
			}
		}
		_, _, active, err := s.fixture.counts()
		return active == 0, err
	})
	if err != nil {
		return fmt.Errorf("child/session/proxy cleanup was not proven: %w", err)
	}
	session.retired = true
	return nil
}

func (s *suite) cleanup(ctx context.Context) error {
	for _, session := range s.sessions {
		if session.retired {
			continue
		}
		status, err := s.client.Status(ctx)
		if err != nil {
			return err
		}
		present := false
		for _, existing := range status.Sessions {
			present = present || existing.RuntimeSessionUID == session.fence.RuntimeSessionUID
		}
		if present {
			if err := s.deleteSession(ctx, session); err != nil {
				return err
			}
		}
	}
	status, err := s.client.Status(ctx)
	if err != nil {
		return err
	}
	if len(status.Sessions) != 0 || len(status.ActivePrompts) != 0 || status.Pressure.LiveDescendants != 0 || status.Pressure.ActivePrompts != 0 {
		return errors.New("runtime retained sessions, prompts, or descendants")
	}
	entries, err := os.ReadDir("/sessions")
	if err != nil {
		return err
	}
	for _, entry := range entries {
		if entry.IsDir() {
			return errors.New("runtime left a private session directory behind")
		}
	}
	children, err := childProcesses()
	if err != nil {
		return err
	}
	if len(children) != 0 {
		return errors.New("runtime left child processes behind")
	}
	_, _, active, err := s.fixture.counts()
	if err != nil {
		return err
	}
	if active != 0 {
		return errors.New("fixture still has active provider or broker requests")
	}
	return nil
}

func eventually(ctx context.Context, check func() (bool, error)) error {
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()
	for {
		ok, err := check()
		if err != nil || ok {
			return err
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
		}
	}
}

func (s *suite) verifyChild(session *runtimeSession) error {
	children, err := childProcesses()
	if err != nil {
		return err
	}
	if len(children) != 1 {
		return fmt.Errorf("expected one real ACP child, found %d", len(children))
	}
	child := children[0]
	if s.usedUIDs[child.uid] {
		return errors.New("supervisor reused a retired session UID")
	}
	s.usedUIDs[child.uid] = true
	session.uid, session.pid = child.uid, child.pid
	if err := verifyProcess(child.pid, child.uid); err != nil {
		return err
	}
	entries, err := os.ReadDir("/sessions")
	if err != nil {
		return err
	}
	for _, entry := range entries {
		if entry.IsDir() {
			if session.directory != "" {
				return errors.New("more than one private session tree exists")
			}
			session.directory = filepath.Join("/sessions", entry.Name())
		}
	}
	if session.directory == "" {
		return errors.New("real ACP child has no private session tree")
	}
	return verifyPrivateDirectory(session.directory, child.uid)
}
