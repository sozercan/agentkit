# Human approval for Orka tools

AgentKit can wait while a person reviews an Orka-managed tool call. Orka stores
the proposed action, checks reviewer permissions, and decides whether the tool
runs. AgentKit receives the final result for the original call and continues
without resubmitting the prompt or repeating completed tool steps.

This requires matching Orka, AgentKit, and, for hosted execution, Foundry broker
images qualified for `supportsBrokeredToolApprovals`. The direct path uses the
Microsoft Agent Framework ACP adapter and Orka's loopback MCP server. The hosted
path uses AgentKit's brokered Responses adapter through `agent-runtime-foundry`.
Do not advertise approval support for other runtime combinations without their
matching acceptance checks.

AgentKit's local `tools[].approval` setting is separate. Omission and `never`
remain valid; `auto` and `always` remain unsupported. `supportsPermissions`
stays false. Approval of shell commands or files inside the runtime is outside
this feature.

## Wait limits and setup

| Limit | Qualified setup |
|---|---|
| Human review | At most 600 seconds, subject to tighter task/session limits |
| Tool execution after approval | At most 240 seconds |
| Direct MCP `tools/call` | Orka injects `AGENTKIT_MCP_TIMEOUT=900` into the ACP child |
| Hosted AgentKit pending state | Set `AGENTKIT_FOUNDRY_RESPONSE_STATE_TTL_SECONDS=1800` |
| Foundry hosted-session idle timeout | Configure at least 1800 seconds |

The complete task needs time for initial model work, review, tool execution,
result delivery, and model continuation. A larger MCP or state timeout does not
extend a task deadline, an expired lease, or a platform session limit. A review
that reaches its deadline expires without executing the proposed action.

The direct MAF default remains 120 seconds outside the qualified Orka child.
Confirm the child receives the 900-second override, rather than setting it only
on the supervisor process. The MAF adapter applies that value to both its MCP
request timeout and HTTP client. It does not retry a tool call after connection
loss because the action might already have run.

Hosted AgentKit's `/readiness` response reports
`foundryResponses.stateTtlSeconds`; check that it is 1800 in the configured
deployment. Also verify the idle timeout on the deployed Foundry agent version.
[Foundry's hosted-session documentation](https://learn.microsoft.com/azure/foundry/agents/how-to/manage-hosted-sessions)
describes that platform setting. Keep the same `agent_session_id` and the
authenticated continuation proof on subsequent requests.

For AgentKit process recovery, configure `AGENTKIT_FOUNDRY_RESPONSE_STATE_FILE`
on access-controlled storage that survives the process restart. Use one writer
and preserve the same hosted session identity. Missing or expired state rejects
the continuation; it never authorizes restarting the original action. This
does not provide recovery of an entire lost Foundry runtime session.

## Outcomes

While review is pending, the tool execution count remains zero. The direct MCP
request stays open. Hosted AgentKit retains its pending `function_call`; Orka
sends no `function_call_output` until the result is final. `approved:false`
must never be sent merely because the person has not decided.

| Final outcome | AgentKit behavior |
|---|---|
| Approved and executed | Continue the original call with its actual output |
| `approval_declined` | Report that the person declined the call |
| `approval_expired` | Report that the review expired |
| `approval_cancelled` | Report cancellation when a final result can be delivered |
| `approval_stale` | Report that the approval no longer authorizes the call |
| `tool_execution_failed` | Report execution failure, distinct from a human decline |
| `tool_outcome_unknown` | Stop automatic continuation; never retry the uncertain action |

The direct adapter accepts these codes only from an MCP error result's
`structuredContent.code`. It replaces tool-controlled messages with fixed text.
An unknown outcome fails the direct prompt. Hosted AgentKit returns a fixed
unknown-outcome response and caches it, without asking the model for another
tool call. Identical hosted result delivery returns the cached response;
conflicting results are rejected.

Orka controls cancellation. Cancelling a task or losing its authority stops
the waiting runtime. A later approval must not revive it. The AgentKit hosted
endpoint does not independently grant permission to resume a cancelled task.

## Acceptance through Orka

Use Orka's
[`examples/human-approval-v2`](https://github.com/orka-agents/orka/tree/main/examples/human-approval-v2)
fixture with automatic `read-inventory` and approval-required
`create-work-order`. Both are simulated; the action records an execution count.
Follow its setup and Task fixture for each qualified runtime. Keep separate
runtime capacity available for an independent task while the first waits.

Inspect the pending review in Orka's task approval panel or through the normal
API. The following commands operate on an already-created disposable task.
`ORKA_CURL_CONFIG` names a private curl configuration containing the required
authentication. Keep its credential out of shell history and command arguments.

```sh
curl --fail --silent --show-error --config "$ORKA_CURL_CONFIG" \
  "$ORKA_API_URL/api/v1/tasks/$TASK_NAME/approvals?namespace=$ORKA_NAMESPACE" |
  jq '{taskName, approvals: [.approvals[] | {
    id, action, targetTool, targetArgsPreview, targetArgsDigest,
    status, expiresAt, executionOutcome
  }]}'
```

Check the exact tool, safe argument preview, task, and expiry before selecting
`APPROVAL_ID`. Confirm the simulated action count is zero, then approve it:

```sh
curl --fail --silent --show-error --config "$ORKA_CURL_CONFIG" \
  -H 'Content-Type: application/json' \
  --data '{"decision":"approve","reason":"Simulated action acceptance"}' \
  "$ORKA_API_URL/api/v1/tasks/$TASK_NAME/approvals/$APPROVAL_ID/decision?namespace=$ORKA_NAMESPACE"
```

The action must run once and its receipt must reach the original conversation.
Repeat with a fresh task and `"decision":"decline"`; the action count must stay
zero. Use fresh disposable tasks for cancellation, review expiry, a tool error
after approval, and duplicate or competing decisions. Check both execution
count and Orka's stored `executionOutcome`. Do not treat a successful API request
alone as evidence that a tool ran.

For direct execution, include a review longer than 120 seconds and within the
600-second limit. For hosted execution, restart only the AgentKit process with
saved pending state and prove the matching continuation works. Repeat without
saved state and confirm it fails without a new action. Verify an unrelated
conversation progresses while review is pending.

## Local compatibility tests

The deterministic tests cover waiting, named outcomes, cancellation, independent
work, saved-state recovery, and duplicate continuations. The HTTP test uses the
production MAF client and MCP SDK with a counted simulated broker. Run it with
a real wait longer than the old default:

```sh
AGENTKIT_TEST_APPROVAL_WAIT_SECONDS=121 \
  uv run --directory runtimes/microsoft-agent-framework --extra dev \
  pytest -q tests/test_orka_approvals.py -k http_mcp
uv run --directory runtimes/common --extra dev \
  pytest -q tests/test_foundry_approvals.py
```

These tests establish runtime compatibility. Live Orka approval API execution
and public Foundry gateway behavior require the separate acceptance run above.
