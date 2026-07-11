from __future__ import annotations

import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from types import TracebackType
from typing import Any

from fastapi.testclient import TestClient
import httpx
import pytest

import agentkit_serve_common.foundry as foundry_module
from agentkit_serve_common.config import AgentSpec
from agentkit_serve_common.conversation import RunRequest
from agentkit_serve_common.foundry import create_foundry_app
from agentkit_serve_common.foundry_model_loop import BrokeredChatModelLoop
from agentkit_serve_common.runtime import AgentRunError, RunResult, RuntimeSession


CONTINUATION_PROOF = "test-orka-continuation-proof"
CONTINUATION_AUTH = {"x-agentkit-brokered-continuation-proof": CONTINUATION_PROOF}
CONTINUATION_PROOF_BODY_FIELD = "brokered_continuation_proof"


def _app(spec: AgentSpec | None = None, factory: NoDirectRunFactory | None = None, **kwargs: Any):
    return create_foundry_app(
        spec or _spec(),
        factory or NoDirectRunFactory(),
        brokered_continuation_proof=CONTINUATION_PROOF,
        **kwargs,
    )


def _spec(*, tool_name: str = "conformance_read", brokered_class: str = "read") -> AgentSpec:
    return AgentSpec.model_validate(
        {
            "abiVersion": "v0",
            "metadata": {"name": "foundry-brokered-test"},
            "model": {
                "provider": "openai-compatible",
                "baseURL": "https://api.openai.com/v1",
                "name": "gpt-4o-mini",
            },
            "instructions": "Be helpful.",
            "tools": [],
            "brokeredTools": [
                {
                    "name": tool_name,
                    "description": "Safe deterministic conformance tool.",
                    "brokeredClass": brokered_class,
                    "parameters": {"type": "object", "properties": {"probe": {"type": "boolean"}}},
                }
            ],
            "expose": {"openai": True, "port": 8080},
        }
    )


def _multi_tool_spec() -> AgentSpec:
    data = _spec().model_dump(by_alias=True)
    data["brokeredTools"] = [
        {
            "name": "check-network-telemetry",
            "description": "Read telemetry.",
            "brokeredClass": "read",
            "parameters": {"type": "object", "properties": {"site": {"type": "string"}}, "required": ["site"]},
        },
        {
            "name": "get-active-incidents",
            "description": "Read active incidents.",
            "brokeredClass": "read",
            "parameters": {"type": "object"},
        },
    ]
    return AgentSpec.model_validate(data)


class NoDirectRunRuntime:
    def __init__(self) -> None:
        self.run_requests: list[RunRequest] = []

    async def __aenter__(self) -> RuntimeSession:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        return None

    async def run(self, request: RunRequest) -> RunResult:
        self.run_requests.append(request)
        raise AssertionError("Foundry brokered /responses must not execute direct AgentKit-owned tools")


class NoDirectRunFactory:
    def __init__(self) -> None:
        self.runtime = NoDirectRunRuntime()

    def build_runtime(self, spec: AgentSpec) -> RuntimeSession:
        return self.runtime


def _start(client: TestClient, prompt: str = "please read telemetry") -> dict[str, Any]:
    resp = client.post("/responses", json={"input": prompt})
    assert resp.status_code == 200, resp.text
    return resp.json()


def _continuation(response_id: str, call_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "previous_response_id": response_id,
        "input": [
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(payload, separators=(",", ":"), sort_keys=True),
                "status": "completed",
            }
        ],
    }


def _call(body: dict[str, Any]) -> dict[str, Any]:
    assert body["status"] == "completed"
    assert body["id"].startswith("caresp_")
    output = body["output"]
    assert len(output) == 1
    assert output[0]["type"] == "function_call"
    return output[0]


def _message_text(body: dict[str, Any]) -> str:
    assert body["status"] == "completed"
    message = body["output"][0]
    assert message["type"] == "message"
    return message["content"][0]["text"]


def test_foundry_brokered_requires_continuation_proof_for_readiness_and_initial_call():
    app = create_foundry_app(_spec(), NoDirectRunFactory())

    with TestClient(app) as client:
        readiness = client.get("/readiness")
        initial = client.post("/responses", json={"input": "please read telemetry"})

    assert readiness.status_code == 503
    assert readiness.json()["ready"] is False
    assert readiness.json()["foundryResponses"]["continuationAuth"] == "missing"
    assert initial.status_code == 503
    assert initial.json()["error"]["code"] == "brokered_continuation_auth_required"


def test_foundry_brokered_initial_response_emits_static_function_call_without_direct_execution():
    factory = NoDirectRunFactory()
    app = _app(_spec(), factory)

    with TestClient(app) as client:
        readiness = client.get("/readiness")
        body = _start(client)

    assert readiness.status_code == 200
    assert readiness.json()["foundryResponses"] == {
        "brokeredTools": 1,
        "ownedToolsDisabled": 0,
        "stateBackend": "memory",
        "stateTtlSeconds": 900.0,
        "stateMaxPending": 128,
        "continuationAuth": "configured",
        "runtime": "deterministic",
        "scaling": "single-replica-or-sticky-routing-required",
    }
    call = _call(body)
    assert call["name"] == "conformance_read"
    assert call["call_id"] == f"call_{body['id']}_1"
    assert json.loads(call["arguments"]) == {"probe": True}
    assert factory.runtime.run_requests == []


def test_foundry_brokered_deterministic_arguments_reject_unsafe_prompt_text():
    spec = _spec(tool_name="check-network-telemetry")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {"site": {"type": "string"}},
        "required": ["site"],
    }
    app = _app(spec)

    with TestClient(app) as client:
        resp = client.post("/responses", json={"input": "check-network-telemetry https://internal.example"})

    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "UnsafeBrokeredArguments"


def test_foundry_brokered_synthesizes_arguments_from_required_schema_fields():
    spec = _spec(tool_name="check-network-telemetry")
    tool = spec.brokered_tools[0]
    tool.parameters["required"] = ["site"]
    app = _app(spec)

    with TestClient(app) as client:
        body = _start(client, "please call check-network-telemetry")

    call = _call(body)
    assert call["name"] == "check-network-telemetry"
    assert json.loads(call["arguments"]) == {"site": "please call check-network-telemetry"}


def test_foundry_brokered_normal_followup_with_previous_response_id_is_not_treated_as_tool_output():
    app = _app()

    with TestClient(app) as client:
        resp = client.post("/responses", json={"previous_response_id": "caresp_completed_elsewhere", "input": "next question"})

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["previous_response_id"] == "caresp_completed_elsewhere"
    call = _call(body)
    assert call["type"] == "function_call"


def test_foundry_brokered_rejects_normal_followup_while_previous_response_is_pending_tool_output():
    app = _app()

    with TestClient(app) as client:
        initial = _start(client)
        resp = client.post("/responses", json={"previous_response_id": initial["id"], "input": "next question"})

    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "response_pending_function_call_output"


def test_foundry_brokered_pending_state_store_is_bounded():
    app = _app(max_pending_responses=1)

    with TestClient(app) as client:
        first = _start(client)
        second = client.post("/responses", json={"input": "another pending request"})

    assert _call(first)
    assert second.status_code == 429
    assert second.json()["error"]["code"] == "brokered_response_state_full"


def test_foundry_brokered_failed_initial_state_persist_is_retryable_without_consuming_capacity(monkeypatch, tmp_path):
    state_file = tmp_path / "responses-state.json"
    original_replace = Path.replace
    failed_once = False

    def fail_first_replace(path: Path, target: Path) -> Path:
        nonlocal failed_once
        if not failed_once and path.name == f".{state_file.name}.tmp":
            failed_once = True
            raise OSError("simulated state storage failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_first_replace)
    app = _app(response_state_file=state_file, max_pending_responses=1)

    with TestClient(app, raise_server_exceptions=False) as client:
        failed = client.post("/responses", json={"input": "please read telemetry"})
        retried = client.post("/responses", json={"input": "please read telemetry"})

    assert failed.status_code == 503
    assert failed.json()["error"] == {
        "message": "brokered response state storage unavailable",
        "code": "brokered_response_state_storage_error",
    }
    assert retried.status_code == 200, retried.text
    assert _call(retried.json())


def test_foundry_brokered_state_transactions_do_not_deepcopy_existing_state_graph(monkeypatch):
    app = _app(max_pending_responses=3)
    original_deepcopy = foundry_module.deepcopy

    def reject_full_store_deepcopy(value: Any, memo: dict[int, Any] | None = None) -> Any:
        if isinstance(value, dict) and value and all(type(entry).__name__ == "_HostedResponseState" for entry in value.values()):
            raise AssertionError("state transactions must not deepcopy the entire state store")
        return original_deepcopy(value, memo) if memo is not None else original_deepcopy(value)

    with TestClient(app, raise_server_exceptions=False) as client:
        first = _start(client)
        first_call = _call(first)
        monkeypatch.setattr(foundry_module, "deepcopy", reject_full_store_deepcopy)
        second = client.post("/responses", json={"input": "please read telemetry again"})
        completed = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_continuation(first["id"], first_call["call_id"], {"approved": True, "output": {"ok": True}}),
        )

    assert second.status_code == 200, second.text
    assert completed.status_code == 200, completed.text


def test_foundry_bounded_json_walk_rejects_wide_mapping_without_bulk_child_copy():
    class WideMapping(dict[str, int]):
        def __init__(self) -> None:
            super().__init__()
            self.item_yields = 0

        def keys(self):
            raise AssertionError("bounded traversal must not bulk-copy mapping keys")

        def values(self):
            raise AssertionError("bounded traversal must not bulk-copy mapping values")

        def items(self):
            for index in range(10_000):
                self.item_yields += 1
                yield str(index), index

    value = WideMapping()
    try:
        foundry_module._bounded_json_bytes(value, max_bytes=128)
    except foundry_module._SerializedPayloadTooLarge:
        pass
    else:
        raise AssertionError("wide mapping must exceed the bounded JSON budget")

    assert value.item_yields <= 65


def test_foundry_bounded_json_rejects_escaped_string_before_encoder(monkeypatch):
    def reject_encoder(*_args: Any, **_kwargs: Any):
        raise AssertionError("oversized escaped strings must be rejected before JSONEncoder")

    monkeypatch.setattr(json.JSONEncoder, "iterencode", reject_encoder)
    try:
        foundry_module._bounded_json_bytes("é" * 20, max_bytes=64)
    except foundry_module._SerializedPayloadTooLarge:
        pass
    else:
        raise AssertionError("escaped JSON string must exceed the byte budget")


def test_foundry_state_byte_eviction_serializes_aggregate_once():
    tool = foundry_module.brokered_tool_definitions(_spec())[0]

    def state(response_id: str, *, completed: bool) -> Any:
        call_id = f"call_{response_id}"
        call = foundry_module._PendingCall(
            call_id=call_id,
            item_id=f"item_{response_id}",
            tool=tool,
            arguments={},
        )
        return foundry_module._HostedResponseState(
            response_id=response_id,
            session_id=None,
            pending_calls={call_id: call},
            expires_at=time.time() + 60,
            status="completed" if completed else "pending",
            accepted_output_digests={call_id: "sha256:" + "0" * 64} if completed else {},
            final_payload={"result": "x" * 200} if completed else None,
            model_messages=[{"role": "user", "content": "y" * 500}] if not completed else None,
        )

    candidate = state("candidate", completed=False)
    states = {
        "completed-1": state("completed-1", completed=True),
        "completed-2": state("completed-2", completed=True),
        "completed-3": state("completed-3", completed=True),
        candidate.response_id: candidate,
    }
    store = foundry_module._FoundryResponseStateStore(
        ttl_seconds=60,
        max_entries=10,
        max_bytes=1_000_000,
    )
    store.max_bytes = max(
        len(store._serialize({response_id: states[response_id], candidate.response_id: candidate}))
        for response_id in ("completed-1", "completed-2", "completed-3")
    )
    original_serialize = store._serialize
    aggregate_serializations = 0

    def count_aggregate(current_states: dict[str, Any]) -> bytes:
        nonlocal aggregate_serializations
        if len(current_states) > 1:
            aggregate_serializations += 1
        return original_serialize(current_states)

    store._serialize = count_aggregate  # type: ignore[method-assign]
    store._serialize_candidate(
        states,
        response_id=candidate.response_id,
        excluded_response_ids=set(),
    )

    assert aggregate_serializations <= 1
    assert candidate.response_id in states
    assert len(states) == 2


def test_foundry_state_cache_fallback_contains_capacity_failure(monkeypatch):
    stores: list[Any] = []
    original_store = foundry_module._FoundryResponseStateStore

    def capture_store(*args: Any, **kwargs: Any) -> Any:
        store = original_store(*args, **kwargs)
        stores.append(store)
        return store

    monkeypatch.setattr(foundry_module, "_FoundryResponseStateStore", capture_store)
    app = _app()
    with TestClient(app) as client:
        initial = _start(client)

    state = stores[0].get(initial["id"])

    def reject_cache(*_args: Any, **_kwargs: Any):
        raise foundry_module._StateStoreFull("simulated concurrent capacity pressure")

    monkeypatch.setattr(stores[0], "_serialize_candidate", reject_cache)
    assert stores[0].cache_in_memory(state) is False


def test_foundry_brokered_completed_state_is_evicted_before_rejecting_new_pending_state():
    app = _app(max_pending_responses=1)

    with TestClient(app) as client:
        initial = _start(client)
        call = _call(initial)
        completed = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_continuation(initial["id"], call["call_id"], {"approved": True, "output": {"success": True}}),
        )
        next_initial = client.post("/responses", json={"input": "another pending request"})

    assert completed.status_code == 200, completed.text
    assert next_initial.status_code == 200, next_initial.text
    assert _call(next_initial.json())


def test_foundry_brokered_arguments_honor_typeless_const_and_enum_values():
    spec = _spec(tool_name="typeless-constraints")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {
            "probe": {"const": True},
            "mode": {"enum": ["safe"]},
        },
        "required": ["probe", "mode"],
    }
    app = _app(spec)

    with TestClient(app) as client:
        body = client.post("/responses", json={"input": "typeless-constraints"})

    assert body.status_code == 200, body.text
    assert json.loads(_call(body.json())["arguments"]) == {"probe": True, "mode": "safe"}


def test_foundry_brokered_min_properties_runs_dependent_required_closure():
    spec = _spec(tool_name="minprops-dependent")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {
            "a": {"type": "string", "const": "a"},
            "b": {"type": "string", "const": "b"},
        },
        "minProperties": 1,
        "dependentRequired": {"a": ["b"]},
    }
    app = _app(spec)

    with TestClient(app) as client:
        body = client.post("/responses", json={"input": "minprops-dependent"})

    assert body.status_code == 200, body.text
    assert json.loads(_call(body.json())["arguments"]) == {"a": "a", "b": "b"}


def test_foundry_brokered_arguments_honor_dependent_required_and_null_type():
    spec = _spec(tool_name="dependent-null")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {
            "region": {"type": "string", "const": "west"},
            "site": {"type": "string", "const": "sfo"},
            "marker": {"type": "null"},
        },
        "required": ["region", "marker"],
        "dependentRequired": {"region": ["site"]},
    }
    app = _app(spec)

    with TestClient(app) as client:
        body = client.post("/responses", json={"input": "dependent-null"})

    assert body.status_code == 200, body.text
    assert json.loads(_call(body.json())["arguments"]) == {"region": "west", "site": "sfo", "marker": None}


def test_foundry_brokered_integer_arguments_honor_float_bounds():
    spec = _spec(tool_name="bounded-integer")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {
            "count": {"type": "integer", "minimum": 1.0},
            "after": {"type": "integer", "exclusiveMinimum": 0.5},
            "below": {"type": "integer", "exclusiveMaximum": 4.5},
        },
        "required": ["count", "after", "below"],
    }
    app = _app(spec)

    with TestClient(app) as client:
        body = client.post("/responses", json={"input": "bounded-integer"})

    assert body.status_code == 200, body.text
    assert json.loads(_call(body.json())["arguments"]) == {"count": 1, "after": 1, "below": 0}


def test_foundry_brokered_synthesizes_arguments_that_honor_basic_constraints():
    spec = _spec(tool_name="bounded-lookup")
    tool = spec.brokered_tools[0]
    tool.parameters = {
        "type": "object",
        "properties": {
            "count": {"type": "integer", "minimum": 1},
            "mode": {"type": "string", "enum": ["safe"]},
            "label": {"type": "string", "minLength": 5, "maxLength": 7},
        },
        "required": ["count", "mode", "label"],
    }
    app = _app(spec)

    with TestClient(app) as client:
        body = _start(client, "please call bounded-lookup")

    assert json.loads(_call(body)["arguments"]) == {"count": 1, "mode": "safe", "label": "label"}


def test_foundry_brokered_rejects_schema_constraints_it_cannot_synthesize():
    spec = _spec(tool_name="pattern-lookup")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {"site": {"type": "string", "pattern": "^[A-Z]+$"}},
        "required": ["site"],
    }
    app = _app(spec)

    with TestClient(app) as client:
        resp = client.post("/responses", json={"input": "pattern-lookup sfo"})

    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "UnsupportedBrokeredSchema"


def test_foundry_brokered_numeric_argument_synthesis_respects_upper_bounds():
    spec = _spec(tool_name="check-network-telemetry")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {
            "count": {"type": "integer", "maximum": -1},
            "ratio": {"type": "number", "exclusiveMaximum": 0},
        },
        "required": ["count", "ratio"],
    }
    app = _app(spec)

    with TestClient(app) as client:
        body = client.post("/responses", json={"input": "check-network-telemetry"}).json()

    assert json.loads(_call(body)["arguments"]) == {"count": -1, "ratio": -1}


def test_foundry_brokered_optional_prompt_argument_is_synthesized_through_schema():
    spec = _spec(tool_name="prompt-tool")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {"prompt": {"type": "string", "const": "fixed"}},
    }
    app = _app(spec)

    with TestClient(app) as client:
        body = client.post("/responses", json={"input": "please call prompt-tool with arbitrary text"}).json()

    assert json.loads(_call(body)["arguments"]) == {"prompt": "fixed"}


def test_foundry_brokered_deterministic_arguments_honor_root_const_object_schema():
    spec = _spec(tool_name="check-network-telemetry")
    spec.brokered_tools[0].parameters = {"type": "object", "const": {"site": "iad"}}
    app = _app(spec)

    with TestClient(app) as client:
        body = client.post("/responses", json={"input": "check-network-telemetry"}).json()

    assert json.loads(_call(body)["arguments"]) == {"site": "iad"}


def test_foundry_brokered_validates_deterministic_arguments_before_emitting_call():
    spec = _spec(tool_name="check-network-telemetry")
    spec.brokered_tools[0].parameters = {"type": "object", "required": ["site"], "additionalProperties": False}
    app = _app(spec)

    with TestClient(app) as client:
        resp = client.post("/responses", json={"input": "check-network-telemetry"})

    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "InvalidToolArguments"


def test_foundry_brokered_argument_synthesis_honors_dependent_required_and_min_properties():
    spec = _spec(tool_name="check-network-telemetry")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {
            "site": {"type": "string"},
            "region": {"type": "string", "default": "west"},
            "extra": {"type": "boolean"},
        },
        "required": ["site"],
        "dependentRequired": {"site": ["region"]},
        "minProperties": 3,
    }
    app = _app(spec)

    with TestClient(app) as client:
        body = client.post("/responses", json={"input": "check-network-telemetry"}).json()

    assert json.loads(_call(body)["arguments"]) == {"site": "check-network-telemetry", "region": "west", "extra": True}


def test_foundry_brokered_integer_synthesis_honors_multiple_of():
    spec = _spec(tool_name="check-network-telemetry")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {"n": {"type": "integer", "minimum": 1, "multipleOf": 2}},
        "required": ["n"],
    }
    app = _app(spec)

    with TestClient(app) as client:
        body = client.post("/responses", json={"input": "check-network-telemetry"}).json()

    assert json.loads(_call(body)["arguments"]) == {"n": 2}


def test_foundry_brokered_number_synthesis_uses_midpoint_for_fractional_exclusive_range():
    spec = _spec(tool_name="check-network-telemetry")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {"ratio": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1}},
        "required": ["ratio"],
    }
    app = _app(spec)

    with TestClient(app) as client:
        body = client.post("/responses", json={"input": "check-network-telemetry"}).json()

    assert json.loads(_call(body)["arguments"]) == {"ratio": 0.5}


def test_foundry_brokered_rejects_unbounded_schema_synthesis_before_allocating():
    spec = _spec(tool_name="check-network-telemetry")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {"site": {"type": "string", "minLength": 1_000_000_000}},
        "required": ["site"],
    }
    app = _app(spec)

    with TestClient(app) as client:
        resp = client.post("/responses", json={"input": "check-network-telemetry"})

    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "brokered_arguments_too_large"


def test_foundry_brokered_tool_selection_requires_token_boundary_match():
    data = _multi_tool_spec().model_dump(by_alias=True)
    data["brokeredTools"] = [
        {"name": "read", "description": "read", "brokeredClass": "read", "parameters": {"type": "object"}},
        {"name": "read_telemetry", "description": "read telemetry", "brokeredClass": "read", "parameters": {"type": "object"}},
    ]
    app = _app(AgentSpec.model_validate(data))

    with TestClient(app) as client:
        telemetry = client.post("/responses", json={"input": "please call read_telemetry"})
        unrelated = client.post("/responses", json={"input": "already done"})

    assert telemetry.status_code == 200, telemetry.text
    assert _call(telemetry.json())["name"] == "read_telemetry"
    assert unrelated.status_code == 400
    assert unrelated.json()["error"]["code"] == "brokered_tool_selection_required"


def test_foundry_brokered_rejects_schema_that_would_generate_huge_arguments_before_allocating():
    spec = _spec(tool_name="huge-args")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {
            "items": {"type": "array", "minItems": 1000000000, "items": {"type": "string"}},
            "label": {"type": "string", "minLength": 1000000000},
        },
        "required": ["items", "label"],
    }
    app = _app(spec)

    with TestClient(app) as client:
        resp = client.post("/responses", json={"input": "huge-args"})

    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "brokered_arguments_too_large"


def test_foundry_brokered_rejects_arguments_that_exceed_state_budget():
    spec = _spec(tool_name="check-network-telemetry")
    spec.brokered_tools[0].parameters["required"] = ["site"]
    app = _app(spec, max_brokered_argument_bytes=8)

    with TestClient(app) as client:
        resp = client.post("/responses", json={"input": "check-network-telemetry: a prompt that is too large for the site argument"})

    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "brokered_arguments_too_large"


def test_foundry_brokered_single_read_tool_requires_explicit_name_except_conformance():
    app = _app(_spec(tool_name="check-network-telemetry"))

    with TestClient(app) as client:
        unrelated = client.post("/responses", json={"input": "my password should stay in chat"})
        explicit = client.post("/responses", json={"input": "please call check-network-telemetry"})

    assert unrelated.status_code == 400
    assert unrelated.json()["error"]["code"] == "brokered_tool_selection_required"
    assert explicit.status_code == 200, explicit.text
    assert _call(explicit.json())["name"] == "check-network-telemetry"


def test_foundry_brokered_single_write_tool_requires_explicit_tool_name():
    app = _app(_spec(tool_name="dispatch-work-order", brokered_class="write"))

    with TestClient(app) as client:
        resp = client.post("/responses", json={"input": "hello"})
        explicit = client.post("/responses", json={"input": "please call dispatch-work-order"})

    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "brokered_tool_selection_required"
    assert explicit.status_code == 200, explicit.text
    assert _call(explicit.json())["name"] == "dispatch-work-order"


def test_foundry_brokered_selects_named_tool_when_multiple_schemas_are_configured():
    app = _app(_multi_tool_spec())

    with TestClient(app) as client:
        resp = client.post("/responses", json={"input": "please call get-active-incidents"})

    assert resp.status_code == 200, resp.text
    assert _call(resp.json())["name"] == "get-active-incidents"


def test_foundry_brokered_rejects_ambiguous_multi_tool_prompt():
    app = _app(_multi_tool_spec())

    with TestClient(app) as client:
        resp = client.post("/responses", json={"input": "please inspect the network"})

    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "brokered_tool_selection_required"


def test_foundry_brokered_rejects_nonfinite_function_call_output_values():
    app = _app()

    with TestClient(app) as client:
        initial = _start(client)
        call = _call(initial)
        response = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json={
                "previous_response_id": initial["id"],
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": call["call_id"],
                        "output": '{"approved":true,"output":{"value":NaN}}',
                    }
                ],
            },
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_function_call_output"


def test_foundry_nonfinite_validation_traverses_wide_lists_lazily():
    class ExplodingTailList(list):
        def __iter__(self):
            yield float("nan")
            raise AssertionError("non-finite validation eagerly consumed the full list")

    with pytest.raises(ValueError, match="must be finite"):
        foundry_module._reject_nonfinite_json_values(
            {"result": ExplodingTailList([0, 1])},
            path="function_call_output.output",
        )


def test_foundry_nonfinite_validation_binds_sibling_paths():
    with pytest.raises(ValueError, match=r"function_call_output\.output\.items\[1\] must be finite"):
        foundry_module._reject_nonfinite_json_values(
            {"items": [0.0, float("inf")]},
            path="function_call_output.output",
        )


def test_foundry_request_ceiling_ignores_expired_replay_sizes():
    store = foundry_module._FoundryResponseStateStore(
        ttl_seconds=60,
        max_entries=2,
        max_bytes=4 * 1024,
    )
    store._states["expired"] = foundry_module._HostedResponseState(
        response_id="expired",
        session_id=None,
        pending_calls={},
        expires_at=time.time() - 1,
        status="completed",
        accepted_output_digests={"call-1": "sha256:" + "0" * 64},
        accepted_output_sizes={"call-1": 500_000},
        final_payload={"status": "completed"},
    )

    assert store.max_accepted_output_bytes() == 0


def test_foundry_brokered_rejects_function_output_with_lone_surrogate():
    app = _app()

    with TestClient(app) as client:
        initial = _start(client)
        call = _call(initial)
        payload = _continuation(initial["id"], call["call_id"], {"approved": True, "output": {}})
        payload["input"][0]["output"] = r'{"approved":true,"output":{"value":"\ud800"}}'
        rejected = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)
        accepted = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_continuation(initial["id"], call["call_id"], {"approved": True, "output": {"ok": True}}),
        )

    assert rejected.status_code == 400
    assert rejected.json()["error"]["code"] == "invalid_function_call_output"
    assert accepted.status_code == 200, accepted.text


def test_foundry_brokered_rejects_oversized_function_call_output_before_state_change(tmp_path):
    state_file = tmp_path / "responses-state.json"
    app = _app(response_state_file=state_file, max_brokered_output_bytes=128)

    with TestClient(app) as client:
        initial = _start(client)
        call = _call(initial)
        persisted_before = state_file.read_bytes()
        oversized = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_continuation(
                initial["id"],
                call["call_id"],
                {"approved": True, "output": {"blob": "x" * 256}},
            ),
        )
        persisted_after_rejection = state_file.read_bytes()
        accepted = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_continuation(initial["id"], call["call_id"], {"approved": True, "output": {"ok": True}}),
        )

    assert oversized.status_code == 413
    assert oversized.json()["error"] == {
        "message": "brokered function_call_output is too large",
        "code": "brokered_output_too_large",
    }
    assert persisted_after_rejection == persisted_before
    assert accepted.status_code == 200, accepted.text


def test_foundry_brokered_rejects_oversized_request_body_without_truncation_or_state_change(tmp_path):
    state_file = tmp_path / "responses-state.json"
    app = _app(
        response_state_file=state_file,
        max_brokered_output_bytes=128,
        max_response_state_bytes=4 * 1024,
    )

    with TestClient(app) as client:
        initial = _start(client)
        call = _call(initial)
        persisted_before = state_file.read_bytes()
        oversized_payload = _continuation(
            initial["id"],
            call["call_id"],
            {"approved": True, "output": {"ok": True}},
        )
        oversized_payload["ignored_padding"] = "x" * 100_000
        rejected = client.post(
            "/responses",
            headers={**CONTINUATION_AUTH, "content-type": "application/json"},
            content=json.dumps(oversized_payload, separators=(",", ":")),
        )
        persisted_after_rejection = state_file.read_bytes()
        accepted = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_continuation(initial["id"], call["call_id"], {"approved": True, "output": {"ok": True}}),
        )

    assert rejected.status_code == 413
    assert rejected.json()["error"] == {
        "message": "brokered Responses request body is too large",
        "code": "brokered_request_too_large",
    }
    assert persisted_after_rejection == persisted_before
    assert accepted.status_code == 200, accepted.text


def test_foundry_brokered_bounded_json_value_error_returns_400():
    app = _app(max_response_state_bytes=32 * 1024)
    oversized_integer = '{"input":' + '9' * 5_000 + '}'

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/responses",
            headers={"content-type": "application/json"},
            content=oversized_integer,
        )

    assert response.status_code == 400
    assert response.json()["error"] == {
        "message": "Request body must be JSON",
        "code": "invalid_json",
    }


def test_foundry_brokered_request_limit_allows_valid_reescaped_model_output():
    spec = _spec(tool_name="check-network-telemetry")
    fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "model-generated-call",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": "{}"},
                        }
                    ],
                }
            ),
            _chat_response({"role": "assistant", "content": "Large tool output accepted."}),
        ]
    )
    app = _model_loop_app(
        spec,
        fake,
        max_brokered_output_bytes=64 * 1024,
        max_response_state_bytes=96 * 1024,
    )
    message = "\x7f" * 10_800
    raw_output = '{"approved":false,"error":{"message":"' + message + '"}}'
    raw_output += "\n" * (64 * 1024 - len(raw_output) - 100)
    parsed_output = json.loads(raw_output)
    canonical_output = json.dumps(parsed_output, separators=(",", ":"), sort_keys=True)
    assert len(raw_output.encode("utf-8")) < 64 * 1024
    assert len(canonical_output.encode("utf-8")) < 64 * 1024

    with TestClient(app) as client:
        initial = client.post("/responses", json={"input": "check-network-telemetry"})
        call = _call(initial.json())
        payload = _continuation(initial.json()["id"], call["call_id"], {"approved": False})
        payload["input"][0]["output"] = raw_output
        encoded_request = json.dumps(payload, separators=(",", ":"))
        assert len(encoded_request.encode("utf-8")) > 2 * 64 * 1024 + 16 * 1024
        final = client.post("/responses", headers=CONTINUATION_AUTH, content=encoded_request)

    assert final.status_code == 200, final.text
    assert _message_text(final.json()) == "Large tool output accepted."


def test_foundry_brokered_bounds_object_valued_function_call_output():
    app = _app(max_brokered_output_bytes=128)

    with TestClient(app) as client:
        initial = _start(client)
        call = _call(initial)
        payload = _continuation(initial["id"], call["call_id"], {"approved": True, "output": {}})
        payload["input"][0]["output"] = {"approved": True, "output": {"blob": "x" * 256}}
        rejected = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)
        accepted = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_continuation(initial["id"], call["call_id"], {"approved": True, "output": {"ok": True}}),
        )

    assert rejected.status_code == 413
    assert rejected.json()["error"]["code"] == "brokered_output_too_large"
    assert accepted.status_code == 200, accepted.text


def test_foundry_brokered_rejects_continuation_that_would_exceed_total_state_bytes(tmp_path):
    state_file = tmp_path / "responses-state.json"
    app = _app(
        response_state_file=state_file,
        max_brokered_output_bytes=8 * 1024,
        max_response_state_bytes=4 * 1024,
    )

    with TestClient(app) as client:
        initial = _start(client)
        call = _call(initial)
        persisted_before = state_file.read_bytes()
        oversized_state = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_continuation(
                initial["id"],
                call["call_id"],
                {"approved": True, "output": {"blob": "x" * 5_000}},
            ),
        )
        persisted_after_rejection = state_file.read_bytes()
        accepted = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_continuation(initial["id"], call["call_id"], {"approved": True, "output": {"ok": True}}),
        )

    assert oversized_state.status_code == 413
    assert oversized_state.json()["error"] == {
        "message": "brokered response state exceeds the configured byte limit",
        "code": "brokered_response_state_too_large",
    }
    assert persisted_after_rejection == persisted_before
    assert accepted.status_code == 200, accepted.text


def test_foundry_brokered_model_loop_counts_output_limit_in_utf8_bytes():
    spec = _spec(tool_name="check-network-telemetry")
    fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "model_generated_call_id",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": "{}"},
                        }
                    ],
                }
            ),
            _chat_response({"role": "assistant", "content": "Bounded resume worked."}),
        ]
    )
    app = _model_loop_app(spec, fake, max_brokered_output_bytes=80)

    with TestClient(app) as client:
        initial = client.post("/responses", json={"input": "check-network-telemetry"})
        call = _call(initial.json())
        oversized_payload = _continuation(initial.json()["id"], call["call_id"], {"approved": True, "output": {}})
        oversized_output = json.dumps(
            {"approved": True, "output": {"value": "é" * 21}},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        assert len(oversized_output) < 80 < len(oversized_output.encode("utf-8"))
        oversized_payload["input"][0]["output"] = oversized_output
        rejected = client.post("/responses", headers=CONTINUATION_AUTH, json=oversized_payload)
        accepted = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_continuation(initial.json()["id"], call["call_id"], {"approved": True, "output": {"ok": True}}),
        )

    assert rejected.status_code == 413
    assert rejected.json()["error"]["code"] == "brokered_output_too_large"
    assert accepted.status_code == 200, accepted.text
    assert _message_text(accepted.json()) == "Bounded resume worked."
    assert len(fake.requests) == 2


def test_foundry_brokered_continuation_accepts_matching_tool_output_and_completes():
    app = _app()

    with TestClient(app) as client:
        initial = _start(client)
        call = _call(initial)
        cont = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_continuation(initial["id"], call["call_id"], {"approved": True, "output": {"success": True}}),
        )

    assert cont.status_code == 200, cont.text
    final = cont.json()
    assert final["previous_response_id"] == initial["id"]
    assert final["id"].startswith("caresp_")
    assert final["id"] != initial["id"]
    assert _message_text(final) == 'Brokered tool conformance_read completed with output: {"success":true}'


def test_foundry_brokered_accepts_body_continuation_proof_without_header():
    app = _app()

    with TestClient(app) as client:
        initial = _start(client)
        call = _call(initial)
        payload = _continuation(initial["id"], call["call_id"], {"approved": True, "output": {"success": True}})
        payload[CONTINUATION_PROOF_BODY_FIELD] = CONTINUATION_PROOF
        response = client.post("/responses", json=payload)

    assert response.status_code == 200, response.text
    assert _message_text(response.json()) == 'Brokered tool conformance_read completed with output: {"success":true}'


def test_foundry_brokered_accepts_either_matching_proof_candidate():
    app = _app()

    with TestClient(app) as client:
        first_initial = _start(client)
        first_call = _call(first_initial)
        first_payload = _continuation(first_initial["id"], first_call["call_id"], {"approved": True, "output": {"success": True}})
        first_payload[CONTINUATION_PROOF_BODY_FIELD] = CONTINUATION_PROOF
        correct_body = client.post(
            "/responses",
            headers={"x-agentkit-brokered-continuation-proof": "wrong-header"},
            json=first_payload,
        )

        second_initial = _start(client)
        second_call = _call(second_initial)
        second_payload = _continuation(second_initial["id"], second_call["call_id"], {"approved": True, "output": {"success": True}})
        second_payload[CONTINUATION_PROOF_BODY_FIELD] = "wrong-body"
        correct_header = client.post("/responses", headers=CONTINUATION_AUTH, json=second_payload)

    assert correct_body.status_code == 200, correct_body.text
    assert correct_header.status_code == 200, correct_header.text


def test_foundry_brokered_rejects_wrong_or_non_string_body_continuation_proof():
    app = _app()

    with TestClient(app) as client:
        initial = _start(client)
        call = _call(initial)
        payload = _continuation(initial["id"], call["call_id"], {"approved": True, "output": {"success": True}})
        payload[CONTINUATION_PROOF_BODY_FIELD] = "wrong-proof"
        wrong = client.post("/responses", json=payload)
        payload[CONTINUATION_PROOF_BODY_FIELD] = 123
        non_string = client.post("/responses", json=payload)
        payload[CONTINUATION_PROOF_BODY_FIELD] = ""
        empty = client.post("/responses", json=payload)
        payload[CONTINUATION_PROOF_BODY_FIELD] = "wrong-💥"
        non_ascii = client.post("/responses", json=payload)
        payload[CONTINUATION_PROOF_BODY_FIELD] = "\ud800"
        unpaired_surrogate = client.post(
            "/responses",
            content=json.dumps(payload, ensure_ascii=True),
            headers={"content-type": "application/json"},
        )

    assert wrong.status_code == 403
    assert wrong.json()["error"]["code"] == "brokered_continuation_forbidden"
    assert non_string.status_code == 403
    assert non_string.json()["error"]["code"] == "brokered_continuation_forbidden"
    assert empty.status_code == 403
    assert empty.json()["error"]["code"] == "brokered_continuation_forbidden"
    assert non_ascii.status_code == 403
    assert non_ascii.json()["error"]["code"] == "brokered_continuation_forbidden"
    assert unpaired_surrogate.status_code == 403
    assert unpaired_surrogate.json()["error"]["code"] == "brokered_continuation_forbidden"


def test_foundry_brokered_rejects_function_call_output_without_orka_continuation_auth():
    app = _app()

    with TestClient(app) as client:
        initial = _start(client)
        call = _call(initial)
        resp = client.post(
            "/responses",
            json=_continuation(initial["id"], call["call_id"], {"approved": True, "output": {"success": True}}),
        )

    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == "brokered_continuation_forbidden"


def test_foundry_brokered_rejects_orphan_function_call_output_without_previous_response_id():
    app = _app()

    with TestClient(app) as client:
        resp = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json={
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": "call_missing",
                        "output": '{"approved":true,"output":{}}',
                    }
                ]
            },
        )

    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "missing_previous_response_id"


def test_foundry_brokered_rejects_unknown_previous_response_id():
    app = _app()

    with TestClient(app) as client:
        resp = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_continuation("caresp_unknown", "call_unknown", {"approved": True, "output": {}}),
        )

    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "unknown_previous_response_id"


def test_foundry_brokered_rejects_unknown_call_id():
    app = _app()

    with TestClient(app) as client:
        initial = _start(client)
        resp = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_continuation(initial["id"], "call_other", {"approved": True, "output": {}}),
        )

    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "unknown_call_id"


def test_foundry_brokered_duplicate_continuation_is_idempotent_but_conflicts_are_rejected():
    app = _app()

    with TestClient(app) as client:
        initial = _start(client)
        call = _call(initial)
        payload = _continuation(initial["id"], call["call_id"], {"approved": True, "output": {"success": True}})
        first = client.post("/responses", json=payload, headers=CONTINUATION_AUTH)
        duplicate = client.post("/responses", json=payload, headers=CONTINUATION_AUTH)
        conflicting_payload = deepcopy(payload)
        conflicting_payload["input"][0]["output"] = '{"approved":true,"output":{"success":false}}'
        conflict = client.post("/responses", json=conflicting_payload, headers=CONTINUATION_AUTH)

    assert first.status_code == 200
    assert duplicate.status_code == 200
    assert duplicate.json() == first.json()
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "conflicting_duplicate_continuation"


def test_foundry_brokered_persists_one_output_copy_while_preserving_idempotency(tmp_path):
    state_file = tmp_path / "foundry-state.json"
    marker = "unique-brokered-output-marker"
    app = _app(response_state_file=state_file)

    with TestClient(app) as client:
        initial = _start(client)
        call = _call(initial)
        payload = _continuation(
            initial["id"],
            call["call_id"],
            {"approved": True, "output": {"value": marker}},
        )
        completed = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)
        duplicate = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)
        conflicting = deepcopy(payload)
        conflicting["input"][0]["output"] = '{"approved":true,"output":{"value":"different"}}'
        conflict = client.post("/responses", headers=CONTINUATION_AUTH, json=conflicting)

    assert completed.status_code == 200, completed.text
    assert duplicate.json() == completed.json()
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "conflicting_duplicate_continuation"
    assert state_file.read_text(encoding="utf-8").count(marker) == 1


def test_foundry_brokered_file_state_survives_restart_for_deterministic_continuation(tmp_path):
    state_file = tmp_path / "foundry-state.json"

    with TestClient(_app(response_state_file=state_file)) as client:
        initial = _start(client)
        call = _call(initial)
        readiness = client.get("/readiness")

    assert readiness.json()["foundryResponses"]["stateBackend"] == "file"
    assert state_file.exists()

    with TestClient(_app(response_state_file=state_file)) as client:
        final = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_continuation(initial["id"], call["call_id"], {"approved": True, "output": {"success": True}}),
        )

    assert final.status_code == 200, final.text
    assert _message_text(final.json()) == 'Brokered tool conformance_read completed with output: {"success":true}'

    with TestClient(_app(response_state_file=state_file)) as client:
        duplicate = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_continuation(initial["id"], call["call_id"], {"approved": True, "output": {"success": True}}),
        )

    assert duplicate.status_code == 200
    assert duplicate.json() == final.json()


def test_foundry_brokered_loads_legacy_full_output_state_for_idempotent_replay(tmp_path):
    state_file = tmp_path / "foundry-legacy-state.json"

    with TestClient(_app(response_state_file=state_file)) as client:
        initial = _start(client)
        call = _call(initial)
        payload = _continuation(initial["id"], call["call_id"], {"approved": True, "output": {"success": True}})
        completed = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)

    persisted = json.loads(state_file.read_text(encoding="utf-8"))
    state = persisted["states"][initial["id"]]
    state.pop("acceptedOutputDigests")
    state["acceptedOutputs"] = {call["call_id"]: payload["input"][0]["output"]}
    state_file.write_text(json.dumps(persisted, separators=(",", ":"), sort_keys=True), encoding="utf-8")

    with TestClient(_app(response_state_file=state_file)) as client:
        duplicate = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)

    assert completed.status_code == 200, completed.text
    assert duplicate.status_code == 200, duplicate.text
    assert duplicate.json() == completed.json()


def test_foundry_brokered_replays_legacy_output_larger_than_current_limit(tmp_path):
    state_file = tmp_path / "foundry-legacy-large-output-state.json"
    spec = _spec(tool_name="check-network-telemetry")
    large_output = {"approved": True, "output": {"blob": "x" * 80_000}}
    fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "model-generated-call",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": "{}"},
                        }
                    ],
                }
            ),
            _chat_response({"role": "assistant", "content": "Persisted large output replay."}),
        ]
    )
    with TestClient(
        _model_loop_app(
            spec,
            fake,
            response_state_file=state_file,
            max_brokered_output_bytes=128 * 1024,
        )
    ) as client:
        initial = client.post("/responses", json={"input": "check-network-telemetry"})
        call = _call(initial.json())
        payload = _continuation(initial.json()["id"], call["call_id"], large_output)
        completed = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)

    persisted = json.loads(state_file.read_text(encoding="utf-8"))
    state = persisted["states"][initial.json()["id"]]
    state.pop("acceptedOutputDigests")
    state["acceptedOutputs"] = {call["call_id"]: payload["input"][0]["output"]}
    state_file.write_text(json.dumps(persisted, separators=(",", ":"), sort_keys=True), encoding="utf-8")

    with TestClient(
        _app(
            spec,
            response_state_file=state_file,
            max_brokered_output_bytes=64 * 1024,
        )
    ) as client:
        duplicate = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)

    assert completed.status_code == 200, completed.text
    assert duplicate.status_code == 200, duplicate.text
    assert duplicate.json() == completed.json()


def test_foundry_brokered_replays_digest_output_larger_than_current_and_state_limits(tmp_path):
    state_file = tmp_path / "foundry-digest-large-output-state.json"
    spec = _spec(tool_name="check-network-telemetry")
    message = "\x7f" * 10_800
    raw_output = '{"approved":false,"error":{"message":"' + message + '"}}'
    raw_output += "\n" * (400 * 1024 - len(raw_output) - 100)
    parsed_output = json.loads(raw_output)
    canonical_output = json.dumps(parsed_output, separators=(",", ":"), sort_keys=True)
    assert len(raw_output.encode("utf-8")) < 512 * 1024
    assert len(canonical_output.encode("utf-8")) < 128 * 1024
    fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "model-generated-call",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": "{}"},
                        }
                    ],
                }
            ),
            _chat_response({"role": "assistant", "content": "Persisted digest replay."}),
        ]
    )
    with TestClient(
        _model_loop_app(
            spec,
            fake,
            response_state_file=state_file,
            max_brokered_output_bytes=512 * 1024,
            max_response_state_bytes=128 * 1024,
        )
    ) as client:
        initial = client.post("/responses", json={"input": "check-network-telemetry"})
        call = _call(initial.json())
        payload = _continuation(initial.json()["id"], call["call_id"], {"approved": False})
        payload["input"][0]["output"] = raw_output
        completed = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)

    persisted = json.loads(state_file.read_text(encoding="utf-8"))
    stored = persisted["states"][initial.json()["id"]]
    assert stored["acceptedOutputSizes"][call["call_id"]] == len(raw_output.encode("utf-8"))
    assert state_file.stat().st_size < 128 * 1024

    with TestClient(
        _app(
            spec,
            response_state_file=state_file,
            max_brokered_output_bytes=64 * 1024,
            max_response_state_bytes=128 * 1024,
        )
    ) as client:
        encoded_request = json.dumps(payload, separators=(",", ":"))
        assert len(encoded_request.encode("utf-8")) > 6 * 128 * 1024 + 16 * 1024
        duplicate = client.post("/responses", headers=CONTINUATION_AUTH, content=encoded_request)

    assert completed.status_code == 200, completed.text
    assert duplicate.status_code == 200, duplicate.text
    assert duplicate.json() == completed.json()


def test_foundry_brokered_rejects_legacy_state_that_expands_past_byte_limit(tmp_path):
    state_file = tmp_path / "foundry-legacy-state.json"

    with TestClient(_app(response_state_file=state_file)) as client:
        initial = _start(client)
        call = _call(initial)
        payload = _continuation(initial["id"], call["call_id"], {"approved": False})
        completed = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)

    persisted = json.loads(state_file.read_text(encoding="utf-8"))
    state = persisted["states"][initial["id"]]
    state.pop("acceptedOutputDigests")
    state["acceptedOutputs"] = {call["call_id"]: payload["input"][0]["output"]}
    state_file.write_text(json.dumps(persisted, separators=(",", ":"), sort_keys=True), encoding="utf-8")
    legacy_bytes = state_file.stat().st_size

    with TestClient(_app(response_state_file=state_file, max_response_state_bytes=legacy_bytes)) as client:
        duplicate = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)

    assert completed.status_code == 200, completed.text
    assert duplicate.status_code == 404
    assert duplicate.json()["error"]["code"] == "unknown_previous_response_id"


def test_foundry_brokered_rejects_caller_session_conflicts_with_trusted_gateway_header(monkeypatch):
    monkeypatch.delenv("FOUNDRY_AGENT_SESSION_ID", raising=False)
    app = _app()

    with TestClient(app) as client:
        body_conflict = client.post(
            "/responses",
            headers={"x-agent-session-id": "trusted-session"},
            json={"input": "please read telemetry", "agent_session_id": "caller-session"},
        )
        query_conflict = client.post(
            "/responses?session_id=caller-session",
            headers={"x-agent-session-id": "trusted-session"},
            json={"input": "please read telemetry"},
        )

    for response in (body_conflict, query_conflict):
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "response_session_mismatch"


def test_foundry_brokered_rejects_session_conflicts_with_hosted_environment(monkeypatch):
    monkeypatch.setenv("FOUNDRY_AGENT_SESSION_ID", "hosted-session")
    app = _app()

    with TestClient(app) as client:
        gateway_conflict = client.post(
            "/responses",
            headers={"x-agent-session-id": "other-session"},
            json={"input": "please read telemetry"},
        )
        caller_conflict = client.post(
            "/responses",
            json={"input": "please read telemetry", "agent_session_id": "other-session"},
        )
        matching = client.post(
            "/responses?session_id=hosted-session",
            headers={"x-agent-session-id": "hosted-session"},
            json={"input": "please read telemetry", "agent_session_id": "hosted-session"},
        )

    for response in (gateway_conflict, caller_conflict):
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "response_session_mismatch"
    assert matching.status_code == 200, matching.text


def test_foundry_brokered_trusted_gateway_session_accepts_matching_local_compatibility(monkeypatch, tmp_path):
    state_file = tmp_path / "foundry-trusted-session-state.json"
    monkeypatch.delenv("FOUNDRY_AGENT_SESSION_ID", raising=False)

    with TestClient(_app(response_state_file=state_file)) as client:
        response = client.post(
            "/responses?session_id=gateway-session",
            headers={
                "x-agent-session-id": "gateway-session",
                "x-agentkit-session-id": "gateway-session",
            },
            json={"input": "please read telemetry", "agent_session_id": "gateway-session"},
        )

    assert response.status_code == 200, response.text
    stored = json.loads(state_file.read_text(encoding="utf-8"))["states"][response.json()["id"]]
    assert stored["sessionID"] == "gateway-session"


def test_foundry_brokered_preserves_local_session_fallback_precedence_without_trusted_identity(monkeypatch, tmp_path):
    state_file = tmp_path / "foundry-local-session-state.json"
    monkeypatch.delenv("FOUNDRY_AGENT_SESSION_ID", raising=False)

    with TestClient(_app(response_state_file=state_file)) as client:
        response = client.post(
            "/responses?agent_session_id=query-session",
            headers={"x-agentkit-session-id": "legacy-header-session"},
            json={
                "input": "please read telemetry",
                "agent_session_id": "body-session",
                "session_id": "legacy-body-session",
            },
        )

    assert response.status_code == 200, response.text
    stored = json.loads(state_file.read_text(encoding="utf-8"))["states"][response.json()["id"]]
    assert stored["sessionID"] == "body-session"


def test_foundry_brokered_file_state_tracks_session_and_rejects_cross_session_continuation(monkeypatch, tmp_path):
    state_file = tmp_path / "foundry-session-state.json"
    monkeypatch.delenv("FOUNDRY_AGENT_SESSION_ID", raising=False)

    with TestClient(_app(response_state_file=state_file)) as client:
        initial_response = client.post(
            "/responses",
            json={"input": "please read telemetry", "agent_session_id": "session-a"},
        )
        assert initial_response.status_code == 200, initial_response.text
        initial = initial_response.json()
        call = _call(initial)

    stored = json.loads(state_file.read_text(encoding="utf-8"))["states"][initial["id"]]
    assert stored["sessionID"] == "session-a"

    payload = _continuation(initial["id"], call["call_id"], {"approved": True, "output": {"success": True}})
    payload[CONTINUATION_PROOF_BODY_FIELD] = CONTINUATION_PROOF
    with TestClient(_app(response_state_file=state_file)) as client:
        missing = client.post("/responses", json=payload)

    assert missing.status_code == 409
    assert missing.json()["error"]["code"] == "response_session_mismatch"

    payload["agent_session_id"] = "session-b"
    with TestClient(_app(response_state_file=state_file)) as client:
        mismatch = client.post("/responses", json=payload)

    assert mismatch.status_code == 409
    assert mismatch.json()["error"]["code"] == "response_session_mismatch"

    payload["agent_session_id"] = "session-a"
    with TestClient(_app(response_state_file=state_file)) as client:
        completed = client.post("/responses", json=payload)

    assert completed.status_code == 200, completed.text


def test_foundry_brokered_body_proof_is_not_persisted_or_echoed(tmp_path):
    state_file = tmp_path / "foundry-proof-state.json"
    app = _app(response_state_file=state_file)

    with TestClient(app) as client:
        initial_response = client.post(
            "/responses",
            json={"input": "please read telemetry", CONTINUATION_PROOF_BODY_FIELD: CONTINUATION_PROOF},
        )
        assert initial_response.status_code == 200, initial_response.text
        initial = initial_response.json()
        call = _call(initial)
        payload = _continuation(initial["id"], call["call_id"], {"approved": True, "output": {"success": True}})
        payload[CONTINUATION_PROOF_BODY_FIELD] = CONTINUATION_PROOF
        final_response = client.post("/responses", json=payload)

    assert final_response.status_code == 200, final_response.text
    assert CONTINUATION_PROOF not in json.dumps(initial, sort_keys=True)
    assert CONTINUATION_PROOF not in json.dumps(final_response.json(), sort_keys=True)
    assert CONTINUATION_PROOF not in state_file.read_text(encoding="utf-8")


def test_foundry_brokered_file_state_survives_restart_for_model_loop_continuation(tmp_path):
    state_file = tmp_path / "foundry-model-state.json"
    spec = _spec(tool_name="check-network-telemetry")
    spec.brokered_tools[0].parameters["required"] = ["site"]
    first_fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_model",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": '{"site":"sfo"}'},
                        }
                    ],
                }
            )
        ]
    )

    with TestClient(_model_loop_app(spec, first_fake, response_state_file=state_file)) as client:
        initial = client.post("/responses", json={"input": "call check-network-telemetry"}).json()
        call = _call(initial)

    second_fake = _FakeChatTransport([_chat_response({"role": "assistant", "content": "Restart resume worked."})])
    with TestClient(_model_loop_app(spec, second_fake, response_state_file=state_file)) as client:
        final = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_continuation(initial["id"], call["call_id"], {"approved": True, "output": {"status": "ok"}}),
        )

    assert final.status_code == 200, final.text
    assert _message_text(final.json()) == "Restart resume worked."
    assert second_fake.requests[0]["messages"][-1]["tool_call_id"] == call["call_id"]


def test_foundry_brokered_completion_drops_stored_transcript_when_model_loop_is_disabled(tmp_path):
    state_file = tmp_path / "foundry-model-state.json"
    spec = _spec(tool_name="check-network-telemetry")
    first_fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_model",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": "{}"},
                        }
                    ],
                }
            )
        ]
    )

    with TestClient(_model_loop_app(spec, first_fake, response_state_file=state_file)) as client:
        initial = client.post("/responses", json={"input": "check-network-telemetry"}).json()
        call = _call(initial)

    with TestClient(_app(spec, response_state_file=state_file, brokered_model_loop_enabled=False)) as client:
        completed = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_continuation(initial["id"], call["call_id"], {"approved": True, "output": {"status": "ok"}}),
        )

    assert completed.status_code == 200, completed.text
    persisted = json.loads(state_file.read_text(encoding="utf-8"))["states"][initial["id"]]
    assert persisted["modelMessages"] is None
    assert persisted["initialUsage"] == {}


def test_foundry_brokered_rejects_expired_response_state():
    app = _app(state_ttl_seconds=0)

    with TestClient(app) as client:
        initial = _start(client)
        call = _call(initial)
        time.sleep(0.01)
        resp = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_continuation(initial["id"], call["call_id"], {"approved": True, "output": {}}),
        )

    assert resp.status_code == 410
    assert resp.json()["error"]["code"] == "response_state_expired"


def test_foundry_brokered_refuses_to_synthesize_nonliteral_write_arguments():
    spec = _spec(tool_name="dispatch-work-order", brokered_class="write")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {"incident": {"type": "string"}},
        "required": ["incident"],
    }
    app = _app(spec)

    with TestClient(app) as client:
        resp = client.post("/responses", json={"input": "dispatch-work-order"})

    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "UnsupportedBrokeredSchema"


def test_foundry_brokered_refuses_multi_value_enum_write_arguments():
    spec = _spec(tool_name="dispatch-work-order", brokered_class="write")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {"operation": {"type": "string", "enum": ["delete", "create"]}},
        "required": ["operation"],
    }
    app = _app(spec)

    with TestClient(app) as client:
        resp = client.post("/responses", json={"input": "dispatch-work-order"})

    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "UnsupportedBrokeredSchema"


def test_foundry_brokered_allows_single_value_enum_write_arguments():
    spec = _spec(tool_name="dispatch-work-order", brokered_class="write")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {"operation": {"type": "string", "enum": ["create"]}},
        "required": ["operation"],
    }
    app = _app(spec)

    with TestClient(app) as client:
        resp = client.post("/responses", json={"input": "dispatch-work-order"})

    assert resp.status_code == 200
    assert json.loads(_call(resp.json())["arguments"]) == {"operation": "create"}


def test_foundry_brokered_allows_literal_write_arguments_for_conformance():
    spec = _spec(tool_name="dispatch-work-order", brokered_class="write")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {"incident": {"type": "string", "const": "INC-1"}},
        "required": ["incident"],
    }
    app = _app(spec)

    with TestClient(app) as client:
        body = client.post("/responses", json={"input": "dispatch-work-order"})

    assert body.status_code == 200, body.text
    assert json.loads(_call(body.json())["arguments"]) == {"incident": "INC-1"}


def test_foundry_brokered_decline_policy_rejection_and_execution_error_are_truthful_final_answers():
    cases = [
        (
            {"approved": False, "error": {"code": "approval_declined", "message": "Human declined dispatch-work-order"}},
            "approval_declined: Human declined dispatch-work-order",
        ),
        (
            {"approved": False, "error": {"code": "tool_policy_rejected", "message": "writes are disabled"}},
            "tool_policy_rejected: writes are disabled",
        ),
        (
            {"approved": False, "error": {"code": "tool_execution_failed", "message": "downstream timed out"}},
            "tool_execution_failed: downstream timed out",
        ),
    ]

    for payload, expected in cases:
        app = _app(_spec(tool_name="dispatch-work-order", brokered_class="write"))
        with TestClient(app) as client:
            initial = _start(client, "please call dispatch-work-order")
            call = _call(initial)
            resp = client.post("/responses", json=_continuation(initial["id"], call["call_id"], payload), headers=CONTINUATION_AUTH)
        assert resp.status_code == 200, resp.text
        assert _message_text(resp.json()) == f"Brokered tool dispatch-work-order was not performed: {expected}"


def test_foundry_brokered_rejects_multiple_tool_outputs_deterministically():
    app = _app()

    with TestClient(app) as client:
        initial = _start(client)
        call = _call(initial)
        request = _continuation(initial["id"], call["call_id"], {"approved": True, "output": {}})
        request["input"].append(dict(request["input"][0]))
        resp = client.post("/responses", json=request, headers=CONTINUATION_AUTH)

    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "multiple_tool_outputs_unsupported"



def test_foundry_brokered_disables_invocations_direct_runtime_bypass():
    app = _app()

    with TestClient(app) as client:
        resp = client.post("/invocations", json={"message": "bypass"})

    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invocations_disabled_in_brokered_mode"

class _FakeChatTransport:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content.decode("utf-8")))
        if not self.responses:
            return httpx.Response(500, json={"error": "unexpected extra model call"})
        return httpx.Response(200, json=self.responses.pop(0))


def _chat_response(message: dict[str, Any], *, prompt_tokens: int = 1, completion_tokens: int = 1) -> dict[str, Any]:
    return {
        "choices": [{"message": message}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _model_loop_app(spec: AgentSpec, fake: _FakeChatTransport, **kwargs: Any):
    client = httpx.AsyncClient(transport=httpx.MockTransport(fake.handler))
    return _app(spec, brokered_model_loop_enabled=True, brokered_model_http_client=client, **kwargs)


def test_foundry_brokered_max_pending_reserves_capacity_before_initial_model_work():
    spec = _spec(tool_name="check-network-telemetry")

    class BlockingInitialTransport(httpx.AsyncBaseTransport):
        def __init__(self) -> None:
            self.requests: list[dict[str, Any]] = []
            self.first_started = threading.Event()
            self.release_first = threading.Event()

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            self.requests.append(json.loads(request.content.decode("utf-8")))
            if len(self.requests) == 1:
                self.first_started.set()
                while not self.release_first.is_set():
                    await asyncio.sleep(0.001)
            return httpx.Response(
                200,
                request=request,
                json=_chat_response(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": f"model_call_{len(self.requests)}",
                                "type": "function",
                                "function": {"name": "check-network-telemetry", "arguments": "{}"},
                            }
                        ],
                    }
                ),
            )

    transport = BlockingInitialTransport()
    app = _app(
        spec,
        brokered_model_loop_enabled=True,
        brokered_model_http_client=httpx.AsyncClient(transport=transport),
        max_pending_responses=1,
    )

    with TestClient(app) as client, ThreadPoolExecutor(max_workers=1) as executor:
        first_future = executor.submit(client.post, "/responses", json={"input": "check-network-telemetry"})
        try:
            assert transport.first_started.wait(timeout=2)
            excess = client.post("/responses", json={"input": "check-network-telemetry again"})
        finally:
            transport.release_first.set()
        first = first_future.result(timeout=2)

    assert first.status_code == 200, first.text
    assert excess.status_code == 429
    assert excess.json()["error"]["code"] == "brokered_response_state_full"
    assert len(transport.requests) == 1


def test_foundry_brokered_initial_model_reservation_releases_for_final_and_error_paths():
    spec = _spec(tool_name="check-network-telemetry")
    fake = _FakeChatTransport(
        [
            _chat_response({"role": "assistant", "content": "No tool needed."}),
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "unknown_call",
                            "type": "function",
                            "function": {"name": "unknown-tool", "arguments": "{}"},
                        }
                    ],
                }
            ),
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "valid_call",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": "{}"},
                        }
                    ],
                }
            ),
        ]
    )
    app = _model_loop_app(spec, fake, max_pending_responses=1)

    with TestClient(app) as client:
        final = client.post("/responses", json={"input": "say hello"})
        invalid = client.post("/responses", json={"input": "request an unknown tool"})
        pending = client.post("/responses", json={"input": "check-network-telemetry"})

    assert final.status_code == 200, final.text
    assert _message_text(final.json()) == "No tool needed."
    assert invalid.status_code == 400
    assert invalid.json()["error"]["code"] == "unknown_brokered_tool"
    assert pending.status_code == 200, pending.text
    assert _call(pending.json())["name"] == "check-network-telemetry"
    assert len(fake.requests) == 3


def test_foundry_brokered_final_model_result_does_not_evict_completed_replay_state():
    spec = _spec(tool_name="check-network-telemetry")
    fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "initial_tool_call",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": "{}"},
                        }
                    ],
                }
            ),
            _chat_response({"role": "assistant", "content": "Stored completion."}),
            _chat_response({"role": "assistant", "content": "No pending state needed."}),
        ]
    )
    app = _model_loop_app(spec, fake, max_pending_responses=1)

    with TestClient(app) as client:
        initial = client.post("/responses", json={"input": "check-network-telemetry"})
        call = _call(initial.json())
        continuation = _continuation(initial.json()["id"], call["call_id"], {"approved": True, "output": {"ok": True}})
        completed = client.post("/responses", headers=CONTINUATION_AUTH, json=continuation)
        final_only = client.post("/responses", json={"input": "say hello without a tool"})
        replay = client.post("/responses", headers=CONTINUATION_AUTH, json=continuation)

    assert completed.status_code == 200, completed.text
    assert final_only.status_code == 200, final_only.text
    assert _message_text(final_only.json()) == "No pending state needed."
    assert replay.status_code == 200, replay.text
    assert replay.json() == completed.json()


def test_foundry_brokered_initial_model_reservation_releases_after_state_persist_failure(monkeypatch, tmp_path):
    state_file = tmp_path / "model-loop-responses-state.json"
    spec = _spec(tool_name="check-network-telemetry")
    fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "first_call",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": "{}"},
                        }
                    ],
                }
            ),
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "retry_call",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": "{}"},
                        }
                    ],
                }
            ),
        ]
    )
    app = _model_loop_app(
        spec,
        fake,
        response_state_file=state_file,
        max_pending_responses=1,
    )
    original_replace = Path.replace
    failed_once = False

    def fail_first_replace(path: Path, target: Path) -> Path:
        nonlocal failed_once
        if not failed_once and path.name == f".{state_file.name}.tmp":
            failed_once = True
            raise OSError("simulated model-loop state storage failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_first_replace)
    with TestClient(app, raise_server_exceptions=False) as client:
        failed = client.post("/responses", json={"input": "check-network-telemetry"})
        retried = client.post("/responses", json={"input": "check-network-telemetry"})

    assert failed.status_code == 503
    assert failed.json()["error"]["code"] == "brokered_response_state_storage_error"
    assert retried.status_code == 200, retried.text
    assert _call(retried.json())["name"] == "check-network-telemetry"
    assert len(fake.requests) == 2


def test_foundry_brokered_cancelled_initial_model_work_releases_reserved_capacity():
    spec = _spec(tool_name="check-network-telemetry")

    class CancellingInitialTransport(httpx.AsyncBaseTransport):
        def __init__(self) -> None:
            self.calls = 0

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            self.calls += 1
            if self.calls == 1:
                raise asyncio.CancelledError
            return httpx.Response(
                200,
                request=request,
                json=_chat_response(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "model_call_after_cancellation",
                                "type": "function",
                                "function": {"name": "check-network-telemetry", "arguments": "{}"},
                            }
                        ],
                    }
                ),
            )

    transport = CancellingInitialTransport()
    app = _app(
        spec,
        brokered_model_loop_enabled=True,
        brokered_model_http_client=httpx.AsyncClient(transport=transport),
        max_pending_responses=1,
    )

    async def exercise() -> httpx.Response:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            try:
                await client.post("/responses", json={"input": "cancel this model call"})
            except asyncio.CancelledError:
                pass
            return await client.post("/responses", json={"input": "check-network-telemetry"})

    response = asyncio.run(exercise())

    assert response.status_code == 200, response.text
    assert _call(response.json())["name"] == "check-network-telemetry"
    assert transport.calls == 2


def test_foundry_brokered_active_resume_survives_ttl_and_completed_state_retains_from_completion():
    spec = _spec(tool_name="check-network-telemetry")

    class BlockingResumeTransport(httpx.AsyncBaseTransport):
        def __init__(self) -> None:
            self.requests: list[dict[str, Any]] = []
            self.resume_started = threading.Event()
            self.release_resume = threading.Event()
            self.initial_calls = 0

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content.decode("utf-8"))
            self.requests.append(payload)
            if "tools" in payload:
                self.initial_calls += 1
                return httpx.Response(
                    200,
                    request=request,
                    json=_chat_response(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": f"model_call_{self.initial_calls}",
                                    "type": "function",
                                    "function": {"name": "check-network-telemetry", "arguments": "{}"},
                                }
                            ],
                        }
                    ),
                )
            self.resume_started.set()
            while not self.release_resume.is_set():
                await asyncio.sleep(0.001)
            return httpx.Response(
                200,
                request=request,
                json=_chat_response({"role": "assistant", "content": "Resume completed after the lease window."}),
            )

    transport = BlockingResumeTransport()
    model_client = httpx.AsyncClient(transport=transport)
    app = _app(
        spec,
        brokered_model_loop_enabled=True,
        brokered_model_http_client=model_client,
        state_ttl_seconds=0.05,
    )

    with TestClient(app) as client, ThreadPoolExecutor(max_workers=1) as executor:
        initial = client.post("/responses", json={"input": "check-network-telemetry"})
        call = _call(initial.json())
        payload = _continuation(initial.json()["id"], call["call_id"], {"approved": True, "output": {"ok": True}})
        future = executor.submit(client.post, "/responses", headers=CONTINUATION_AUTH, json=payload)
        try:
            assert transport.resume_started.wait(timeout=2)
            time.sleep(0.08)
            another_pending = client.post("/responses", json={"input": "check-network-telemetry again"})
            active_duplicate = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)
        finally:
            transport.release_resume.set()
        completed = future.result(timeout=2)
        immediate_duplicate = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)

    assert another_pending.status_code == 200, another_pending.text
    assert active_duplicate.status_code == 409
    assert active_duplicate.json()["error"]["code"] == "duplicate_continuation_in_progress"
    assert completed.status_code == 200, completed.text
    assert _message_text(completed.json()) == "Resume completed after the lease window."
    assert immediate_duplicate.status_code == 200, immediate_duplicate.text
    assert immediate_duplicate.json() == completed.json()


def test_foundry_brokered_abandoned_resuming_state_expires_and_releases_capacity(monkeypatch):
    stores: list[Any] = []
    original_store = foundry_module._FoundryResponseStateStore

    def capture_store(*args: Any, **kwargs: Any) -> Any:
        store = original_store(*args, **kwargs)
        stores.append(store)
        return store

    monkeypatch.setattr(foundry_module, "_FoundryResponseStateStore", capture_store)
    app = _app(max_pending_responses=1)

    with TestClient(app) as client:
        initial = _start(client)
        abandoned = stores[0].get(initial["id"])
        abandoned.status = "resuming"
        abandoned.expires_at = time.time() - 1
        stores[0].save(abandoned)
        replacement = client.post("/responses", json={"input": "please read telemetry again"})

    assert replacement.status_code == 200, replacement.text
    assert _call(replacement.json())


def test_foundry_brokered_model_loop_emits_model_requested_tool_and_resumes_to_final_answer():
    spec = _spec(tool_name="check-network-telemetry")
    spec.brokered_tools[0].parameters["required"] = ["site"]
    fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "model_generated_call_id",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": '{"site":"sfo"}'},
                        }
                    ],
                },
                prompt_tokens=2,
                completion_tokens=3,
            ),
            _chat_response({"role": "assistant", "content": "Telemetry is healthy."}, prompt_tokens=5, completion_tokens=7),
        ]
    )
    app = _model_loop_app(spec, fake)

    with TestClient(app) as client:
        initial = client.post("/responses", json={"input": "Check SFO telemetry"})
        call = _call(initial.json())
        continuation = _continuation(initial.json()["id"], call["call_id"], {"approved": True, "output": {"status": "healthy"}})
        continuation[CONTINUATION_PROOF_BODY_FIELD] = CONTINUATION_PROOF
        final = client.post(
            "/responses",
            json=continuation,
        )

    assert initial.status_code == 200, initial.text
    assert call["name"] == "check-network-telemetry"
    assert call["call_id"] == f"call_{initial.json()['id']}_1"
    assert json.loads(call["arguments"]) == {"site": "sfo"}
    assert initial.json()["usage"] == {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5}
    assert final.status_code == 200, final.text
    assert _message_text(final.json()) == "Telemetry is healthy."
    assert final.json()["usage"] == {"input_tokens": 7, "output_tokens": 10, "total_tokens": 17}
    assert fake.requests[0]["tools"][0]["function"]["name"] == "check-network-telemetry"
    assert fake.requests[0]["tools"][0]["function"]["description"].startswith("Brokered class: read.")
    assert fake.requests[1]["messages"][-1]["role"] == "tool"
    assert fake.requests[1]["messages"][-1]["tool_call_id"] == call["call_id"]
    assert "tools" not in fake.requests[1]
    assert CONTINUATION_PROOF not in json.dumps(fake.requests, sort_keys=True)


def test_foundry_brokered_model_loop_missing_api_key_is_not_ready_or_caller_error(monkeypatch):
    spec_data = _spec(tool_name="check-network-telemetry").model_dump(by_alias=True)
    spec_data["model"]["apiKeyEnv"] = "MISSING_FOUNDRY_MODEL_KEY"
    spec = AgentSpec.model_validate(spec_data)
    monkeypatch.delenv("MISSING_FOUNDRY_MODEL_KEY", raising=False)
    fake = _FakeChatTransport([_chat_response({"role": "assistant", "content": "must not run"})])
    app = _model_loop_app(spec, fake)

    with TestClient(app) as client:
        readiness = client.get("/readiness")
        response = client.post("/responses", json={"input": "check-network-telemetry"})

    assert readiness.status_code == 503
    assert readiness.json()["ready"] is False
    assert readiness.json()["foundryResponses"]["modelAuth"] == "missing"
    assert response.status_code == 503
    assert response.json()["error"] == {
        "message": "model authentication is not configured",
        "code": "ModelAuthMissing",
    }
    assert "MISSING_FOUNDRY_MODEL_KEY" not in readiness.text + response.text
    assert fake.requests == []


def test_foundry_brokered_model_loop_missing_key_precedes_capacity_errors(monkeypatch, tmp_path):
    state_file = tmp_path / "responses-state.json"
    spec_data = _spec(tool_name="check-network-telemetry").model_dump(by_alias=True)
    spec_data["model"]["apiKeyEnv"] = "FOUNDRY_MODEL_KEY"
    spec = AgentSpec.model_validate(spec_data)
    monkeypatch.setenv("FOUNDRY_MODEL_KEY", "mock-token")
    first_fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "model_generated_call_id",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": "{}"},
                        }
                    ],
                }
            )
        ]
    )
    with TestClient(
        _model_loop_app(
            spec,
            first_fake,
            response_state_file=state_file,
            max_pending_responses=1,
        )
    ) as client:
        initial = client.post("/responses", json={"input": "check-network-telemetry"})
    assert initial.status_code == 200, initial.text

    monkeypatch.delenv("FOUNDRY_MODEL_KEY")
    second_fake = _FakeChatTransport([_chat_response({"role": "assistant", "content": "must not run"})])
    with TestClient(
        _model_loop_app(
            spec,
            second_fake,
            response_state_file=state_file,
            max_pending_responses=1,
        )
    ) as client:
        response = client.post("/responses", json={"input": "check-network-telemetry again"})

    assert response.status_code == 503
    assert response.json()["error"] == {
        "message": "model authentication is not configured",
        "code": "ModelAuthMissing",
    }
    assert second_fake.requests == []


def test_foundry_brokered_model_loop_missing_key_precedes_continuation_state_errors(monkeypatch, tmp_path):
    state_file = tmp_path / "responses-state.json"
    spec_data = _spec(tool_name="check-network-telemetry").model_dump(by_alias=True)
    spec_data["model"]["apiKeyEnv"] = "FOUNDRY_MODEL_KEY"
    spec = AgentSpec.model_validate(spec_data)
    monkeypatch.setenv("FOUNDRY_MODEL_KEY", "mock-token")
    first_fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "model-generated-call",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": "{}"},
                        }
                    ],
                }
            )
        ]
    )
    with TestClient(_model_loop_app(spec, first_fake, response_state_file=state_file)) as client:
        initial = client.post("/responses", json={"input": "check-network-telemetry"})
        call = _call(initial.json())
    persisted_before = state_file.read_bytes()

    monkeypatch.delenv("FOUNDRY_MODEL_KEY")
    second_fake = _FakeChatTransport([_chat_response({"role": "assistant", "content": "must not run"})])
    with TestClient(_model_loop_app(spec, second_fake, response_state_file=state_file)) as client:
        response = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_continuation(initial.json()["id"], call["call_id"], {"approved": True, "output": {"ok": True}}),
        )

    assert response.status_code == 503
    assert response.json()["error"] == {
        "message": "model authentication is not configured",
        "code": "ModelAuthMissing",
    }
    assert state_file.read_bytes() == persisted_before
    assert second_fake.requests == []


def test_foundry_brokered_model_loop_sanitizes_upstream_auth_failures():
    spec_data = _spec(tool_name="check-network-telemetry").model_dump(by_alias=True)
    internal_url = "https://private-model.internal.example/v1"
    spec_data["model"]["baseURL"] = internal_url
    spec = AgentSpec.model_validate(spec_data)

    for upstream_status in (401, 403):
        def reject(request: httpx.Request, *, status: int = upstream_status) -> httpx.Response:
            return httpx.Response(status, request=request, json={"error": "credential rejected"})

        model_client = httpx.AsyncClient(transport=httpx.MockTransport(reject))
        app = _app(
            spec,
            brokered_model_loop_enabled=True,
            brokered_model_http_client=model_client,
        )
        with TestClient(app) as client:
            response = client.post("/responses", json={"input": "check-network-telemetry"})

        assert response.status_code == 503
        assert response.json()["error"] == {
            "message": "model service rejected configured credentials",
            "code": "ModelAuthRejected",
        }
        assert internal_url not in response.text
        assert "/chat/completions" not in response.text


def test_foundry_brokered_model_loop_sanitizes_transport_failure_urls():
    spec_data = _spec(tool_name="check-network-telemetry").model_dump(by_alias=True)
    internal_url = "https://private-model.internal.example/v1"
    spec_data["model"]["baseURL"] = internal_url
    spec = AgentSpec.model_validate(spec_data)

    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"cannot connect to {internal_url}", request=request)

    model_client = httpx.AsyncClient(transport=httpx.MockTransport(fail))
    app = _app(
        spec,
        brokered_model_loop_enabled=True,
        brokered_model_http_client=model_client,
    )

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "check-network-telemetry"})

    assert response.status_code == 502
    assert response.json()["error"] == {
        "message": "model service request failed",
        "code": "ModelUpstreamError",
    }
    assert internal_url not in response.text
    assert "/chat/completions" not in response.text


def test_foundry_brokered_model_loop_normalizes_unexpected_json_decode_failure():
    class RecursiveJsonResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            raise RecursionError("nested model response")

    class RecursiveJsonClient:
        async def post(self, *_args: Any, **_kwargs: Any) -> RecursiveJsonResponse:
            return RecursiveJsonResponse()

    app = _app(
        _spec(tool_name="check-network-telemetry"),
        brokered_model_loop_enabled=True,
        brokered_model_http_client=RecursiveJsonClient(),
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/responses", json={"input": "check-network-telemetry"})

    assert response.status_code == 502
    assert response.json()["error"] == {
        "message": "model service returned an invalid JSON response",
        "code": "InvalidModelResponse",
    }


def test_foundry_model_loop_rejects_oversized_output_before_encoding():
    class ExplodingEncodeString(str):
        def encode(self, *args, **kwargs):
            raise AssertionError("oversized output must be rejected before encoding")

    loop = BrokeredChatModelLoop(
        _spec(tool_name="check-network-telemetry"),
        [],
        max_output_bytes=128,
    )

    with pytest.raises(AgentRunError) as exc_info:
        asyncio.run(
            loop.resume(
                [{"role": "assistant", "content": None}],
                call_id="call-1",
                output=ExplodingEncodeString("x" * 129),
            )
        )

    assert exc_info.value.status == 413
    assert exc_info.value.code == "brokered_output_too_large"


def test_foundry_model_loop_rejects_lone_surrogate_tool_output_before_encoding():
    loop = BrokeredChatModelLoop(
        _spec(tool_name="check-network-telemetry"),
        [],
    )

    with pytest.raises(AgentRunError) as exc_info:
        asyncio.run(
            loop.resume(
                [{"role": "assistant", "content": None}],
                call_id="call-1",
                output='{"value":"\ud800"}',
            )
        )

    assert exc_info.value.status == 400
    assert exc_info.value.code == "InvalidToolOutput"
    assert "valid Unicode" in str(exc_info.value)


def test_foundry_brokered_model_loop_rejects_lone_surrogate_in_decoded_tool_arguments():
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            json=_chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "model-generated-call",
                            "type": "function",
                            "function": {
                                "name": "check-network-telemetry",
                                "arguments": '{"value":"\\ud800"}',
                            },
                        }
                    ],
                }
            ),
        )

    app = _app(
        _spec(tool_name="check-network-telemetry"),
        brokered_model_loop_enabled=True,
        brokered_model_http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/responses", json={"input": "check-network-telemetry"})

    assert response.status_code == 502
    assert response.json()["error"] == {
        "message": "model service returned an invalid JSON response",
        "code": "InvalidModelResponse",
    }


def test_foundry_brokered_model_loop_rejects_decoded_lone_surrogate():
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            content=(
                b'{"choices":[{"message":{"role":"assistant","content":"\\ud800"}}],'
                b'"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}'
            ),
            headers={"content-type": "application/json"},
        )

    model_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    app = _app(
        _spec(tool_name="check-network-telemetry"),
        brokered_model_loop_enabled=True,
        brokered_model_http_client=model_client,
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/responses", json={"input": "check-network-telemetry"})

    assert response.status_code == 502
    assert response.json()["error"] == {
        "message": "model service returned an invalid JSON response",
        "code": "InvalidModelResponse",
    }


def test_foundry_brokered_model_loop_sanitizes_upstream_auth_failure_on_resume():
    spec_data = _spec(tool_name="check-network-telemetry").model_dump(by_alias=True)
    internal_url = "https://private-model.internal.example/v1"
    spec_data["model"]["baseURL"] = internal_url
    spec = AgentSpec.model_validate(spec_data)
    calls = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                200,
                request=request,
                json=_chat_response(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "model_generated_call_id",
                                "type": "function",
                                "function": {"name": "check-network-telemetry", "arguments": "{}"},
                            }
                        ],
                    }
                ),
            )
        if calls == 2:
            return httpx.Response(401, request=request, json={"error": "credential rejected"})
        return httpx.Response(
            200,
            request=request,
            json=_chat_response({"role": "assistant", "content": "Retry after auth recovery."}),
        )

    model_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    app = _app(
        spec,
        brokered_model_loop_enabled=True,
        brokered_model_http_client=model_client,
    )

    with TestClient(app) as client:
        initial = client.post("/responses", json={"input": "check-network-telemetry"})
        call = _call(initial.json())
        payload = _continuation(initial.json()["id"], call["call_id"], {"approved": True, "output": {"ok": True}})
        failed = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)
        retried = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)

    assert failed.status_code == 503
    assert failed.json()["error"] == {"message": "model resume failed", "code": "ModelResumeError"}
    assert internal_url not in failed.text
    assert retried.status_code == 200, retried.text
    assert _message_text(retried.json()) == "Retry after auth recovery."


def test_foundry_brokered_invalid_file_state_starts_with_empty_store(tmp_path):
    state_file = tmp_path / "responses-state.json"
    state_file.write_text("{not valid json", encoding="utf-8")
    app = _app(response_state_file=state_file)

    with TestClient(app) as client:
        response = client.get("/readiness")
        initial = client.post("/responses", json={"input": "conformance_read"})

    assert response.status_code == 200
    assert initial.status_code == 200, initial.text
    assert _call(initial.json())["name"] == "conformance_read"


def test_foundry_brokered_ignores_state_file_larger_than_configured_limit(tmp_path):
    state_file = tmp_path / "responses-state.json"

    with TestClient(_app(response_state_file=state_file)) as client:
        initial = _start(client)
        call = _call(initial)

    persisted = json.loads(state_file.read_text(encoding="utf-8"))
    persisted["padding"] = "x" * 5_000
    state_file.write_text(json.dumps(persisted, separators=(",", ":"), sort_keys=True), encoding="utf-8")
    assert state_file.stat().st_size > 4 * 1024

    with TestClient(_app(response_state_file=state_file, max_response_state_bytes=4 * 1024)) as client:
        continuation = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_continuation(initial["id"], call["call_id"], {"approved": True, "output": {"ok": True}}),
        )

    assert continuation.status_code == 404
    assert continuation.json()["error"]["code"] == "unknown_previous_response_id"


def test_foundry_brokered_file_state_is_written_with_private_permissions(tmp_path):
    state_file = tmp_path / "responses-state.json"
    app = _app(response_state_file=state_file)

    with TestClient(app) as client:
        _start(client)

    assert state_file.exists()
    assert state_file.stat().st_mode & 0o777 == 0o600


def test_foundry_brokered_failed_continuation_state_persist_is_retryable_without_stranding_progress(monkeypatch, tmp_path):
    state_file = tmp_path / "responses-state.json"
    spec = _spec(tool_name="check-network-telemetry")
    fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "model_generated_call_id",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": "{}"},
                        }
                    ],
                }
            ),
            _chat_response({"role": "assistant", "content": "Recovered after storage retry."}),
        ]
    )
    app = _model_loop_app(spec, fake, response_state_file=state_file)
    original_replace = Path.replace
    failed_once = False

    def fail_first_resuming_replace(path: Path, target: Path) -> Path:
        nonlocal failed_once
        if not failed_once and path.name == f".{state_file.name}.tmp":
            persisted = json.loads(path.read_text(encoding="utf-8"))
            stored_state = next(iter(persisted["states"].values()))
            if stored_state["status"] == "resuming":
                failed_once = True
                raise OSError("simulated continuation state storage failure")
        return original_replace(path, target)

    with TestClient(app, raise_server_exceptions=False) as client:
        initial = client.post("/responses", json={"input": "check-network-telemetry"})
        call = _call(initial.json())
        payload = _continuation(initial.json()["id"], call["call_id"], {"approved": True, "output": {"ok": True}})
        monkeypatch.setattr(Path, "replace", fail_first_resuming_replace)
        failed = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)
        retried = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)

    assert failed.status_code == 503
    assert failed.json()["error"] == {
        "message": "brokered response state storage unavailable",
        "code": "brokered_response_state_storage_error",
    }
    assert retried.status_code == 200, retried.text
    assert _message_text(retried.json()) == "Recovered after storage retry."
    assert len(fake.requests) == 2


def test_foundry_brokered_failed_final_state_persist_retries_cached_completion_without_second_resume(monkeypatch, tmp_path):
    state_file = tmp_path / "responses-state.json"
    spec = _spec(tool_name="check-network-telemetry")
    fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "model_generated_call_id",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": "{}"},
                        }
                    ],
                }
            ),
            _chat_response({"role": "assistant", "content": "Persist this exact completion."}),
            _chat_response({"role": "assistant", "content": "A duplicate resume incorrectly ran."}),
        ]
    )
    app = _model_loop_app(spec, fake, response_state_file=state_file)
    original_replace = Path.replace
    failed_once = False

    def fail_first_completed_replace(path: Path, target: Path) -> Path:
        nonlocal failed_once
        if not failed_once and path.name == f".{state_file.name}.tmp":
            persisted = json.loads(path.read_text(encoding="utf-8"))
            stored_state = next(iter(persisted["states"].values()))
            if stored_state["status"] == "completed":
                failed_once = True
                raise OSError("simulated completed state storage failure")
        return original_replace(path, target)

    with TestClient(app, raise_server_exceptions=False) as client:
        initial = client.post("/responses", json={"input": "check-network-telemetry"})
        call = _call(initial.json())
        payload = _continuation(initial.json()["id"], call["call_id"], {"approved": True, "output": {"ok": True}})
        monkeypatch.setattr(Path, "replace", fail_first_completed_replace)
        failed = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)
        retried = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)

    assert failed.status_code == 503
    assert failed.json()["error"]["code"] == "brokered_response_state_storage_error"
    assert retried.status_code == 200, retried.text
    assert _message_text(retried.json()) == "Persist this exact completion."
    assert len(fake.requests) == 2
    persisted_state = json.loads(state_file.read_text(encoding="utf-8"))["states"][initial.json()["id"]]
    assert persisted_state["status"] == "completed"
    assert persisted_state["finalPayload"] == retried.json()


def test_foundry_brokered_unrelated_full_map_persist_marks_cached_completion_durable(monkeypatch, tmp_path):
    state_file = tmp_path / "responses-state.json"
    spec = _spec(tool_name="check-network-telemetry")

    def tool_request(call_id: str) -> dict[str, Any]:
        return _chat_response(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "check-network-telemetry", "arguments": "{}"},
                    }
                ],
            }
        )

    fake = _FakeChatTransport(
        [
            tool_request("model_generated_call_id"),
            _chat_response({"role": "assistant", "content": "Persisted by an unrelated transaction."}),
            tool_request("unrelated_call_1"),
            tool_request("unrelated_call_2"),
        ]
    )
    app = _model_loop_app(spec, fake, response_state_file=state_file, max_pending_responses=2)
    original_replace = Path.replace
    first_completed_write_failed = False
    fail_next_write = False
    second_failure_triggered = False

    def fail_selected_replaces(path: Path, target: Path) -> Path:
        nonlocal first_completed_write_failed, second_failure_triggered
        if path.name == f".{state_file.name}.tmp":
            persisted = json.loads(path.read_text(encoding="utf-8"))
            if not first_completed_write_failed and any(state["status"] == "completed" for state in persisted["states"].values()):
                first_completed_write_failed = True
                raise OSError("simulated completed state storage failure")
            if fail_next_write:
                second_failure_triggered = True
                raise OSError("simulated later storage failure")
        return original_replace(path, target)

    with TestClient(app, raise_server_exceptions=False) as client:
        initial = client.post("/responses", json={"input": "check-network-telemetry"})
        call = _call(initial.json())
        payload = _continuation(initial.json()["id"], call["call_id"], {"approved": True, "output": {"ok": True}})
        monkeypatch.setattr(Path, "replace", fail_selected_replaces)
        failed = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)
        unrelated = client.post("/responses", json={"input": "check-network-telemetry unrelated"})
        durable_state = json.loads(state_file.read_text(encoding="utf-8"))["states"][initial.json()["id"]]
        fail_next_write = True
        retried = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)
        fail_next_write = False
        replacement = client.post("/responses", json={"input": "check-network-telemetry replacement"})

    assert failed.status_code == 503
    assert unrelated.status_code == 200, unrelated.text
    assert durable_state["status"] == "completed"
    assert durable_state["finalPayload"] is not None
    assert retried.status_code == 200, retried.text
    assert _message_text(retried.json()) == "Persisted by an unrelated transaction."
    assert second_failure_triggered is False
    assert replacement.status_code == 200, replacement.text
    assert _call(replacement.json())
    assert len(fake.requests) == 4


def test_foundry_brokered_recovered_storage_persists_and_evicts_cached_completion_at_capacity(monkeypatch, tmp_path):
    state_file = tmp_path / "responses-state.json"
    spec = _spec(tool_name="check-network-telemetry")
    fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "model_generated_call_id",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": "{}"},
                        }
                    ],
                }
            ),
            _chat_response({"role": "assistant", "content": "Completion cached while storage failed."}),
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "replacement_model_call",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": "{}"},
                        }
                    ],
                }
            ),
        ]
    )
    app = _model_loop_app(spec, fake, response_state_file=state_file, max_pending_responses=1)
    original_replace = Path.replace
    failed_once = False

    def fail_first_completed_replace(path: Path, target: Path) -> Path:
        nonlocal failed_once
        if not failed_once and path.name == f".{state_file.name}.tmp":
            persisted = json.loads(path.read_text(encoding="utf-8"))
            if any(state["status"] == "completed" for state in persisted["states"].values()):
                failed_once = True
                raise OSError("simulated completed state storage failure")
        return original_replace(path, target)

    with TestClient(app, raise_server_exceptions=False) as client:
        initial = client.post("/responses", json={"input": "check-network-telemetry"})
        call = _call(initial.json())
        payload = _continuation(initial.json()["id"], call["call_id"], {"approved": True, "output": {"ok": True}})
        monkeypatch.setattr(Path, "replace", fail_first_completed_replace)
        failed = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)
        replacement = client.post("/responses", json={"input": "check-network-telemetry replacement"})

    assert failed.status_code == 503
    assert replacement.status_code == 200, replacement.text
    replacement_body = replacement.json()
    assert _call(replacement_body)
    persisted_states = json.loads(state_file.read_text(encoding="utf-8"))["states"]
    assert initial.json()["id"] not in persisted_states
    assert replacement_body["id"] in persisted_states
    assert len(fake.requests) == 3


def test_foundry_brokered_file_state_recovers_unfinalized_accepted_continuation_after_restart(tmp_path):
    state_file = tmp_path / "responses-state.json"
    spec = _spec(tool_name="check-network-telemetry")
    fake_first = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "model_generated_call_id",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": '{}'},
                        }
                    ],
                }
            )
        ]
    )
    first_app = _app(spec, brokered_model_loop_enabled=True, brokered_model_http_client=httpx.AsyncClient(transport=httpx.MockTransport(fake_first.handler)), response_state_file=state_file)

    with TestClient(first_app) as client:
        initial = client.post("/responses", json={"input": "check-network-telemetry"})
        call = _call(initial.json())

    payload = _continuation(initial.json()["id"], call["call_id"], {"approved": True, "output": {"ok": True}})
    state_data = json.loads(state_file.read_text(encoding="utf-8"))
    state = state_data["states"][initial.json()["id"]]
    state["acceptedOutputs"] = {call["call_id"]: payload["input"][0]["output"]}
    state["status"] = "resuming"
    state["finalPayload"] = None
    state_file.write_text(json.dumps(state_data, separators=(",", ":"), sort_keys=True), encoding="utf-8")

    fake_second = _FakeChatTransport([_chat_response({"role": "assistant", "content": "Recovered."})])
    second_app = _app(spec, brokered_model_loop_enabled=True, brokered_model_http_client=httpx.AsyncClient(transport=httpx.MockTransport(fake_second.handler)), response_state_file=state_file)
    with TestClient(second_app) as client:
        retried = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)

    assert retried.status_code == 200, retried.text
    assert _message_text(retried.json()) == "Recovered."


def test_foundry_brokered_model_loop_accepts_integer_arguments_encoded_as_integral_float():
    spec = _spec(tool_name="retry-tool")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {"retries": {"type": "integer"}},
        "required": ["retries"],
    }
    fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_model",
                            "type": "function",
                            "function": {"name": "retry-tool", "arguments": '{"retries":1.0}'},
                        }
                    ],
                }
            )
        ]
    )
    app = _model_loop_app(spec, fake)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "call retry-tool"})

    assert response.status_code == 200, response.text
    assert json.loads(_call(response.json())["arguments"]) == {"retries": 1.0}


def test_foundry_brokered_model_loop_unexpected_resume_failure_can_be_retried():
    spec = _spec(tool_name="check-network-telemetry")

    class FlakyResumeTransport:
        def __init__(self) -> None:
            self.requests: list[dict[str, Any]] = []
            self.resume_attempts = 0

        def handler(self, request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content.decode("utf-8"))
            self.requests.append(payload)
            if len(self.requests) == 1:
                return httpx.Response(
                    200,
                    json=_chat_response(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "model_generated_call_id",
                                    "type": "function",
                                    "function": {"name": "check-network-telemetry", "arguments": '{}'},
                                }
                            ],
                        }
                    ),
                )
            self.resume_attempts += 1
            if self.resume_attempts == 1:
                raise RuntimeError("transient resume failure")
            return httpx.Response(200, json=_chat_response({"role": "assistant", "content": "Recovered after retry."}))

    fake = FlakyResumeTransport()
    app = _model_loop_app(spec, fake)

    with TestClient(app) as client:
        initial = client.post("/responses", json={"input": "check-network-telemetry"})
        call = _call(initial.json())
        payload = _continuation(initial.json()["id"], call["call_id"], {"approved": True, "output": {"ok": True}})
        failed = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)
        retried = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)

    assert failed.status_code == 502
    assert failed.json()["error"] == {"message": "model resume failed", "code": "ModelResumeError"}
    assert retried.status_code == 200, retried.text
    assert _message_text(retried.json()) == "Recovered after retry."


def test_foundry_brokered_model_loop_failed_resume_can_be_retried():
    spec = _spec(tool_name="check-network-telemetry")
    fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "model_generated_call_id",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": '{}'},
                        }
                    ],
                }
            ),
            {"choices": []},
            _chat_response({"role": "assistant", "content": "Retry worked."}),
        ]
    )
    app = _model_loop_app(spec, fake)

    with TestClient(app) as client:
        initial = client.post("/responses", json={"input": "check-network-telemetry"})
        call = _call(initial.json())
        payload = _continuation(initial.json()["id"], call["call_id"], {"approved": True, "output": {"ok": True}})
        failed = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)
        retried = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)

    assert failed.status_code == 502
    assert retried.status_code == 200, retried.text
    assert _message_text(retried.json()) == "Retry worked."


def test_foundry_brokered_model_loop_oversized_completion_can_be_retried():
    spec = _spec(tool_name="check-network-telemetry")
    fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "model_generated_call_id",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": "{}"},
                        }
                    ],
                }
            ),
            _chat_response({"role": "assistant", "content": "x" * 5_000}),
            _chat_response({"role": "assistant", "content": "Retry stayed bounded."}),
        ]
    )
    app = _model_loop_app(spec, fake, max_response_state_bytes=4 * 1024)

    with TestClient(app) as client:
        initial = client.post("/responses", json={"input": "check-network-telemetry"})
        call = _call(initial.json())
        payload = _continuation(initial.json()["id"], call["call_id"], {"approved": True, "output": {"ok": True}})
        failed = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)
        retried = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)

    assert failed.status_code == 502
    assert failed.json()["error"] == {
        "message": "model response is too large to retain safely",
        "code": "ModelResponseTooLarge",
    }
    assert retried.status_code == 200, retried.text
    assert _message_text(retried.json()) == "Retry stayed bounded."
    assert len(fake.requests) == 3


def test_foundry_brokered_model_loop_discards_resume_transcript_before_final_state_sizing(tmp_path):
    state_file = tmp_path / "responses-state.json"
    spec = _spec(tool_name="check-network-telemetry")
    first_fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "model-generated-call",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": "{}"},
                        }
                    ],
                }
            )
        ]
    )
    with TestClient(_model_loop_app(spec, first_fake, response_state_file=state_file)) as client:
        initial = client.post("/responses", json={"input": "check-network-telemetry " + "x" * 2_000})
        call = _call(initial.json())
    pending_state_bytes = state_file.stat().st_size

    second_fake = _FakeChatTransport([_chat_response({"role": "assistant", "content": "Small final answer."})])
    with TestClient(
        _model_loop_app(
            spec,
            second_fake,
            response_state_file=state_file,
            max_response_state_bytes=pending_state_bytes + 256,
        )
    ) as client:
        completed = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_continuation(initial.json()["id"], call["call_id"], {"approved": True, "output": {"ok": True}}),
        )

    assert completed.status_code == 200, completed.text
    assert _message_text(completed.json()) == "Small final answer."
    persisted = json.loads(state_file.read_text(encoding="utf-8"))["states"][initial.json()["id"]]
    assert persisted["modelMessages"] is None
    assert persisted["initialUsage"] == {}


def test_foundry_brokered_model_loop_aggregate_state_pressure_is_terminal_for_duplicate(tmp_path):
    state_file = tmp_path / "responses-state.json"
    spec = _spec(tool_name="check-network-telemetry")

    def tool_request(call_id: str) -> dict[str, Any]:
        return _chat_response(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "check-network-telemetry", "arguments": "{}"},
                    }
                ],
            }
        )

    fake = _FakeChatTransport(
        [
            tool_request("first-model-call"),
            tool_request("second-model-call"),
            _chat_response({"role": "assistant", "content": "Computed once under capacity pressure."}),
            _chat_response({"role": "assistant", "content": "must not be called for duplicate"}),
        ]
    )
    app = _model_loop_app(
        spec,
        fake,
        response_state_file=state_file,
        max_pending_responses=3,
        max_response_state_bytes=2_500,
    )

    with TestClient(app) as client:
        first = client.post("/responses", json={"input": "check-network-telemetry first"})
        first_call = _call(first.json())
        second = client.post("/responses", json={"input": "check-network-telemetry second"})
        _call(second.json())
        payload = _continuation(first.json()["id"], first_call["call_id"], {"approved": True, "output": {"ok": True}})
        failed = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)
        duplicate = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)

    restart_fake = _FakeChatTransport([_chat_response({"role": "assistant", "content": "must not run after restart"})])
    with TestClient(
        _model_loop_app(
            spec,
            restart_fake,
            response_state_file=state_file,
            max_pending_responses=3,
            max_response_state_bytes=2_500,
        )
    ) as client:
        restarted_duplicate = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert failed.status_code == 429
    assert failed.json()["error"]["code"] == "brokered_response_state_full"
    assert duplicate.status_code == 429
    assert duplicate.json() == failed.json()
    assert len(fake.requests) == 3
    assert restarted_duplicate.status_code == 429
    assert restarted_duplicate.json() == failed.json()
    assert restart_fake.requests == []


def test_foundry_brokered_bounds_initial_model_state_before_copying(monkeypatch):
    spec = _spec(tool_name="check-network-telemetry")
    fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": "x" * 5_000,
                    "tool_calls": [
                        {
                            "id": "model_generated_call_id",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": "{}"},
                        }
                    ],
                }
            )
        ]
    )
    app = _model_loop_app(spec, fake, max_response_state_bytes=4 * 1024)
    original_deepcopy = foundry_module.deepcopy

    def reject_oversized_state_copy(value: Any, memo: dict[int, Any] | None = None) -> Any:
        if type(value).__name__ == "_HostedResponseState" and value.model_messages:
            if any(len(message.get("content") or "") > 4 * 1024 for message in value.model_messages):
                raise AssertionError("oversized state must be rejected before deepcopy")
        return original_deepcopy(value, memo) if memo is not None else original_deepcopy(value)

    monkeypatch.setattr(foundry_module, "deepcopy", reject_oversized_state_copy)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/responses", json={"input": "check-network-telemetry"})

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "brokered_response_state_too_large"


def test_foundry_brokered_model_loop_can_return_final_message_without_tool_call():
    fake = _FakeChatTransport([_chat_response({"role": "assistant", "content": "No tool needed."})])
    app = _model_loop_app(_spec(tool_name="check-network-telemetry"), fake)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "Say hello"})

    assert response.status_code == 200, response.text
    assert _message_text(response.json()) == "No tool needed."
    assert fake.requests[0]["tool_choice"] == "auto"


def test_foundry_brokered_model_loop_rejects_unsupported_pattern_deterministically():
    spec = _spec(tool_name="check-network-telemetry")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {"site": {"type": "string", "pattern": "\\p{L}+"}},
        "required": ["site"],
    }
    fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_model",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": '{"site":"sfo"}'},
                        }
                    ],
                }
            )
        ]
    )
    app = _model_loop_app(spec, fake)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "call check-network-telemetry"})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "UnsupportedBrokeredSchema"


def test_foundry_brokered_model_loop_validates_additional_properties_schema():
    spec = _spec(tool_name="check-network-telemetry")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {"site": {"type": "string"}},
        "additionalProperties": {"type": "string"},
    }
    fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_model",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": '{"site":"sfo","count":1}'},
                        }
                    ],
                }
            )
        ]
    )
    app = _model_loop_app(spec, fake)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "call check-network-telemetry"})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "InvalidToolArguments"


def test_foundry_brokered_model_loop_rejects_unknown_model_tool_request():
    fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_model",
                            "type": "function",
                            "function": {"name": "unknown", "arguments": "{}"},
                        }
                    ],
                }
            )
        ]
    )
    app = _model_loop_app(_spec(tool_name="check-network-telemetry"), fake)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "call unknown"})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unknown_brokered_tool"


def test_foundry_brokered_model_loop_validates_schema_valued_additional_properties():
    spec = _spec(tool_name="flex-tool")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "additionalProperties": {"type": "integer"},
    }
    fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_model",
                            "type": "function",
                            "function": {"name": "flex-tool", "arguments": '{"safe":"not-int"}'},
                        }
                    ],
                }
            )
        ]
    )
    app = _model_loop_app(spec, fake)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "call flex-tool"})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "InvalidToolArguments"


def test_foundry_brokered_model_loop_rejects_nonfinite_model_arguments():
    spec = _spec(tool_name="check-network-telemetry")
    fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_model",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": '{"value":NaN}'},
                        }
                    ],
                }
            )
        ]
    )
    app = _model_loop_app(spec, fake)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "call check-network-telemetry"})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "InvalidToolArguments"


def test_foundry_brokered_model_loop_uses_type_strict_const_and_enum_matching():
    for property_schema, arguments in [
        ({"const": 1}, '{"value":true}'),
        ({"enum": [0]}, '{"value":false}'),
    ]:
        spec = _spec(tool_name="check-network-telemetry")
        spec.brokered_tools[0].parameters = {
            "type": "object",
            "properties": {"value": property_schema},
            "required": ["value"],
        }
        fake = _FakeChatTransport(
            [
                _chat_response(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_model",
                                "type": "function",
                                "function": {"name": "check-network-telemetry", "arguments": arguments},
                            }
                        ],
                    }
                )
            ]
        )
        app = _model_loop_app(spec, fake)

        with TestClient(app) as client:
            response = client.post("/responses", json={"input": "call check-network-telemetry"})

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "InvalidToolArguments"


def test_foundry_brokered_model_loop_rejects_arguments_that_violate_declared_schema():
    spec = _spec(tool_name="check-network-telemetry")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {"site": {"type": "string"}},
        "required": ["site"],
        "additionalProperties": False,
    }
    fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_model",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": '{"site":123,"extra":"nope"}'},
                        }
                    ],
                }
            )
        ]
    )
    app = _model_loop_app(spec, fake)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "call check-network-telemetry"})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "InvalidToolArguments"


def test_foundry_brokered_model_loop_validates_large_integer_bounds_exactly():
    spec = _spec(tool_name="check-network-telemetry")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {"count": {"type": "integer", "maximum": 9007199254740992}},
        "required": ["count"],
    }
    fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_model",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": '{"count":9007199254740993}'},
                        }
                    ],
                }
            )
        ]
    )
    app = _model_loop_app(spec, fake)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "call check-network-telemetry"})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "InvalidToolArguments"


def test_foundry_brokered_model_loop_rejects_float_arguments_that_cannot_round_trip_exactly():
    spec = _spec(tool_name="check-network-telemetry")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {"count": {"type": "integer", "maximum": 9007199254740992}},
        "required": ["count"],
    }
    fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_model",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": '{"count":9007199254740993.0}'},
                        }
                    ],
                }
            )
        ]
    )
    app = _model_loop_app(spec, fake)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "call check-network-telemetry"})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "InvalidToolArguments"


def test_foundry_brokered_model_loop_rejects_unsafe_model_generated_arguments():
    fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_model",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": '{"site":"sfo","tokenValue":"ghp_not_real"}'},
                        }
                    ],
                }
            )
        ]
    )
    app = _model_loop_app(_spec(tool_name="check-network-telemetry"), fake)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "call check-network-telemetry"})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "UnsafeBrokeredArguments"


def test_foundry_brokered_rejects_tool_choice_none_instead_of_ignoring_it():
    app = _app()

    with TestClient(app) as client:
        resp = client.post("/responses", json={"input": "hi", "tool_choice": "none"})

    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "tool_choice_unsupported"


def test_foundry_brokered_rejects_request_supplied_tools_even_when_static_tools_exist():
    app = _app()

    with TestClient(app) as client:
        resp = client.post("/responses", json={"input": "hi", "tools": [{"type": "function"}]})

    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "tools_unsupported"


def _fixture(name: str) -> dict[str, Any]:
    path = __import__("pathlib").Path(__file__).parent / "fixtures" / "foundry_brokered" / name
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_initial_response(body: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(body)
    actual_response_id = normalized["id"]
    normalized["id"] = "caresp_test"
    normalized["created_at"] = 0
    item = normalized["output"][0]
    item["id"] = "fc_test"
    item["call_id"] = item["call_id"].replace(actual_response_id, "caresp_test")
    item["response_id"] = "caresp_test"
    return normalized


def _normalize_final_response(body: dict[str, Any], *, previous_response_id: str) -> dict[str, Any]:
    normalized = deepcopy(body)
    actual_response_id = normalized["id"]
    normalized["id"] = "caresp_final"
    normalized["created_at"] = 0
    normalized["previous_response_id"] = "caresp_test"
    item = normalized["output"][0]
    item["id"] = "msg_final"
    item["response_id"] = "caresp_final"
    assert previous_response_id
    assert actual_response_id
    return normalized


def _materialize_continuation(fixture: dict[str, Any], *, response_id: str, call_id: str) -> dict[str, Any]:
    materialized = deepcopy(fixture)
    materialized["previous_response_id"] = response_id
    materialized["input"][0]["call_id"] = call_id
    return materialized


def test_foundry_brokered_golden_fixtures_pin_function_call_loop_and_errors():
    app = _app()

    with TestClient(app) as client:
        initial = client.post("/responses", json=_fixture("initial_request.json"))
        assert initial.status_code == 200, initial.text
        initial_body = initial.json()
        call = _call(initial_body)
        assert _normalize_initial_response(initial_body) == _fixture("function_call_response.json")

        continuation = _materialize_continuation(
            _fixture("continuation_request.json"),
            response_id=initial_body["id"],
            call_id=call["call_id"],
        )
        final = client.post("/responses", json=continuation, headers=CONTINUATION_AUTH)
        assert final.status_code == 200, final.text
        assert _normalize_final_response(final.json(), previous_response_id=initial_body["id"]) == _fixture("final_message_response.json")

        unknown_prev = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_materialize_continuation(
                _fixture("continuation_request.json"),
                response_id="caresp_unknown",
                call_id=call["call_id"],
            ),
        )
        assert unknown_prev.status_code == 404
        assert unknown_prev.json() == _fixture("unknown_previous_response_id_error.json")

    app_for_errors = _app()
    with TestClient(app_for_errors) as client:
        initial_body = _start(client)
        call = _call(initial_body)
        unknown_call = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_materialize_continuation(
                _fixture("continuation_request.json"),
                response_id=initial_body["id"],
                call_id="call_unknown",
            ),
        )
        assert unknown_call.status_code == 400
        assert unknown_call.json() == _fixture("unknown_call_id_error.json")

        multiple = _materialize_continuation(
            _fixture("continuation_request.json"),
            response_id=initial_body["id"],
            call_id=call["call_id"],
        )
        multiple["input"].append(dict(multiple["input"][0]))
        multiple_resp = client.post("/responses", json=multiple, headers=CONTINUATION_AUTH)
        assert multiple_resp.status_code == 400
        assert multiple_resp.json() == _fixture("multiple_function_calls_unsupported_error.json")
