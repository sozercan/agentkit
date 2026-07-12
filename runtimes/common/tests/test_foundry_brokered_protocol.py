from __future__ import annotations

import json
import math
import time
from copy import deepcopy
from types import TracebackType
from typing import Any

from fastapi.testclient import TestClient
import httpx
import pytest

from agentkit_serve_common import foundry as foundry_module
from agentkit_serve_common import foundry_model_loop as foundry_model_loop_module
from agentkit_serve_common.adapter_support import NO_AUTH_API_KEY
from agentkit_serve_common.config import AgentSpec
from agentkit_serve_common.conversation import RunRequest
from agentkit_serve_common.foundry import create_foundry_app
from agentkit_serve_common.runtime import RunResult, RuntimeSession


CONTINUATION_PROOF = "test-orka-continuation-proof"
CONTINUATION_AUTH = {"x-agentkit-brokered-continuation-proof": CONTINUATION_PROOF}


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


@pytest.mark.parametrize("ttl_seconds", [float("nan"), float("inf"), float("-inf")])
def test_foundry_brokered_nonfinite_explicit_state_ttl_uses_default(ttl_seconds: float):
    app = _app(state_ttl_seconds=ttl_seconds)

    with TestClient(app) as client:
        readiness = client.get("/readiness")

    assert readiness.status_code == 200
    assert readiness.json()["foundryResponses"]["stateTtlSeconds"] == 900.0


@pytest.mark.parametrize("raw_ttl", ["nan", "inf", "-inf"])
def test_foundry_brokered_nonfinite_state_ttl_env_uses_default(monkeypatch, raw_ttl: str):
    monkeypatch.setenv("AGENTKIT_FOUNDRY_RESPONSE_STATE_TTL_SECONDS", raw_ttl)
    app = _app()

    with TestClient(app) as client:
        readiness = client.get("/readiness")

    assert readiness.status_code == 200
    assert readiness.json()["foundryResponses"]["stateTtlSeconds"] == 900.0


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


def test_foundry_brokered_rejects_normal_followup_while_previous_response_is_resuming(monkeypatch):
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
        state.status = "resuming"
        stores[0].save(state)
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


def test_foundry_brokered_integer_arguments_preserve_large_integer_bounds():
    huge = 10**100
    spec = _spec(tool_name="large-integer-bounds")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {
            "minimum": {"type": "integer", "minimum": huge},
            "exclusiveMinimum": {"type": "integer", "exclusiveMinimum": huge},
            "exclusiveMaximum": {"type": "integer", "exclusiveMaximum": -huge},
        },
        "required": ["minimum", "exclusiveMinimum", "exclusiveMaximum"],
    }
    app = _app(spec)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "large-integer-bounds"})

    assert response.status_code == 200, response.text
    assert json.loads(_call(response.json())["arguments"]) == {
        "minimum": huge,
        "exclusiveMinimum": huge + 1,
        "exclusiveMaximum": -huge - 1,
    }


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


def test_foundry_brokered_integer_synthesis_rejects_unsupported_multiple_of():
    spec = _spec(tool_name="check-network-telemetry")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {"n": {"type": "integer", "minimum": 1, "multipleOf": 2}},
        "required": ["n"],
    }
    app = _app(spec)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "check-network-telemetry"})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "UnsupportedBrokeredSchema"


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


def test_foundry_brokered_number_synthesis_handles_large_finite_bounds_without_overflow():
    spec = _spec(tool_name="large-number-bounds")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {
            "midpoint": {"type": "number", "minimum": 1e308, "maximum": 1.1e308},
            "above": {"type": "number", "exclusiveMinimum": 1e308},
            "below": {"type": "number", "exclusiveMaximum": -1e308},
            "exactInteger": {"type": "number", "minimum": 9007199254740993, "maximum": 9007199254740994},
            "roundedMidpoint": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 0.9999999999999999},
            "narrowValue": {"type": "number", "exclusiveMinimum": 1.0, "maximum": 1.0000000000000002},
        },
        "required": ["midpoint", "above", "below", "exactInteger", "roundedMidpoint", "narrowValue"],
    }
    app = _app(spec)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "large-number-bounds"})

    assert response.status_code == 200, response.text
    arguments = json.loads(_call(response.json())["arguments"])
    assert arguments["midpoint"] == 1e308
    assert arguments["above"] == math.nextafter(1e308, math.inf)
    assert arguments["below"] == math.nextafter(-1e308, -math.inf)
    assert arguments["exactInteger"] == 9007199254740993
    assert 0 < arguments["roundedMidpoint"] < 0.9999999999999999
    assert arguments["narrowValue"] == 1.0000000000000002


def test_foundry_brokered_number_synthesis_rejects_unsupported_multiple_of():
    spec = _spec(tool_name="number-multiple")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {"value": {"type": "number", "minimum": 0, "multipleOf": 0.5}},
        "required": ["value"],
    }
    app = _app(spec)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "number-multiple"})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "UnsupportedBrokeredSchema"


def test_foundry_brokered_number_synthesis_combines_all_declared_bounds():
    spec = _spec(tool_name="combined-number-bounds")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {
            "value": {"type": "number", "minimum": 0, "exclusiveMinimum": 0, "maximum": 1},
        },
        "required": ["value"],
    }
    app = _app(spec)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "combined-number-bounds"})

    assert response.status_code == 200, response.text
    assert json.loads(_call(response.json())["arguments"]) == {"value": 1}


@pytest.mark.parametrize("bound", [float("nan"), float("inf"), float("-inf")])
def test_foundry_brokered_number_synthesis_rejects_nonfinite_bounds(bound: float):
    spec = _spec(tool_name="nonfinite-number-bound")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {"value": {"type": "number", "minimum": bound}},
        "required": ["value"],
    }
    app = _app(spec)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "nonfinite-number-bound"})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "UnsupportedBrokeredSchema"


def test_foundry_brokered_number_synthesis_prefers_compact_zero_within_wide_bounds():
    spec = _spec(tool_name="compact-number-bounds")
    properties = {
        f"value{index}": {"type": "number", "minimum": -1e308, "maximum": 1e308}
        for index in range(27)
    }
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": properties,
        "required": list(properties),
    }
    app = _app(spec)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "compact-number-bounds"})

    assert response.status_code == 200, response.text
    assert json.loads(_call(response.json())["arguments"]) == {name: 0 for name in properties}


def test_foundry_brokered_number_synthesis_prefers_compact_float_boundary_over_large_integer():
    spec = _spec(tool_name="compact-large-number-bounds")
    properties = {
        f"value{index}": {"type": "number", "minimum": 1e308, "maximum": 1.1e308}
        for index in range(27)
    }
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": properties,
        "required": list(properties),
    }
    app = _app(spec)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "compact-large-number-bounds"})

    assert response.status_code == 200, response.text
    assert json.loads(_call(response.json())["arguments"]) == {name: 1e308 for name in properties}


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


def test_foundry_brokered_rejects_lossy_function_call_output_float_before_state_change(tmp_path):
    state_file = tmp_path / "responses-state.json"
    app = _app(response_state_file=state_file)

    with TestClient(app) as client:
        initial = _start(client)
        call = _call(initial)
        persisted_before = state_file.read_bytes()
        lossy = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json={
                "previous_response_id": initial["id"],
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": call["call_id"],
                        "output": '{"approved":true,"output":{"id":9007199254740993.0}}',
                    }
                ],
            },
        )
        persisted_after_rejection = state_file.read_bytes()
        exact = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json={
                "previous_response_id": initial["id"],
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": call["call_id"],
                        "output": '{"approved":true,"output":{"id":9007199254740992.0}}',
                    }
                ],
            },
        )

    assert lossy.status_code == 400
    assert lossy.json()["error"]["code"] == "invalid_function_call_output"
    assert persisted_after_rejection == persisted_before
    assert exact.status_code == 200, exact.text
    assert _message_text(exact.json()).endswith('{"id":9007199254740992.0}')


def test_foundry_brokered_rejects_object_valued_function_call_output_before_state_change(tmp_path):
    state_file = tmp_path / "responses-state.json"
    app = _app(response_state_file=state_file)

    with TestClient(app) as client:
        initial = _start(client)
        call = _call(initial)
        persisted_before = state_file.read_bytes()
        response = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json={
                "previous_response_id": initial["id"],
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": call["call_id"],
                        "output": {"approved": True, "output": {"id": 9007199254740993.0}},
                    }
                ],
            },
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_function_call_output"
    assert state_file.read_bytes() == persisted_before


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


def test_foundry_brokered_rejects_oversized_persisted_replay_after_limit_is_lowered(tmp_path):
    state_file = tmp_path / "responses-state.json"

    with TestClient(_app(response_state_file=state_file, max_brokered_output_bytes=1024)) as client:
        initial = _start(client)
        call = _call(initial)
        payload = _continuation(
            initial["id"],
            call["call_id"],
            {"approved": True, "output": {"blob": "x" * 256}},
        )
        completed = client.post("/responses", json=payload, headers=CONTINUATION_AUTH)

    with TestClient(_app(response_state_file=state_file, max_brokered_output_bytes=128)) as client:
        replay = client.post("/responses", json=payload, headers=CONTINUATION_AUTH)

    assert completed.status_code == 200, completed.text
    assert replay.status_code == 413
    assert replay.json()["error"]["code"] == "brokered_output_too_large"


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


@pytest.mark.parametrize("brokered_class", ["write", "coordination"])
def test_foundry_brokered_refuses_multi_value_root_enum_side_effecting_arguments(brokered_class: str):
    spec = _spec(tool_name="dispatch-work-order", brokered_class=brokered_class)
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "enum": [
            {"operation": "delete"},
            {"operation": "create"},
        ],
    }
    app = _app(spec)

    with TestClient(app) as client:
        resp = client.post("/responses", json={"input": "dispatch-work-order"})

    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "UnsupportedBrokeredSchema"


def test_foundry_brokered_allows_single_value_root_enum_write_arguments():
    spec = _spec(tool_name="dispatch-work-order", brokered_class="write")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "enum": [{"operation": "create"}],
    }
    app = _app(spec)

    with TestClient(app) as client:
        resp = client.post("/responses", json={"input": "dispatch-work-order"})

    assert resp.status_code == 200
    assert json.loads(_call(resp.json())["arguments"]) == {"operation": "create"}


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


@pytest.mark.parametrize("brokered_class", ["write", "coordination"])
def test_foundry_brokered_refuses_nonliteral_optional_prompt_for_side_effecting_tools(brokered_class: str):
    spec = _spec(tool_name="dispatch-work-order", brokered_class=brokered_class)
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {"prompt": {"type": "string"}},
    }
    app = _app(spec)

    with TestClient(app) as client:
        resp = client.post("/responses", json={"input": "dispatch-work-order with arbitrary payload"})

    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "UnsupportedBrokeredSchema"


def test_foundry_brokered_allows_literal_optional_prompt_for_write_tools():
    spec = _spec(tool_name="dispatch-work-order", brokered_class="write")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {"prompt": {"type": "string", "const": "fixed"}},
    }
    app = _app(spec)

    with TestClient(app) as client:
        resp = client.post("/responses", json={"input": "dispatch-work-order with arbitrary payload"})

    assert resp.status_code == 200
    assert json.loads(_call(resp.json())["arguments"]) == {"prompt": "fixed"}


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


def test_foundry_brokered_model_loop_sends_placeholder_api_key_when_auth_is_omitted(monkeypatch):
    captured_headers: dict[str, str] = {}

    class FakeClient:
        def __init__(self, *, headers: dict[str, str], timeout: int) -> None:
            assert timeout == 60
            captured_headers.update(headers)

        async def post(self, url: str, *, json: dict[str, Any]) -> httpx.Response:
            request = httpx.Request("POST", url, json=json)
            return httpx.Response(
                200,
                request=request,
                json=_chat_response({"role": "assistant", "content": "done"}),
            )

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(foundry_model_loop_module.httpx, "AsyncClient", FakeClient)
    app = _app(_spec(tool_name="check-network-telemetry"), brokered_model_loop_enabled=True)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "check-network-telemetry"})

    assert response.status_code == 200, response.text
    assert captured_headers["Authorization"] == f"Bearer {NO_AUTH_API_KEY}"


@pytest.mark.parametrize(
    "usage",
    [
        {"prompt_tokens": {"unexpected": 1}},
        {"completion_tokens": "not-a-number"},
        {"total_tokens": 1.5},
    ],
)
def test_foundry_brokered_model_loop_normalizes_malformed_usage(usage: dict[str, Any]):
    fake = _FakeChatTransport(
        [
            {
                "choices": [{"message": {"role": "assistant", "content": "done"}}],
                "usage": usage,
            }
        ]
    )
    app = _model_loop_app(_spec(tool_name="check-network-telemetry"), fake)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "check-network-telemetry"})

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "InvalidModelResponse"


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
        final = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_continuation(initial.json()["id"], call["call_id"], {"approved": True, "output": {"status": "healthy"}}),
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


def test_foundry_brokered_model_loop_preserves_total_only_usage_across_resume():
    spec = _spec(tool_name="check-network-telemetry")
    initial_model_response = _chat_response(
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
    initial_model_response["usage"] = {"total_tokens": 5}
    final_model_response = _chat_response({"role": "assistant", "content": "done"})
    final_model_response["usage"] = {"total_tokens": 7}
    app = _model_loop_app(spec, _FakeChatTransport([initial_model_response, final_model_response]))

    with TestClient(app) as client:
        initial = client.post("/responses", json={"input": "check-network-telemetry"})
        call = _call(initial.json())
        final = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_continuation(initial.json()["id"], call["call_id"], {"approved": True, "output": {"ok": True}}),
        )

    assert initial.json()["usage"] == {"input_tokens": 0, "output_tokens": 0, "total_tokens": 5}
    assert final.status_code == 200, final.text
    assert final.json()["usage"] == {"input_tokens": 0, "output_tokens": 0, "total_tokens": 12}


def test_foundry_brokered_model_loop_rejects_noncanonical_denied_output_before_resume(tmp_path):
    state_file = tmp_path / "responses-state.json"
    spec = _spec(tool_name="dispatch-work-order", brokered_class="write")
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
                            "function": {"name": "dispatch-work-order", "arguments": "{}"},
                        }
                    ],
                }
            )
        ]
    )
    app = _model_loop_app(spec, fake, response_state_file=state_file)

    with TestClient(app) as client:
        initial = client.post("/responses", json={"input": "dispatch-work-order"})
        call = _call(initial.json())
        persisted_before = state_file.read_bytes()
        denied = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_continuation(
                initial.json()["id"],
                call["call_id"],
                {
                    "approved": False,
                    "error": {"code": "approval_declined", "message": "denied"},
                    "output": {"sensitiveDiagnostic": "must-not-reach-model"},
                },
            ),
        )

    assert denied.status_code == 400
    assert denied.json()["error"]["code"] == "invalid_function_call_output"
    assert state_file.read_bytes() == persisted_before
    assert len(fake.requests) == 1


def test_foundry_brokered_model_loop_rejects_oversized_output_before_resume_or_state_change(tmp_path):
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
            )
        ]
    )
    app = _model_loop_app(
        spec,
        fake,
        response_state_file=state_file,
        max_brokered_output_bytes=128,
    )

    with TestClient(app) as client:
        initial = client.post("/responses", json={"input": "check-network-telemetry"})
        call = _call(initial.json())
        persisted_before = state_file.read_bytes()
        oversized = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_continuation(
                initial.json()["id"],
                call["call_id"],
                {"approved": True, "output": {"blob": "x" * 256}},
            ),
        )

    assert oversized.status_code == 413
    assert oversized.json()["error"]["code"] == "brokered_output_too_large"
    assert state_file.read_bytes() == persisted_before
    assert len(fake.requests) == 1


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


def test_foundry_brokered_file_state_is_written_with_private_permissions(tmp_path):
    state_file = tmp_path / "responses-state.json"
    app = _app(response_state_file=state_file)

    with TestClient(app) as client:
        _start(client)

    assert state_file.exists()
    assert state_file.stat().st_mode & 0o777 == 0o600


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


def test_foundry_brokered_model_loop_rejects_object_valued_tool_arguments():
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
                            "function": {
                                "name": "check-network-telemetry",
                                "arguments": {"id": 9007199254740993.0},
                            },
                        }
                    ],
                }
            )
        ]
    )
    app = _model_loop_app(_spec(tool_name="check-network-telemetry"), fake)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "check-network-telemetry"})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "InvalidToolArguments"


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
