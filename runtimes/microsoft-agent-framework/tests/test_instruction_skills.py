from __future__ import annotations

import asyncio

from agent_framework import BaseChatClient, ChatResponse, Content, FunctionInvocationLayer, Message

from agentkit_serve import agent_factory
from agentkit_serve_common.config import AgentSpec
from agentkit_serve_common.conversation import RunRequest
from agentkit_serve_common.skills import _Skill, SkillCatalog


def test_governed_skills_use_snapshot_and_expose_only_load_skill(monkeypatch):
    content = "---\nname: inspection\ndescription: Inspect equipment.\n---\nUse the authorized lookup tool.\n"
    spec = AgentSpec.model_validate(
        {
            "abiVersion": "v0",
            "metadata": {"name": "inspection-agent"},
            "model": {
                "provider": "openai-compatible",
                "baseURL": "http://127.0.0.1:43123/v1",
                "name": "test-model",
            },
            "instructions": "Consult the relevant skill before answering.",
            "context": {
                "providers": [
                    {
                        "type": "skills",
                        "source": "filesystem",
                        "path": "/agent/skills",
                    }
                ]
            },
            "expose": {"openai": True, "port": 8080},
        }
    )
    spec._packaged_skill_catalog = SkillCatalog((_Skill("inspection", "Inspect equipment.", content),))

    class SkillClient(FunctionInvocationLayer, BaseChatClient):
        def __init__(self):
            super().__init__()
            self.requests = 0

        async def _inner_get_response(self, *, messages, stream, options, **kwargs):
            self.requests += 1
            assert [tool.name for tool in options.get("tools", [])] == ["load_skill"]
            if self.requests == 1:
                assert "<name>inspection</name>" in options["instructions"]
                response = Content.from_function_call(
                    call_id="skill-call", name="load_skill", arguments={"skill_name": "inspection"}
                )
            else:
                assert self.requests == 2
                assert any(
                    item.type == "function_result" and content in str(item.result)
                    for message in messages
                    for item in message.contents
                )
                response = Content.from_text("The inspection instructions are loaded.")
            return ChatResponse(messages=[Message(role="assistant", contents=[response])])

    def no_native_provider(*args, **kwargs):
        raise AssertionError("governed skills must not use the filesystem or native SkillsProvider")

    monkeypatch.setattr(agent_factory, "FileSkillsSource", no_native_provider)
    monkeypatch.setattr(agent_factory, "SkillsProvider", no_native_provider)
    client = SkillClient()
    monkeypatch.setattr(agent_factory, "build_client", lambda _: client)

    async def exercise():
        events = []

        async def observe(event):
            events.append(event)

        async with agent_factory.MAFRuntime(spec) as runtime:
            result = await runtime.run(RunRequest("Inspect the equipment.", on_tool_event=observe))
        assert result.text == "The inspection instructions are loaded."
        assert [(event.tool_name, event.status) for event in events] == [
            ("load_skill", "in_progress"),
            ("load_skill", "completed"),
        ]

    asyncio.run(exercise())
