"""Build a Microsoft Agent Framework (MAF) agent from a validated AgentSpec.

This module is the framework-specific translation layer behind the frozen
``/agent/agent.yaml`` ABI. The ABI loader, the ``/v1`` facade, Foundry protocol
wrapper, and CLI live in ``agentkit_serve_common``.
"""

from __future__ import annotations

import asyncio
import importlib
import os
import time
from contextlib import AsyncExitStack
from types import TracebackType
from urllib.parse import urlsplit

from agent_framework import (
    Agent,
    AgentSession,
    FileSkillsSource,
    MCPSkillsSource,
    MCPStdioTool,
    MCPStreamableHTTPTool,
    Message,
    SkillsProvider,
)
from agent_framework.openai import OpenAIChatCompletionClient
from agentkit_serve_common.adapter_support import (
    FORWARDED_ROLES,
    AgentBuildError,
    declared_tool_env,
    normalize_agent_run_error,
    positive_int_env,
    resolve_api_key,
    resolve_workload_identity_token,
    resolve_tool_headers,
    resolve_tool_url,
    same_origin_mcp_httpx_client_factory,
    split_tool_command,
    upstream_status_code,
)
from agentkit_serve_common.config import AgentSpec, ContextProviderSpec, ToolSpec
from agentkit_serve_common.conversation import RunRequest
from agentkit_serve_common.runtime import RunResult, RuntimeSession

_AUTH_WORKLOAD_IDENTITY = "workload-identity-token"
_CONTEXT_TYPE_SEARCH = "search"
_CONTEXT_TYPE_SKILLS = "skills"
_CONTEXT_TYPE_MEMORY = "memory"
_CONTEXT_SOURCE_FILESYSTEM = "filesystem"
_CONTEXT_SOURCE_MCP = "mcp"
_DEFAULT_SEARCH_AUDIENCE = "https://search.azure.com/.default"
_DEFAULT_FOUNDRY_AUDIENCE = "https://ai.azure.com/.default"
_DEFAULT_SESSION_CACHE_MAX = 256


def _mcp_request_timeout() -> int | None:
    """MCP request timeout (seconds), overridable via ``AGENTKIT_MCP_TIMEOUT``."""
    return positive_int_env(default=None)


def _remote_mcp_timeout() -> float:
    """Bound network-backed MCP calls even when stdio timeout is left uncapped."""
    return float(_mcp_request_timeout() or 120)


def _resolve_api_key(spec: AgentSpec) -> str:
    """Compatibility wrapper around the shared adapter support module."""
    return resolve_api_key(spec)


def _project_endpoint_from_openai_base_url(base_url: str) -> str:
    """Extract a Foundry project endpoint from ``.../openai/v1`` base URLs."""
    marker = "/openai/v1"
    if marker not in base_url:
        raise AgentBuildError(
            "model.auth workload-identity-token for the MAF runtime requires "
            "model.baseURL to be a Foundry project OpenAI endpoint ending in /openai/v1"
        )
    return base_url.split(marker, 1)[0].rstrip("/")


def _env_required(name: str | None, *, field: str) -> str:
    if not name:
        raise AgentBuildError(f"{field} is required")
    value = os.environ.get(name)
    if value is None or value == "":
        raise AgentBuildError(
            f"{field} {name!r} is declared in agent.yaml but env var {name!r} is not set; "
            "inject it at runtime"
        )
    return value


class _BearerTokenCredential:
    """Tiny Azure TokenCredential over AgentKit's generic workload token hook."""

    def __init__(self, audience: str) -> None:
        self._audience = audience

    def get_token(self, *scopes: str, **kwargs):  # noqa: ANN003 - Azure credential protocol
        credentials_mod = importlib.import_module("azure.core.credentials")
        access_token = getattr(credentials_mod, "AccessToken")

        audience = scopes[0] if scopes else self._audience
        token = resolve_workload_identity_token(audience or self._audience)
        return access_token(token, int(time.time()) + 300)


class _AsyncBearerTokenCredential:
    """Async TokenCredential over AgentKit's generic workload token hook."""

    def __init__(self, audience: str) -> None:
        self._audience = audience

    async def get_token(self, *scopes: str, **kwargs):  # noqa: ANN003 - Azure async credential protocol
        credentials_mod = importlib.import_module("azure.core.credentials")
        access_token = getattr(credentials_mod, "AccessToken")

        audience = scopes[0] if scopes else self._audience
        token = await asyncio.to_thread(resolve_workload_identity_token, audience or self._audience)
        return access_token(token, int(time.time()) + 300)


def _credential_for_context(
    provider: ContextProviderSpec,
    *,
    default_audience: str,
    async_credential: bool = False,
):
    auth = provider.auth
    if auth is None or auth.type == _AUTH_WORKLOAD_IDENTITY:
        audience = auth.audience if auth and auth.audience else default_audience
        # If AgentKit's generic token hook is configured, use it. Otherwise fall
        # back to DefaultAzureCredential for local az login / hosted MI flows.
        if (
            os.environ.get("AGENTKIT_WORKLOAD_IDENTITY_TOKEN")
            or os.environ.get("AGENTKIT_WORKLOAD_IDENTITY_TOKEN_COMMAND")
        ):
            return _AsyncBearerTokenCredential(audience) if async_credential else _BearerTokenCredential(audience)
        try:
            if async_credential:
                identity_mod = importlib.import_module("azure.identity.aio")
            else:
                identity_mod = importlib.import_module("azure.identity")
            credential_type = getattr(identity_mod, "DefaultAzureCredential")
        except (ImportError, AttributeError) as exc:  # pragma: no cover - dependency guard.
            raise AgentBuildError("context workload identity auth requires azure-identity") from exc
        return credential_type()
    raise AgentBuildError(f"context provider auth type {auth.type!r} is not supported by the MAF runtime")


def _memory_update_delay() -> int:
    return positive_int_env("AGENTKIT_MEMORY_UPDATE_DELAY", default=0) or 0


def _session_cache_max() -> int:
    return positive_int_env("AGENTKIT_SESSION_CACHE_MAX", default=_DEFAULT_SESSION_CACHE_MAX) or _DEFAULT_SESSION_CACHE_MAX


def build_client(spec: AgentSpec):
    """Construct the chat client for the configured model auth mode."""
    auth = spec.model.auth
    if auth is not None and auth.type == _AUTH_WORKLOAD_IDENTITY:
        if token := os.environ.get("AGENTKIT_MODEL_WORKLOAD_IDENTITY_TOKEN"):
            return OpenAIChatCompletionClient(
                model=spec.model.name,
                base_url=spec.model.base_url,
                api_key=token,
            )
        try:
            from agent_framework.foundry import FoundryChatClient
            from azure.identity import DefaultAzureCredential
        except ImportError as exc:  # pragma: no cover - dependency guard.
            raise AgentBuildError(
                "model workload identity auth requires agent-framework-foundry and azure-identity"
            ) from exc
        return FoundryChatClient(
            project_endpoint=_project_endpoint_from_openai_base_url(spec.model.base_url),
            model=spec.model.name,
            credential=DefaultAzureCredential(),
        )

    return OpenAIChatCompletionClient(
        model=spec.model.name,
        base_url=spec.model.base_url,
        api_key=resolve_api_key(spec),
    )


def _tool_env(tool: ToolSpec) -> dict[str, str]:
    """Compatibility wrapper around the shared secret-safe tool env projection."""
    return declared_tool_env(tool)


def build_tool(tool: ToolSpec):
    """Create a stdio or Streamable HTTP MCP server for one tool spec."""
    timeout = _mcp_request_timeout()
    if tool.url_env:
        from httpx import AsyncClient, URL

        url = resolve_tool_url(tool)
        target_url = URL(url)
        target_origin = (target_url.scheme, target_url.host, target_url.port)
        remote_timeout = _remote_mcp_timeout()

        async def inject_headers(request):  # noqa: ANN001
            request_origin = (request.url.scheme, request.url.host, request.url.port)
            if request_origin != target_origin:
                return
            for key, value in (await asyncio.to_thread(resolve_tool_headers, tool)).items():
                request.headers[key] = value

        kwargs: dict[str, object] = {
            "name": tool.name,
            "url": url,
            "tool_name_prefix": tool.name,
            "load_prompts": False,
            "http_client": AsyncClient(
                event_hooks={"request": [inject_headers]},
                follow_redirects=False,
                timeout=remote_timeout,
            ),
        }
        kwargs["request_timeout"] = int(remote_timeout)
        return MCPStreamableHTTPTool(**kwargs)

    command, args = split_tool_command(tool, example='["uvx", "mcp-server-fetch"]')
    kwargs = {
        "name": tool.name,
        "command": command,
        "args": args,
        "env": declared_tool_env(tool),
        "tool_name_prefix": tool.name,
    }
    if timeout is not None:
        kwargs["request_timeout"] = timeout
    return MCPStdioTool(**kwargs)


def build_agent(spec: AgentSpec, *, context_providers=None) -> Agent:
    """Assemble the MAF agent: client + system prompt + tools + context."""
    return Agent(
        client=build_client(spec),
        instructions=spec.instructions,
        name=spec.metadata.name,
        tools=[build_tool(t) for t in spec.tools],
        context_providers=context_providers,
    )


class MAFRuntime:
    """RuntimeSession Adapter around a Microsoft Agent Framework Agent."""

    def __init__(self, spec: AgentSpec) -> None:
        self.spec = spec
        self.stack = AsyncExitStack()
        self.agent: Agent | None = None
        self.sessions: dict[str, AgentSession] = {}
        self.session_cache_max = _session_cache_max()

    async def __aenter__(self) -> RuntimeSession:
        try:
            context_providers = await self._build_context_providers()
            self.agent = build_agent(self.spec, context_providers=context_providers)
            await self.agent.__aenter__()
            return self
        except Exception:
            await self.stack.aclose()
            raise

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        agent_result = None
        if self.agent is not None:
            agent_result = await self.agent.__aexit__(exc_type, exc, tb)
            self.agent = None
        await self.stack.aclose()
        return agent_result

    async def run(self, request: RunRequest) -> RunResult:
        if self.agent is None:
            raise AgentBuildError("MAF runtime session is not initialized")
        session = self._session_for(request.session_id)
        return await run_agent(self.agent, request, session=session)

    def _session_for(self, session_id: str | None) -> AgentSession | None:
        if not session_id:
            return None
        session = self.sessions.pop(session_id, None)
        if session is None:
            session = AgentSession(session_id=session_id)
        self.sessions[session_id] = session
        while len(self.sessions) > self.session_cache_max:
            self.sessions.pop(next(iter(self.sessions)))
        return session

    async def _build_context_providers(self):
        providers = []
        for provider in self.spec.context.providers:
            if provider.type == _CONTEXT_TYPE_SEARCH:
                providers.append(await self._build_search_provider(provider))
            elif provider.type == _CONTEXT_TYPE_SKILLS:
                if provider.source == _CONTEXT_SOURCE_FILESYSTEM:
                    providers.append(SkillsProvider(FileSkillsSource(provider.path)))
                elif provider.source == _CONTEXT_SOURCE_MCP:
                    providers.append(await self._build_mcp_skills_provider(provider))
            elif provider.type == _CONTEXT_TYPE_MEMORY:
                providers.append(self._build_memory_provider(provider))
        return providers or None

    async def _build_search_provider(self, provider: ContextProviderSpec):
        try:
            azure_mod = importlib.import_module("agent_framework.azure")
            search_provider = getattr(azure_mod, "AzureAISearchContextProvider")
        except (ImportError, AttributeError) as exc:
            raise AgentBuildError(
                "search context provider requires agent-framework-azure-ai-search in the MAF runtime"
            ) from exc

        endpoint = _env_required(provider.endpoint_env, field="context.providers[].endpointEnv")
        index = _env_required(provider.index_env, field="context.providers[].indexEnv")
        credential = _credential_for_context(
            provider,
            default_audience=_DEFAULT_SEARCH_AUDIENCE,
            async_credential=True,
        )
        close = getattr(credential, "close", None)
        if callable(close):
            self.stack.push_async_callback(close)
        return search_provider(
            source_id=provider.name or "search",
            endpoint=endpoint,
            index_name=index,
            credential=credential,
        )

    def _build_memory_provider(self, provider: ContextProviderSpec):
        try:
            foundry_mod = importlib.import_module("agent_framework.foundry")
            memory_provider = getattr(foundry_mod, "FoundryMemoryProvider")
        except (ImportError, AttributeError) as exc:
            raise AgentBuildError(
                "memory context provider requires agent-framework-foundry in the MAF runtime"
            ) from exc

        endpoint = _env_required(provider.endpoint_env, field="context.providers[].endpointEnv")
        store_name = _env_required(provider.store_name_env, field="context.providers[].storeNameEnv")
        return memory_provider(
            source_id=provider.name or "memory",
            project_endpoint=endpoint,
            credential=_credential_for_context(provider, default_audience=_DEFAULT_FOUNDRY_AUDIENCE),
            memory_store_name=store_name,
            scope=os.environ.get("AGENTKIT_MEMORY_SCOPE") or provider.name or "default",
            update_delay=_memory_update_delay(),
        )

    async def _build_mcp_skills_provider(self, provider: ContextProviderSpec):
        tool = next((t for t in self.spec.tools if t.name == provider.tool_ref), None)
        if tool is None:
            raise AgentBuildError(f"skills provider references unknown toolRef {provider.tool_ref!r}")
        if not tool.url_env:
            raise AgentBuildError("MCP skills provider currently requires a streamable-http MCP toolRef")

        from mcp.client.session import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        url = resolve_tool_url(tool)
        http_client = same_origin_mcp_httpx_client_factory(
            tool,
            url,
            timeout=_remote_mcp_timeout(),
        )()
        await self.stack.enter_async_context(http_client)
        read, write, _ = await self.stack.enter_async_context(
            streamable_http_client(url=url, http_client=http_client)
        )
        session = await self.stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        return SkillsProvider(MCPSkillsSource(client=session))


def build_runtime(spec: AgentSpec) -> MAFRuntime:
    """Build the runtime session consumed by the shared server."""
    return MAFRuntime(spec)


def _status_of(exc: Exception) -> int:
    """Compatibility wrapper around shared upstream status extraction."""
    return upstream_status_code(exc)


def _result_text(result: object) -> str:
    """Extract the final assistant text from a MAF ``AgentResponse``."""
    text = getattr(result, "text", None)
    return text if isinstance(text, str) else str(result)


def _result_usage(result: object) -> dict[str, int]:
    """Map MAF ``usage_details`` to the OpenAI usage block (zeros if unknown)."""
    details = getattr(result, "usage_details", None)
    get = getattr(details, "get", None)
    if not callable(get):
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    prompt_tokens = int(get("input_token_count", 0) or 0)
    completion_tokens = int(get("output_token_count", 0) or 0)
    total = get("total_token_count", None)
    total_tokens = int(total) if total is not None else prompt_tokens + completion_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _to_messages(request: RunRequest) -> list[Message]:
    """Map a neutral RunRequest to MAF messages."""
    messages: list[Message] = []
    for turn in request.history:
        if turn.role in FORWARDED_ROLES and turn.text:
            messages.append(Message(role=turn.role, contents=[turn.text]))
    messages.append(Message(role="user", contents=[request.prompt]))
    return messages


async def run_agent(agent: Agent, request: RunRequest, *, session: AgentSession | None = None) -> RunResult:
    """Run the MAF agent and return the neutral result shape."""
    messages = _to_messages(request)
    try:
        result = await agent.run(messages, session=session)
    except Exception as exc:  # noqa: BLE001 — normalized for the façade
        raise normalize_agent_run_error(exc) from exc
    return RunResult(text=_result_text(result), usage=_result_usage(result))
