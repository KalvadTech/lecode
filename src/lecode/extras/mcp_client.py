"""The MCP client: connects configured + auto servers and exposes their tools.

Uses the official ``mcp`` SDK over stdio and streamable-HTTP. Each server is
connected lazily at session start with a ~10s budget; failures are isolated
per server (one bad server never blocks the others or startup). Discovered
tools are registered as lecode tools named ``mcp:<server>:<tool>`` — the
permission system already treats exa/context7/grep-app as read-equivalent,
every other MCP tool falls back to the mode default (Allow in yolo, Deny in
readonly).

Auto-configured servers (``[mcp] enable_exa`` / ``enable_context7``):

- **Exa** (default on, needs ``EXA_API_KEY``): the hosted streamable-HTTP
  endpoint ``https://mcp.exa.ai/mcp``. Auth assumption: the key goes as the
  documented ``?exaApiKey=`` query param *and* an ``Authorization: Bearer``
  header — Exa's exact header scheme is not pinned down in their docs.
- **context7** (default off): ``https://mcp.context7.com/mcp``, no auth.

OAuth 2.1 for HTTP servers (``[mcp.servers.<name>] auth = "oauth"``)
rides on the SDK's ``OAuthClientProvider`` (see ``lecode.extras.mcp_auth``):
automatic connections reuse cached credentials and refresh them silently;
interactive login is explicit via ``/mcp auth <name>``, which opens the
browser and serves the redirect on a loopback port. Static bearer tokens via
``[mcp.servers.<name>].headers`` remain the supported path for servers that
do not speak OAuth.

Failed tool calls get exactly one reconnect attempt (fresh session), then an
error result. Everything is fail-open: MCP trouble never blocks the agent.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from lecode.agent.tools.base import Tool, ToolContext, ToolRegistry, ToolResult
from lecode.config.models import Config, McpServerConfig
from lecode.extras.mcp_auth import CALLBACK_TIMEOUT_S

if TYPE_CHECKING:
    from mcp import ClientSession
    from mcp.types import Tool as McpToolDef

log = logging.getLogger(__name__)

#: ``ctx.extras`` key under which the connected manager is installed.
MCP_EXTRA = "mcp"

EXA_URL = "https://mcp.exa.ai/mcp"
CONTEXT7_URL = "https://mcp.context7.com/mcp"

#: Per-server startup budget.
CONNECT_TIMEOUT_S = 10.0

#: Budget for one interactive ``/mcp auth`` login: the loopback callback
#: timeout plus a margin for discovery, registration, and token exchange.
INTERACTIVE_AUTH_BUDGET_S = CALLBACK_TIMEOUT_S + 30.0


def auto_servers(config: Config, env: dict[str, str] | None = None) -> dict[str, McpServerConfig]:
    """The auto-configured servers implied by the ``[mcp]`` flags."""
    env = os.environ if env is None else env
    servers: dict[str, McpServerConfig] = {}
    if config.mcp.enable_exa:
        key = env.get("EXA_API_KEY", "").strip()
        if key:
            servers["exa"] = McpServerConfig(
                transport="http",
                url=f"{EXA_URL}?exaApiKey={key}",
                headers={"Authorization": f"Bearer {key}"},
            )
        else:
            log.debug("mcp: exa enabled but EXA_API_KEY is not set — skipping")
    if config.mcp.enable_context7:
        servers["context7"] = McpServerConfig(transport="http", url=CONTEXT7_URL)
    return servers


def all_server_configs(
    config: Config, env: dict[str, str] | None = None
) -> dict[str, McpServerConfig]:
    """Auto servers merged with ``[mcp.servers]`` (user config wins)."""
    merged = auto_servers(config, env)
    merged.update(config.mcp.servers)
    return merged


@dataclass(frozen=True)
class ServerStatus:
    """One server's state for ``/mcp``."""

    name: str
    state: Literal["connected", "failed", "disabled", "auth_required"]
    tools: int = 0
    error: str | None = None

    @property
    def auth_hint(self) -> str:
        """The canonical line for the ``auth_required`` state."""
        return f"{self.name}: authentication required — /mcp auth {self.name}"


@dataclass
class _Server:
    name: str
    config: McpServerConfig
    session: ClientSession | None = None
    stack: contextlib.AsyncExitStack | None = None
    tools: list[McpToolDef] = field(default_factory=list)
    error: str | None = None
    auth_required: bool = False
    interactive: bool = False
    #: called with the authorization URL during an interactive login (UI hint)
    announce: Callable[[str], Any] | None = None

    @property
    def status(self) -> ServerStatus:
        if not self.config.enabled:
            return ServerStatus(self.name, "disabled")
        if self.session is None:
            if self.auth_required:
                return ServerStatus(self.name, "auth_required", error=self.error)
            return ServerStatus(self.name, "failed", error=self.error)
        return ServerStatus(self.name, "connected", tools=len(self.tools))


def _clean_error(e: BaseException) -> str:
    """One short line; OAuth errors can wrap server HTML/JSON bodies.

    Server-controlled text (OAuth error descriptions, response bodies) must
    never reach the terminal raw: collapse whitespace, drop non-printable
    characters (ANSI/control escapes), and cap the length.
    """
    text = "".join(ch for ch in " ".join(str(e).split()) if ch.isprintable())
    return text[:300] + ("…" if len(text) > 300 else "")


def _unwrap_exceptions(e: BaseException) -> list[BaseException]:
    """Flatten ExceptionGroups (anyio wraps transport task errors in them)."""
    if isinstance(e, BaseExceptionGroup):
        flat: list[BaseException] = []
        for sub in e.exceptions:
            flat.extend(_unwrap_exceptions(sub))
        return flat
    return [e]


def _oauth_leaf(e: BaseException) -> BaseException | None:
    """The OAuthFlowError buried in ``e`` (or an ExceptionGroup), if any."""
    from mcp.client.auth import OAuthFlowError

    return next((x for x in _unwrap_exceptions(e) if isinstance(x, OAuthFlowError)), None)


def _connect_failure(e: BaseException) -> tuple[str, bool]:
    """(error text, needs interactive login) for one failed connection attempt.

    OAuth servers without (working) credentials must not look like a generic
    outage: ``/mcp auth`` is the fix, so they get the auth_required state.
    """
    oauth_leaf = _oauth_leaf(e)
    if oauth_leaf is not None:
        return _clean_error(oauth_leaf), True
    return f"{type(e).__name__}: {_clean_error(e)}", False


@contextlib.contextmanager
def _quiet_oauth_flow_logs() -> Iterator[None]:
    """Mute the SDK's expected dead-end OAuth logging during automatic connects.

    Without stored credentials the SDK still attempts the authorization-code
    grant, hits "No redirect handler provided", and logs it at ERROR with a
    full traceback — splashing raw stderr around the TUI at every startup
    until the user runs ``/mcp auth``. The failure is expected and already
    classified as auth_required; only the noise is lost.
    """
    loggers = [logging.getLogger(n) for n in ("mcp.client.auth", "mcp.client.auth.oauth2")]
    saved = [(lg, lg.level) for lg in loggers]
    for lg in loggers:
        lg.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        for lg, level in saved:
            lg.setLevel(level)


def _result_text(result: Any) -> str:
    """Flatten a CallToolResult's content parts to plain text."""
    parts = [getattr(part, "text", "") for part in result.content or []]
    return "\n".join(text for text in parts if text).strip()


class McpManager:
    """Owns MCP server sessions; bridges their tools into lecode."""

    def __init__(self, config: Config, ctx: ToolContext | None = None) -> None:
        self._config = config
        self._ctx = ctx
        self._servers: dict[str, _Server] = {}
        self._registry: ToolRegistry | None = None
        self._closed = False

    # -- connection ------------------------------------------------------------

    async def connect(self) -> None:
        """Connect every configured + auto server; failures stay per-server."""
        for name, server_config in all_server_configs(self._config).items():
            server = _Server(name, server_config)
            self._servers[name] = server
            if server_config.enabled:
                await self._connect_one(server)

    async def _connect_one(self, server: _Server) -> None:
        if not server.config.enabled:
            return
        quiet = server.config.auth == "oauth" and not server.interactive
        try:
            with _quiet_oauth_flow_logs() if quiet else contextlib.nullcontext():
                await asyncio.wait_for(self._open(server), timeout=CONNECT_TIMEOUT_S)
            server.error = None
            server.auth_required = False
            log.debug("mcp: %s connected (%d tools)", server.name, len(server.tools))
        except Exception as e:
            server.session = None
            server.error, server.auth_required = _connect_failure(e)
            if server.auth_required:
                log.debug("mcp: %s needs OAuth authentication: %s", server.name, e)
            else:
                log.debug("mcp: %s connect failed: %s", server.name, e)

    async def _open(self, server: _Server) -> None:
        from mcp import ClientSession

        stack = contextlib.AsyncExitStack()
        try:
            http_client = None
            if server.config.auth == "oauth" and server.config.transport == "http":
                http_client = await self._build_http_client(server, server.config, stack)
                await self._preflight_auth(server, http_client)
            read, write = await self._open_transport(server, stack, http_client)
            session = await stack.enter_async_context(
                ClientSession(read, write, read_timeout_seconds=server.config.timeout_s)
            )
            await session.initialize()
            result = await session.list_tools()
        except BaseException:
            # BaseException (not Exception) so a cancelled interactive login
            # still closes the loopback listener and HTTP client.
            await stack.aclose()
            raise
        server.session = session
        server.stack = stack
        server.tools = list(result.tools)

    async def _preflight_auth(self, server: _Server, http_client: Any) -> None:
        """One bare POST so the OAuth middleware completes the whole flow here.

        Doing the flow inside ``session.initialize()`` would lose the error:
        the SDK's streamable transport dispatches requests in a task group and
        answers a failed POST by cancelling the waiting sender, so the
        OAuthFlowError would surface only in logs, not to the caller. A plain
        httpx call propagates it directly; the status/body are irrelevant —
        the auth middleware already attached a token or raised.
        """
        from mcp.shared.inbound import MCP_PROTOCOL_VERSION_HEADER
        from mcp_types import LATEST_PROTOCOL_VERSION

        await http_client.post(
            server.config.url,
            headers={
                MCP_PROTOCOL_VERSION_HEADER: LATEST_PROTOCOL_VERSION,
                "Accept": "application/json, text/event-stream",
            },
            json={},
        )

    async def _open_transport(
        self,
        server: _Server,
        stack: contextlib.AsyncExitStack,
        http_client: Any | None = None,
    ) -> tuple[Any, Any]:
        config = server.config
        if config.transport == "http":
            if not config.url:
                raise ValueError(f"mcp server {server.name}: http transport needs a url")
            from mcp.client.streamable_http import streamable_http_client

            if http_client is None:
                http_client = await self._build_http_client(server, config, stack)
            return await stack.enter_async_context(
                streamable_http_client(config.url, http_client=http_client)
            )
        if not config.command:
            raise ValueError(f"mcp server {server.name}: stdio transport needs a command")
        from mcp import StdioServerParameters
        from mcp.client.stdio import get_default_environment, stdio_client

        env = {**get_default_environment(), **config.env} if config.env else None
        params = StdioServerParameters(
            command=config.command,
            args=config.args,
            **({"env": env} if env is not None else {}),
        )
        devnull = open(os.devnull, "w")  # noqa: SIM115 — closed via the stack
        stack.callback(devnull.close)
        return await stack.enter_async_context(stdio_client(params, errlog=devnull))

    async def _build_http_client(
        self, server: _Server, config: McpServerConfig, stack: contextlib.AsyncExitStack
    ) -> Any:
        """The httpx client for one server, with OAuth attached when configured."""
        from mcp.client.streamable_http import create_mcp_http_client

        url = config.url
        if not url:
            raise ValueError(f"mcp server {server.name}: http transport needs a url")
        headers = config.headers or None
        if config.auth != "oauth":
            return create_mcp_http_client(headers=headers)
        if config.transport != "http":
            raise ValueError(f"mcp server {server.name}: auth='oauth' requires transport='http'")
        if any(key.lower() == "authorization" for key in (config.headers or {})):
            raise ValueError(
                f"mcp server {server.name}: auth='oauth' conflicts with a static "
                "Authorization header — drop the header, OAuth manages Authorization"
            )
        from lecode.extras.mcp_auth import (
            FileTokenStorage,
            LoopbackAuthCallback,
            make_oauth_provider,
        )

        storage = FileTokenStorage(url)
        loopback = None
        if server.interactive:
            loopback = await LoopbackAuthCallback.open(storage, announce=server.announce)
            stack.push_async_callback(loopback.aclose)
        return create_mcp_http_client(
            headers=headers, auth=await make_oauth_provider(url, storage, loopback)
        )

    async def _close_server(self, server: _Server) -> None:
        if server.stack is not None:
            try:
                await server.stack.aclose()
            except Exception as e:
                log.debug("mcp: %s close failed: %s", server.name, e)
        server.session = None
        server.stack = None
        server.tools = []

    # -- tools -----------------------------------------------------------------

    def tool_wrappers(self, server_name: str | None = None) -> list[Tool]:
        """lecode tools wrapping connected servers' discovered tools."""
        return [
            McpTool(self, server.name, tool)
            for server in self._servers.values()
            if server.session is not None and (server_name is None or server.name == server_name)
            for tool in server.tools
        ]

    def _sync_tools(self, name: str | None = None) -> None:
        """Align the registry with reality: register new wrappers, drop stale ones.

        Called after every (re)connect and after logout. ``None`` covers the
        initial attach; a server name covers one-server reconnect/auth.
        """
        if self._registry is None:
            return
        affected = [sn for sn in self._servers if name in (None, sn)]
        prefixes = tuple(f"mcp:{sn}:" for sn in affected)
        wrappers = self.tool_wrappers(name)
        wanted = {tool.name for tool in wrappers}
        for existing in self._registry.names():
            if existing.startswith(prefixes) and existing not in wanted:
                self._registry.unregister(existing)
        for wrapper in wrappers:
            self._registry.register(wrapper)

    async def call(self, server_name: str, tool_name: str, args: dict[str, Any]) -> ToolResult:
        """Call an MCP tool; one reconnect attempt on connection failure."""
        server = self._servers.get(server_name)
        if server is None:
            return ToolResult(f"error: unknown MCP server: {server_name}", is_error=True)
        if not server.config.enabled:
            return ToolResult(f"error: MCP server '{server_name}' is disabled", is_error=True)
        if server.session is not None:
            try:
                return await self._call_once(server, tool_name, args)
            except Exception as e:
                log.debug("mcp: %s:%s call failed, reconnecting: %s", server_name, tool_name, e)
        await self._close_server(server)
        await self._connect_one(server)
        self._sync_tools(server_name)
        if server.session is None:
            return ToolResult(
                f"error: MCP server '{server_name}' unavailable: {server.error or 'not connected'}",
                is_error=True,
            )
        try:
            return await self._call_once(server, tool_name, args)
        except Exception as e:
            log.debug("mcp: %s:%s failed after reconnect: %s", server_name, tool_name, e)
            return ToolResult(f"error: MCP call failed: {type(e).__name__}: {e}", is_error=True)

    async def _call_once(self, server: _Server, tool_name: str, args: dict[str, Any]) -> ToolResult:
        assert server.session is not None
        result = await server.session.call_tool(
            tool_name, args, read_timeout_seconds=server.config.timeout_s
        )
        text = _result_text(result)
        is_error = getattr(result, "is_error", getattr(result, "isError", False))
        if is_error:
            return ToolResult(f"error: {text or 'MCP tool failed'}", is_error=True)
        return ToolResult(text or "(no output)")

    # -- management (/mcp, shutdown) ----------------------------------------------

    def status(self) -> list[ServerStatus]:
        """Per-server state, sorted by name."""
        return sorted((server.status for server in self._servers.values()), key=lambda s: s.name)

    def bind_registry(self, registry: ToolRegistry) -> None:
        """Give the manager the registry it keeps in sync (reconnect/auth/logout)."""
        self._registry = registry

    def server_tools(self, name: str) -> list[McpToolDef] | None:
        """One server's discovered tools; None when unknown."""
        server = self._servers.get(name)
        return None if server is None else list(server.tools)

    async def reconnect(self, name: str) -> ServerStatus | None:
        """Drop and re-establish one server's session."""
        server = self._servers.get(name)
        if server is None:
            return None
        await self._close_server(server)
        if server.config.enabled:
            await self._connect_one(server)
        self._sync_tools(name)
        return server.status

    async def authenticate(
        self, name: str, announce: Callable[[str], Any] | None = None
    ) -> ServerStatus | None:
        """Interactive OAuth login for one server (opens the browser).

        Runs outside the normal per-server connect budget: the user needs time
        to approve in the browser. ``announce`` is called with the
        authorization URL just before the browser opens, so the UI can show
        the link (a different browser can be used with it). Non-OAuth servers
        get a plain error state.
        """
        server = self._servers.get(name)
        if server is None:
            return None
        if server.config.auth != "oauth":
            server.error = "this server is not configured with auth = 'oauth'"
            return server.status
        await self._close_server(server)
        server.error = None
        server.auth_required = False
        server.interactive = True
        server.announce = announce
        try:
            await asyncio.wait_for(self._open(server), timeout=INTERACTIVE_AUTH_BUDGET_S)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            server.session = None
            server.error, server.auth_required = _connect_failure(e)
            if not server.auth_required and isinstance(e, TimeoutError):
                server.error = f"timed out (limit {INTERACTIVE_AUTH_BUDGET_S}s)"
            log.debug("mcp: %s authentication failed: %s", name, e)
        finally:
            server.interactive = False
            server.announce = None
        self._sync_tools(name)
        return server.status

    async def logout(self, name: str) -> ServerStatus | None:
        """Drop one server's session and its persisted OAuth credentials."""
        server = self._servers.get(name)
        if server is None:
            return None
        if server.config.auth != "oauth" or not server.config.url:
            server.error = "logout applies only to OAuth servers"
            return server.status
        await self._close_server(server)
        from lecode.extras.mcp_auth import FileTokenStorage

        await FileTokenStorage(server.config.url).clear()
        server.error = None
        server.auth_required = True
        self._sync_tools(name)
        return server.status

    async def shutdown(self) -> None:
        """Close every server session; idempotent, never raises."""
        if self._closed:
            return
        self._closed = True
        for server in self._servers.values():
            await self._close_server(server)


class McpTool(Tool):
    """A lecode tool delegating to one MCP server's tool."""

    def __init__(self, manager: McpManager, server_name: str, mcp_tool: McpToolDef) -> None:
        super().__init__(
            name=f"mcp:{server_name}:{mcp_tool.name}",
            description=(mcp_tool.description or f"MCP tool {mcp_tool.name}").strip(),
            parameters=mcp_tool.input_schema or {"type": "object", "properties": {}},
        )
        self._manager = manager
        self._server_name = server_name
        self._tool_name = mcp_tool.name

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return await self._manager.call(self._server_name, self._tool_name, args)


async def attach_mcp(registry: ToolRegistry, ctx: ToolContext) -> McpManager:
    """Connect MCP servers and register their tools into ``registry``.

    Never raises: per-server failures are isolated inside the manager, and a
    wholesale failure just leaves zero MCP tools registered. The manager is
    always installed under ``ctx.extras["mcp"]`` so ``/mcp`` can report, and
    keeps the registry handle so later reconnects/auth/logout can update it.
    """
    manager = McpManager(ctx.config, ctx)
    manager.bind_registry(registry)
    ctx.extras[MCP_EXTRA] = manager
    try:
        await manager.connect()
    except Exception as e:  # belt-and-braces; connect() isolates per server
        log.debug("mcp: connect failed: %s", e)
    manager._sync_tools()
    return manager
