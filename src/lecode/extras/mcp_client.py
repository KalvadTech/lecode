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

OAuth for HTTP MCP servers is intentionally out of scope: the SDK supports
it but it needs an interactive browser flow; bearer tokens via
``[mcp.servers.<name>].headers`` are the supported path.

Failed tool calls get exactly one reconnect attempt (fresh session), then an
error result. Everything is fail-open: MCP trouble never blocks the agent.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from lecode.agent.tools.base import Tool, ToolContext, ToolRegistry, ToolResult
from lecode.config.models import Config, McpServerConfig

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
    state: str  # "connected" | "failed" | "disabled"
    tools: int = 0
    error: str | None = None


@dataclass
class _Server:
    name: str
    config: McpServerConfig
    session: ClientSession | None = None
    stack: contextlib.AsyncExitStack | None = None
    tools: list[McpToolDef] = field(default_factory=list)
    error: str | None = None

    @property
    def status(self) -> ServerStatus:
        if not self.config.enabled:
            return ServerStatus(self.name, "disabled")
        if self.session is None:
            return ServerStatus(self.name, "failed", error=self.error)
        return ServerStatus(self.name, "connected", tools=len(self.tools))


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
        try:
            await asyncio.wait_for(self._open(server), timeout=CONNECT_TIMEOUT_S)
            server.error = None
            log.debug("mcp: %s connected (%d tools)", server.name, len(server.tools))
        except Exception as e:
            server.session = None
            server.error = f"{type(e).__name__}: {e}"
            log.debug("mcp: %s connect failed: %s", server.name, e)

    async def _open(self, server: _Server) -> None:
        from mcp import ClientSession

        stack = contextlib.AsyncExitStack()
        try:
            read, write = await self._open_transport(server, stack)
            session = await stack.enter_async_context(
                ClientSession(read, write, read_timeout_seconds=server.config.timeout_s)
            )
            await session.initialize()
            result = await session.list_tools()
        except Exception:
            await stack.aclose()
            raise
        server.session = session
        server.stack = stack
        server.tools = list(result.tools)

    async def _open_transport(
        self, server: _Server, stack: contextlib.AsyncExitStack
    ) -> tuple[Any, Any]:
        config = server.config
        if config.transport == "http":
            if not config.url:
                raise ValueError(f"mcp server {server.name}: http transport needs a url")
            from mcp.client.streamable_http import (
                create_mcp_http_client,
                streamable_http_client,
            )

            http_client = create_mcp_http_client(headers=config.headers or None)
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

    def tool_wrappers(self) -> list[Tool]:
        """lecode tools wrapping every connected server's discovered tools."""
        return [
            McpTool(self, server.name, tool)
            for server in self._servers.values()
            if server.session is not None
            for tool in server.tools
        ]

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
    always installed under ``ctx.extras["mcp"]`` so ``/mcp`` can report.
    """
    manager = McpManager(ctx.config, ctx)
    ctx.extras[MCP_EXTRA] = manager
    try:
        await manager.connect()
    except Exception as e:  # belt-and-braces; connect() isolates per server
        log.debug("mcp: connect failed: %s", e)
    for wrapper in manager.tool_wrappers():
        registry.register(wrapper)
    return manager
