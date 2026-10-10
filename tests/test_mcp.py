"""Tests for the MCP client: manager, tool bridge, permissions, /mcp.

The mock server is a real MCP stdio subprocess (``mock_mcp_server.py``) built
on the SDK's MCPServer; one test also exercises streamable-HTTP against an
in-process uvicorn server on a random localhost port.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from lecode.agent.builder import build_runtime
from lecode.config.models import Config, McpServerConfig
from lecode.extras.mcp_client import (
    MCP_EXTRA,
    McpManager,
    attach_mcp,
)
from lecode.permission import Decision, PermissionChecker

MOCK = str(Path(__file__).parent / "mock_mcp_server.py")


def mock_server_config(**kwargs) -> McpServerConfig:
    return McpServerConfig(
        transport="stdio", command=sys.executable, args=[MOCK], timeout_s=5.0, **kwargs
    )


def mcp_config(**server_kwargs) -> Config:
    """A config with the mock as server "test"."""
    config = Config()
    config.mcp.servers["test"] = mock_server_config(**server_kwargs)
    return config


@pytest.fixture
async def manager():
    """A connected manager over the mock stdio server (shut down after)."""
    mgr = McpManager(mcp_config())
    await mgr.connect()
    yield mgr
    await mgr.shutdown()


# -- connect / discovery ---------------------------------------------------------


async def test_connect_discovers_tools(manager):
    status = manager.status()
    assert len(status) == 1
    assert status[0].name == "test"
    assert status[0].state == "connected"
    assert status[0].tools == 4


async def test_tool_wrappers_named_with_schemas(manager):
    wrappers = {tool.name: tool for tool in manager.tool_wrappers()}
    assert set(wrappers) == {
        "mcp:test:echo",
        "mcp:test:boom",
        "mcp:test:crash",
        "mcp:test:flaky",
    }
    echo = wrappers["mcp:test:echo"]
    assert "Echo the text back" in echo.description
    assert "text" in echo.parameters["properties"]


async def test_invocation_round_trip(manager):
    result = await manager.call("test", "echo", {"text": "hello"})
    assert not result.is_error
    assert result.content == "echo: hello"


async def test_tool_level_error_is_error_result(manager):
    result = await manager.call("test", "boom", {})
    assert result.is_error
    assert "boom" in result.content


async def test_unknown_server_call(manager):
    result = await manager.call("nope", "echo", {})
    assert result.is_error
    assert "unknown MCP server" in result.content


async def test_disabled_server_skipped():
    mgr = McpManager(mcp_config(enabled=False))
    try:
        await mgr.connect()
        status = mgr.status()[0]
        assert status.state == "disabled"
        assert mgr.tool_wrappers() == []
        result = await mgr.call("test", "echo", {"text": "hi"})
        assert result.is_error
        assert "disabled" in result.content
    finally:
        await mgr.shutdown()


async def test_set_enabled_disable_disconnects_and_refuses_calls(manager):
    status = await manager.set_enabled("test", False)
    assert status is not None
    assert status.state == "disabled"
    assert manager.tool_wrappers() == []
    result = await manager.call("test", "echo", {"text": "hi"})
    assert result.is_error
    assert "disabled" in result.content


async def test_set_enabled_enable_reconnects(manager):
    await manager.set_enabled("test", False)
    status = await manager.set_enabled("test", True)
    assert status is not None
    assert status.state == "connected"
    assert status.tools == 4
    result = await manager.call("test", "echo", {"text": "back"})
    assert result.content == "echo: back"


async def test_set_enabled_connects_a_config_disabled_server():
    mgr = McpManager(mcp_config(enabled=False))
    try:
        await mgr.connect()
        assert mgr.status()[0].state == "disabled"
        status = await mgr.set_enabled("test", True)
        assert status is not None
        assert status.state == "connected"
        assert {tool.name for tool in mgr.tool_wrappers()} == {
            "mcp:test:echo",
            "mcp:test:boom",
            "mcp:test:crash",
            "mcp:test:flaky",
        }
    finally:
        await mgr.shutdown()


async def test_set_enabled_unknown_server(manager):
    assert await manager.set_enabled("nope", False) is None
    assert await manager.set_enabled("nope", True) is None


async def test_bad_command_isolated(tmp_path):
    config = mcp_config()
    config.mcp.servers["broken"] = McpServerConfig(
        transport="stdio", command="no-such-binary-lecode-test"
    )
    mgr = McpManager(config)
    try:
        await mgr.connect()  # must not raise
        states = {s.name: s for s in mgr.status()}
        assert states["test"].state == "connected"
        assert states["broken"].state == "failed"
        assert states["broken"].error
        # the healthy server still works
        result = await mgr.call("test", "echo", {"text": "hi"})
        assert not result.is_error
    finally:
        await mgr.shutdown()


async def test_shutdown_idempotent(manager):
    await manager.shutdown()
    await manager.shutdown()


# -- crash / reconnect -------------------------------------------------------------


async def test_crash_fails_open_after_reconnect(manager):
    result = await manager.call("test", "crash", {})
    assert result.is_error
    assert "error" in result.content


async def test_reconnect_recovers_from_crash(tmp_path):
    state_file = tmp_path / "flaky-state"
    mgr = McpManager(mcp_config(env={"MOCK_MCP_STATE": str(state_file)}))
    try:
        await mgr.connect()
        # first call kills the server mid-request; the reconnect succeeds.
        result = await mgr.call("test", "flaky", {})
        assert not result.is_error
        assert "ok after 2 calls" in result.content
    finally:
        await mgr.shutdown()


async def test_reconnect_method(manager):
    status = await manager.reconnect("test")
    assert status is not None
    assert status.state == "connected"
    assert status.tools == 4
    assert await manager.reconnect("nope") is None
    # the fresh session works
    result = await manager.call("test", "echo", {"text": "again"})
    assert result.content == "echo: again"


# -- registry + permissions ----------------------------------------------------------


@pytest.fixture
async def runtime_with_mcp(tmp_path, monkeypatch):
    """A build_runtime Runtime with the mock server's tools registered."""
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    runtime = build_runtime(mcp_config(), tmp_path, auto_approve=True)
    manager = await attach_mcp(runtime.registry, runtime.ctx)
    yield runtime, manager
    await manager.shutdown()


async def test_attach_registers_tools(runtime_with_mcp):
    runtime, manager = runtime_with_mcp
    assert "mcp:test:echo" in runtime.registry.names()
    assert runtime.ctx.extras[MCP_EXTRA] is manager


async def test_set_enabled_syncs_the_registry(runtime_with_mcp):
    runtime, manager = runtime_with_mcp
    await manager.set_enabled("test", False)
    assert not any(n.startswith("mcp:test:") for n in runtime.registry.names())
    await manager.set_enabled("test", True)
    assert "mcp:test:echo" in runtime.registry.names()


async def test_dispatch_round_trip(runtime_with_mcp):
    runtime, _ = runtime_with_mcp
    message = await runtime.registry.dispatch(
        "call-1", "mcp:test:echo", '{"text": "via dispatch"}', runtime.ctx
    )
    assert message["role"] == "tool"
    assert message["content"] == "echo: via dispatch"


async def test_dispatch_denied_in_readonly_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    runtime = build_runtime(mcp_config(), tmp_path, mode="readonly")
    manager = await attach_mcp(runtime.registry, runtime.ctx)
    try:
        message = await runtime.registry.dispatch(
            "call-1", "mcp:test:echo", '{"text": "x"}', runtime.ctx
        )
        assert message["content"].startswith("denied")
    finally:
        await manager.shutdown()


def test_permission_ask_rule_for_regular_mcp_server():
    # the two modes never Ask by themselves; an ask rule gates the tool
    config = Config.model_validate(
        {"permissions": {"rules": {"ask": {"mcp:test:echo": [{"pattern": "*"}]}}}}
    )
    checker = PermissionChecker(config)
    assert checker.check("mcp:test:echo", {"text": "x"}).decision == Decision.ASK


def test_permission_allow_for_read_equiv_servers():
    checker = PermissionChecker(Config(), mode="readonly")
    assert checker.check("mcp:exa:web_search_exa", {"query": "x"}).decision == Decision.ALLOW
    assert checker.check("mcp:context7:resolve-library-id", {}).decision == Decision.ALLOW


def test_permission_readonly_allows_exa_denies_others():
    checker = PermissionChecker(Config(), mode="readonly")
    assert checker.check("mcp:exa:web_search_exa", {"query": "x"}).decision == Decision.ALLOW
    assert checker.check("mcp:test:echo", {"text": "x"}).decision == Decision.DENY


# -- streamable-HTTP transport ---------------------------------------------------------


@pytest.fixture
async def http_server():
    """An in-process streamable-HTTP MCP server on a random localhost port."""
    import uvicorn
    from mcp.server.mcpserver import MCPServer

    server = MCPServer("mock-http")

    @server.tool()
    def ping() -> str:
        """Ping the server."""
        return "pong"

    app = server.streamable_http_app()
    uvicorn_config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error")
    uvicorn_server = uvicorn.Server(uvicorn_config)
    task = asyncio.ensure_future(uvicorn_server.serve())
    deadline = asyncio.get_running_loop().time() + 15
    while not uvicorn_server.started:
        if asyncio.get_running_loop().time() > deadline:
            task.cancel()
            raise RuntimeError("uvicorn MCP server did not start")
        await asyncio.sleep(0.02)
    yield uvicorn_server
    uvicorn_server.should_exit = True
    await asyncio.wait_for(task, timeout=10)


@pytest.fixture
async def http_port(http_server):
    """Just the port of ``http_server``."""
    yield http_server.servers[0].sockets[0].getsockname()[1]


async def test_http_transport_round_trip(http_port):
    config = Config()
    config.mcp.servers["web"] = McpServerConfig(
        transport="http", url=f"http://127.0.0.1:{http_port}/mcp", timeout_s=5.0
    )
    mgr = McpManager(config)
    await mgr.connect()
    try:
        status = mgr.status()[0]
        assert status.state == "connected"
        assert status.tools == 1
        assert "mcp:web:ping" in [t.name for t in mgr.tool_wrappers()]
        result = await mgr.call("web", "ping", {})
        assert result.content == "pong"
    finally:
        await mgr.shutdown()


# -- SSE transport ----------------------------------------------------------------


@pytest.fixture
async def sse_port():
    """An in-process legacy-SSE MCP server on a random localhost port."""
    import uvicorn
    from mcp.server.mcpserver import MCPServer

    server = MCPServer("mock-sse")

    @server.tool()
    def ping() -> str:
        """Ping the server."""
        return "pong"

    app = server.sse_app()
    uvicorn_config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error")
    uvicorn_server = uvicorn.Server(uvicorn_config)
    task = asyncio.ensure_future(uvicorn_server.serve())
    deadline = asyncio.get_running_loop().time() + 15
    while not uvicorn_server.started:
        if asyncio.get_running_loop().time() > deadline:
            task.cancel()
            raise RuntimeError("uvicorn MCP server did not start")
        await asyncio.sleep(0.02)
    port = uvicorn_server.servers[0].sockets[0].getsockname()[1]
    yield port
    uvicorn_server.should_exit = True
    await asyncio.wait_for(task, timeout=10)


async def test_sse_transport_round_trip(sse_port):
    config = Config()
    config.mcp.servers["legacy"] = McpServerConfig(
        transport="sse", url=f"http://127.0.0.1:{sse_port}/sse", timeout_s=5.0
    )
    mgr = McpManager(config)
    await mgr.connect()
    try:
        status = mgr.status()[0]
        assert status.state == "connected"
        assert status.tools == 1
        assert "mcp:legacy:ping" in [t.name for t in mgr.tool_wrappers()]
        result = await mgr.call("legacy", "ping", {})
        assert result.content == "pong"
    finally:
        await mgr.shutdown()


# -- OAuth config / wiring ----------------------------------------------------------


def test_config_parses_sse_and_oauth():
    server = McpServerConfig.model_validate(
        {"transport": "sse", "url": "https://mcp.example/sse", "auth": "oauth"}
    )
    assert server.transport == "sse"
    assert server.auth == "oauth"
    # defaults unchanged
    assert McpServerConfig().transport == "stdio"
    assert McpServerConfig().auth is None


async def test_oauth_on_stdio_fails_clearly():
    mgr = McpManager(mcp_config(auth="oauth"))
    try:
        await mgr.connect()
        status = mgr.status()[0]
        assert status.state == "failed"
        assert "oauth" in (status.error or "")
    finally:
        await mgr.shutdown()


# -- /mcp command -----------------------------------------------------------------------


@pytest.fixture
async def app_with_mcp(tmp_path, monkeypatch):
    """A TUI app double with a connected mock-backed MCP manager installed."""
    from tests.test_tui_app import make_app

    app, _, out = make_app(tmp_path, monkeypatch, [])
    manager = McpManager(mcp_config())
    await manager.connect()
    manager.bind_registry(app.runtime.registry)
    app.runtime.ctx.extras[MCP_EXTRA] = manager
    yield app, manager, out
    await manager.shutdown()


async def test_mcp_status_command(app_with_mcp):
    app, _, out = app_with_mcp
    await app.handle_command("/mcp")
    assert "test: connected (4 tools)" in out.getvalue()


async def test_mcp_status_shows_failed_server(tmp_path, monkeypatch):
    from tests.test_tui_app import make_app

    app, _, out = make_app(tmp_path, monkeypatch, [])
    config = Config()
    config.mcp.servers["broken"] = McpServerConfig(
        transport="stdio", command="no-such-binary-lecode-test"
    )
    manager = McpManager(config)
    await manager.connect()
    app.runtime.ctx.extras[MCP_EXTRA] = manager
    await app.handle_command("/mcp")
    assert "broken: failed" in out.getvalue()
    await manager.shutdown()


async def test_mcp_tools_command(app_with_mcp):
    app, _, out = app_with_mcp
    await app.handle_command("/mcp tools test")
    rendered = out.getvalue()
    for tool in ("echo", "boom", "crash", "flaky"):
        assert tool in rendered
    assert "Echo the text back" in rendered


async def test_mcp_reconnect_command(app_with_mcp):
    app, _, out = app_with_mcp
    await app.handle_command("/mcp reconnect test")
    assert "test reconnected (4 tools)" in out.getvalue()
    # tools are registered into the registry after a reconnect
    assert "mcp:test:echo" in app.runtime.registry.names()


async def test_mcp_disable_enable_commands(app_with_mcp):
    app, _, out = app_with_mcp
    await app.handle_command("/mcp disable test")
    assert "test disabled (its tools left the model's context)" in out.getvalue()
    assert not any(n.startswith("mcp:test:") for n in app.runtime.registry.names())
    await app.handle_command("/mcp")
    assert "test: disabled" in out.getvalue()
    await app.handle_command("/mcp enable test")
    assert "test enabled (4 tools)" in out.getvalue()
    assert "mcp:test:echo" in app.runtime.registry.names()


async def test_mcp_enable_disable_unknown_server(app_with_mcp):
    app, _, out = app_with_mcp
    await app.handle_command("/mcp enable nope")
    await app.handle_command("/mcp disable nope")
    assert out.getvalue().count("unknown MCP server: nope") == 2


async def test_mcp_unknown_server(app_with_mcp):
    app, _, out = app_with_mcp
    await app.handle_command("/mcp reconnect nope")
    await app.handle_command("/mcp tools nope")
    assert out.getvalue().count("unknown MCP server: nope") == 2


async def test_mcp_login_logout_unknown_server(app_with_mcp):
    app, _, out = app_with_mcp
    await app.handle_command("/mcp login nope")
    await app.handle_command("/mcp logout nope")
    assert out.getvalue().count("unknown MCP server: nope") == 2


async def test_mcp_login_requires_oauth_config(app_with_mcp):
    app, _, out = app_with_mcp
    await app.handle_command("/mcp login test")  # "test" is stdio without auth = "oauth"
    assert "not configured with auth = 'oauth'" in out.getvalue()


async def test_mcp_logout_without_oauth(app_with_mcp):
    app, _, out = app_with_mcp
    await app.handle_command("/mcp logout test")
    assert "logout applies only to OAuth servers" in out.getvalue()


async def test_mcp_command_without_servers(tmp_path, monkeypatch):
    from tests.test_tui_app import make_app

    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/mcp")
    assert "no MCP servers configured" in out.getvalue()
