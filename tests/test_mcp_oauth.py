"""MCP OAuth over the SSE transport: the full dance, end to end.

These tests pin the SSE half of the unified OAuth stack
(``lecode.extras.mcp_auth`` + ``McpManager``); the streamable-HTTP half plus
the storage/loopback units live in tests/test_mcp_auth.py.

The full authorization-code flow runs against an in-process uvicorn server
that plays both the OAuth authorization server (discovery, registration,
authorize redirect, token exchange) and the protected MCP SSE endpoint. The
"browser" is a patched ``webbrowser.open`` that follows the authorization
URL with a real HTTP client, so the SDK's own stack and lecode's loopback
callback are exercised end to end.

Manual verification against a real server: add
``[mcp.servers.x] transport = "sse" url = "…" auth = "oauth"`` for an
OAuth-protected server, run ``lecode`` interactively, ``/mcp auth x`` —
the browser opens, credentials land in ``<config_dir>/mcp-auth/`` (0600),
and subsequent headless runs (``lecode -p …``) reuse them.
"""

from __future__ import annotations

import asyncio
import json
import stat
from pathlib import Path
from urllib.parse import parse_qsl

import httpx2
import pytest

from lecode.config.models import Config, McpServerConfig
from lecode.extras import mcp_auth as mcp_auth_mod
from lecode.extras.mcp_auth import mcp_auth_file
from lecode.extras.mcp_client import McpManager


class _FakeAuthServer:
    """ASGI app: OAuth AS (discovery/register/authorize/token) + protected MCP."""

    def __init__(self, mcp_app):
        self._mcp_app = mcp_app
        self.base = ""  # set once uvicorn is up
        self.hits = {"register": 0, "authorize": 0, "token": 0, "unauthorized": 0}

    async def __call__(self, scope, receive, send):
        assert scope["type"] == "http"
        path = scope["path"]
        if path.startswith("/.well-known/oauth-protected-resource"):
            return await self._json(
                send,
                {"resource": f"{self.base}/sse", "authorization_servers": [self.base]},
            )
        if path == "/.well-known/oauth-authorization-server":
            return await self._json(
                send,
                {
                    "issuer": self.base,
                    "authorization_endpoint": f"{self.base}/authorize",
                    "token_endpoint": f"{self.base}/token",
                    "registration_endpoint": f"{self.base}/register",
                },
            )
        if path == "/register" and scope["method"] == "POST":
            self.hits["register"] += 1
            data = json.loads(await self._body(receive))
            return await self._json(
                send,
                {**data, "client_id": "lecode-test-client", "token_endpoint_auth_method": "none"},
                status=201,
            )
        if path == "/authorize":
            self.hits["authorize"] += 1
            qs = dict(parse_qsl(scope["query_string"].decode()))
            location = f"{qs['redirect_uri']}?code=test-code&state={qs['state']}"
            await send(
                {
                    "type": "http.response.start",
                    "status": 302,
                    "headers": [(b"location", location.encode())],
                }
            )
            return await send({"type": "http.response.body", "body": b""})
        if path == "/token" and scope["method"] == "POST":
            self.hits["token"] += 1
            params = dict(parse_qsl((await self._body(receive)).decode()))
            assert params["grant_type"] == "authorization_code"
            assert params["code"] == "test-code"
            return await self._json(
                send,
                {
                    "access_token": "good-token",
                    "token_type": "Bearer",
                    "expires_in": 3600,
                    "refresh_token": "refresh-1",
                },
            )
        if path == "/sse" or path.startswith("/messages"):
            headers = dict(scope["headers"])
            if headers.get(b"authorization", b"").decode() != "Bearer good-token":
                self.hits["unauthorized"] += 1
                prm_url = f"{self.base}/.well-known/oauth-protected-resource"
                www_auth = f'Bearer resource_metadata="{prm_url}"'
                return await self._json(
                    send,
                    {"error": "unauthorized"},
                    status=401,
                    headers=[(b"www-authenticate", www_auth.encode())],
                )
            return await self._mcp_app(scope, receive, send)
        return await self._json(send, {"error": "not found"}, status=404)

    @staticmethod
    async def _body(receive) -> bytes:
        body = b""
        while True:
            message = await receive()
            if message["type"] != "http.request":
                break
            body += message.get("body", b"")
            if not message.get("more_body"):
                break
        return body

    @staticmethod
    async def _json(send, payload, *, status=200, headers=None):
        body = json.dumps(payload).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [(b"content-type", b"application/json"), *(headers or [])],
            }
        )
        await send({"type": "http.response.body", "body": body})


@pytest.fixture
async def oauth_server(tmp_path, monkeypatch):
    """The fake AS + protected SSE MCP server; config dir redirected to tmp."""
    import uvicorn
    from mcp.server.mcpserver import MCPServer

    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    server = MCPServer("mock-oauth-sse")

    @server.tool()
    def ping() -> str:
        """Ping the server."""
        return "pong"

    auth_app = _FakeAuthServer(server.sse_app())
    uvicorn_config = uvicorn.Config(auth_app, host="127.0.0.1", port=0, log_level="error")
    uvicorn_server = uvicorn.Server(uvicorn_config)
    task = asyncio.ensure_future(uvicorn_server.serve())
    deadline = asyncio.get_running_loop().time() + 15
    while not uvicorn_server.started:
        if asyncio.get_running_loop().time() > deadline:
            task.cancel()
            raise RuntimeError("uvicorn OAuth server did not start")
        await asyncio.sleep(0.02)
    port = uvicorn_server.servers[0].sockets[0].getsockname()[1]
    auth_app.base = f"http://127.0.0.1:{port}"
    yield auth_app
    uvicorn_server.should_exit = True
    await asyncio.wait_for(task, timeout=10)


@pytest.fixture
def fake_browser(monkeypatch):
    """``webbrowser.open`` stand-in: follow the authorize URL into lecode."""
    opened: list[str] = []

    def _open(url: str) -> bool:
        opened.append(url)
        with httpx2.Client(trust_env=False, follow_redirects=False) as client:
            response = client.get(url)
            assert response.status_code in (302, 303), f"expected a redirect, got {response}"
            location = response.headers["location"]
            assert location.startswith("http://127.0.0.1:"), "redirect must land on loopback"
            assert client.get(location).status_code == 200
        return True

    monkeypatch.setattr(mcp_auth_mod.webbrowser, "open", _open)
    return opened


def _oauth_config(base: str) -> Config:
    config = Config()
    config.mcp.enable_exa = False
    config.mcp.servers["auth"] = McpServerConfig(
        transport="sse", url=f"{base}/sse", auth="oauth", timeout_s=5.0
    )
    return config


def _token_file(base: str) -> Path:
    return mcp_auth_file(f"{base}/sse")


async def test_sse_oauth_full_dance_and_token_reuse(oauth_server, fake_browser):
    base = oauth_server.base
    config = _oauth_config(base)
    notices: list[str] = []
    mgr = McpManager(config, notify=notices.append)
    await mgr.connect()
    try:
        # automatic connect never opens a browser: it reports auth_required
        assert mgr.status()[0].state == "auth_required"
        assert fake_browser == []

        # interactive login: browser dance, then connected with tools
        status = await mgr.authenticate("auth")
        assert status.state == "connected", status.error
        assert status.tools == 1
        result = await mgr.call("auth", "ping", {})
        assert result.content == "pong"
    finally:
        await mgr.shutdown()

    # the browser was sent to /authorize exactly once; the URL was also
    # surfaced through the manager-level notify callback
    assert len(fake_browser) == 1
    assert "/authorize" in fake_browser[0]
    assert any("/authorize" in notice for notice in notices)
    assert oauth_server.hits["register"] == 1
    assert oauth_server.hits["authorize"] == 1
    assert oauth_server.hits["token"] == 1

    # credentials + client registration persisted, 0600
    token_file = _token_file(base)
    assert token_file.is_file()
    assert stat.S_IMODE(token_file.stat().st_mode) == 0o600
    data = json.loads(token_file.read_text())
    assert data["tokens"]["access_token"] == "good-token"
    assert data["client_info"]["client_id"] == "lecode-test-client"

    # a fresh manager reuses the stored credentials: no second dance
    mgr2 = McpManager(config)
    await mgr2.connect()
    try:
        assert mgr2.status()[0].state == "connected"
        result = await mgr2.call("auth", "ping", {})
        assert result.content == "pong"
    finally:
        await mgr2.shutdown()
    assert oauth_server.hits["authorize"] == 1
    assert oauth_server.hits["token"] == 1
    assert len(fake_browser) == 1


async def test_login_reruns_the_flow(oauth_server, fake_browser):
    """``login`` drops cached credentials, so the browser flow runs again."""
    mgr = McpManager(_oauth_config(oauth_server.base))
    await mgr.connect()
    try:
        status = await mgr.authenticate("auth")
        assert status.state == "connected", status.error
        assert oauth_server.hits["authorize"] == 1
        status = await mgr.login("auth")
        assert status is not None
        assert status.state == "connected", status.error
        assert oauth_server.hits["authorize"] == 2
        assert len(fake_browser) == 2
    finally:
        await mgr.shutdown()


async def test_logout_deletes_tokens(oauth_server, fake_browser):
    mgr = McpManager(_oauth_config(oauth_server.base))
    await mgr.connect()
    try:
        status = await mgr.authenticate("auth")
        assert status.state == "connected", status.error
        assert _token_file(oauth_server.base).is_file()

        status = await mgr.logout("auth")
        assert status is not None
        assert status.state == "auth_required"
        assert not _token_file(oauth_server.base).exists()

        # reconnect stays non-interactive: auth_required, never a browser
        status = await mgr.reconnect("auth")
        assert status is not None
        assert status.state == "auth_required"
        assert len(fake_browser) == 1
    finally:
        await mgr.shutdown()


async def test_sse_oauth_headless_without_tokens_fails(oauth_server, fake_browser):
    """No stored credentials: the server reports an actionable auth_required."""
    mgr = McpManager(_oauth_config(oauth_server.base))
    try:
        await mgr.connect()
        status = mgr.status()[0]
        assert status.state == "auth_required"
        assert status.auth_hint == "auth: authentication required — /mcp auth auth"
        assert fake_browser == []
    finally:
        await mgr.shutdown()
