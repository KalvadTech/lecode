"""Tests for MCP OAuth: token storage, the callback listener, the live dance.

The full authorization-code flow runs against an in-process uvicorn server
that plays both the OAuth authorization server (discovery, registration,
authorize redirect, token exchange) and the protected MCP SSE endpoint. The
"browser" is a patched ``webbrowser.open`` that follows the authorization
URL with a real httpx client, so the SDK's own httpx2 stack and the
localhost callback listener are exercised end to end.

Manual verification against a real server: add
``[mcp.servers.x] transport = "http" url = "…" oauth = true`` for an
OAuth-protected server, run ``lecode`` interactively, ``/mcp login x`` —
the browser opens, tokens land in ``<config_dir>/mcp_auth/x.json`` (0600),
and subsequent headless runs (``lecode -p …``) reuse them.
"""

from __future__ import annotations

import asyncio
import json
import stat
import webbrowser
from pathlib import Path
from urllib.parse import parse_qsl, urlparse

import httpx
import pytest

from lecode.config.models import Config, McpServerConfig
from lecode.extras.mcp_client import McpManager
from lecode.extras.mcp_oauth import (
    FileTokenStorage,
    OAuthCallbackListener,
    browser_redirect_handler,
    build_oauth_provider,
    delete_tokens,
)

# -- FileTokenStorage ----------------------------------------------------------


async def test_storage_round_trip(tmp_path):
    from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

    storage = FileTokenStorage("my server", base_dir=tmp_path)  # space is sanitized
    assert await storage.get_tokens() is None
    assert await storage.get_client_info() is None

    await storage.set_tokens(OAuthToken(access_token="tok", expires_in=3600, refresh_token="r1"))
    await storage.set_client_info(
        OAuthClientInformationFull(client_id="cid", redirect_uris=["http://127.0.0.1:1/callback"])
    )

    again = FileTokenStorage("my server", base_dir=tmp_path)
    tokens = await again.get_tokens()
    assert tokens is not None
    assert tokens.access_token == "tok"
    assert tokens.refresh_token == "r1"
    info = await again.get_client_info()
    assert info is not None
    assert info.client_id == "cid"

    path = tmp_path / "my_server.json"
    assert path.is_file()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


async def test_storage_missing_and_corrupt_files(tmp_path):
    storage = FileTokenStorage("srv", base_dir=tmp_path)
    assert await storage.get_tokens() is None

    path = tmp_path / "srv.json"
    path.write_text("not json{")
    assert await storage.get_tokens() is None
    assert await storage.get_client_info() is None

    path.write_text(json.dumps({"tokens": {"unexpected": "shape"}}))
    assert await storage.get_tokens() is None

    path.write_text(json.dumps(["a", "list"]))
    assert await storage.get_tokens() is None


async def test_storage_set_tokens_preserves_client_info(tmp_path):
    from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

    storage = FileTokenStorage("srv", base_dir=tmp_path)
    await storage.set_client_info(OAuthClientInformationFull(client_id="keep-me"))
    await storage.set_tokens(OAuthToken(access_token="tok"))
    info = await storage.get_client_info()
    assert info is not None and info.client_id == "keep-me"


def test_delete_tokens(tmp_path, monkeypatch):
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path))
    assert delete_tokens("srv") is False
    directory = tmp_path / "mcp_auth"
    directory.mkdir()
    (directory / "srv.json").write_text("{}")
    assert delete_tokens("srv") is True
    assert not (directory / "srv.json").exists()


# -- callback listener ------------------------------------------------------------


def _listener_port(listener: OAuthCallbackListener) -> int:
    return int(urlparse(listener.redirect_uri).port)


async def test_listener_captures_code_and_state():
    listener = OAuthCallbackListener(timeout=5)
    await listener.start()
    try:
        port = _listener_port(listener)
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(b"GET /callback?code=abc123&state=xyz HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
        await writer.drain()
        result = await listener.wait()
        assert result.code == "abc123"
        assert result.state == "xyz"
        body = await reader.read()
        assert b"authorization received" in body
        writer.close()
    finally:
        await listener.close()


async def test_listener_times_out():
    listener = OAuthCallbackListener(timeout=0.05)
    await listener.start()
    try:
        with pytest.raises(TimeoutError):
            await listener.wait()
    finally:
        await listener.close()


async def test_listener_wait_before_start_raises():
    listener = OAuthCallbackListener(timeout=5)
    with pytest.raises(RuntimeError):
        _ = listener.redirect_uri


# -- redirect handler / provider wiring --------------------------------------------


async def test_redirect_handler_opens_browser_and_notifies(monkeypatch):
    opened = []
    monkeypatch.setattr(webbrowser, "open", lambda url: opened.append(url) or True)
    notices = []
    await browser_redirect_handler("http://as.example/authorize?x=1", notify=notices.append)
    assert opened == ["http://as.example/authorize?x=1"]
    assert len(notices) == 1
    assert "http://as.example/authorize?x=1" in notices[0]
    assert "opened the authorization page" in notices[0]


async def test_redirect_handler_fail_open_without_browser(monkeypatch):
    def _boom(url):
        raise OSError("no display")

    monkeypatch.setattr(webbrowser, "open", _boom)
    notices = []
    await browser_redirect_handler("http://as.example/auth", notify=notices.append)
    assert "open this URL in a browser" in notices[0]
    assert "http://as.example/auth" in notices[0]

    # no notify callback at all: still never raises
    await browser_redirect_handler("http://as.example/auth")


async def test_build_oauth_provider_wiring(tmp_path, monkeypatch):
    from mcp.client.auth import OAuthClientProvider

    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path))
    provider = await build_oauth_provider("srv", "http://mcp.example/sse", notify=None)
    try:
        assert isinstance(provider, OAuthClientProvider)
        assert isinstance(provider.context.storage, FileTokenStorage)
        metadata = provider.context.client_metadata
        assert metadata.client_name == "lecode"
        # the registered redirect URI matches the live listener
        assert str(metadata.redirect_uris[0]).rstrip("/") == provider.callback_listener.redirect_uri
        assert provider.context.redirect_handler is not None
        assert provider.context.callback_handler is not None
    finally:
        await provider.callback_listener.close()


# -- the full dance against an in-process authorization server ---------------------


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


class _FakeBrowser:
    """``webbrowser.open`` stand-in: follows the authorize URL with httpx."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self.opened: list[str] = []
        self.drive = True

    def __call__(self, url: str) -> bool:
        self.opened.append(url)
        if not self.drive:
            return False

        async def _follow() -> None:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                response = await client.get(url)
                assert response.status_code == 200

        asyncio.run_coroutine_threadsafe(_follow(), self._loop)
        return True


@pytest.fixture
async def oauth_server(tmp_path, monkeypatch):
    """The fake AS + protected SSE MCP server; config dir redirected to tmp."""
    import uvicorn
    from mcp.server.mcpserver import MCPServer

    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    server = MCPServer("mock-oauth")

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
async def fake_browser(monkeypatch):
    browser = _FakeBrowser(asyncio.get_running_loop())
    monkeypatch.setattr(webbrowser, "open", browser)
    return browser


def _oauth_config(base: str) -> Config:
    config = Config()
    config.mcp.enable_exa = False
    config.mcp.servers["auth"] = McpServerConfig(
        transport="sse", url=f"{base}/sse", oauth=True, timeout_s=5.0
    )
    return config


def _token_file(tmp_path: Path) -> Path:
    return tmp_path / "cfg" / "mcp_auth" / "auth.json"


async def test_oauth_full_dance_and_token_reuse(oauth_server, fake_browser, tmp_path):
    config = _oauth_config(oauth_server.base)
    notices: list[str] = []
    mgr = McpManager(config, notify=lambda text: notices.append(text))
    await mgr.connect()
    try:
        status = mgr.status()[0]
        assert status.state == "connected", status.error
        assert status.tools == 1
        result = await mgr.call("auth", "ping", {})
        assert result.content == "pong"
    finally:
        await mgr.shutdown()

    # the browser was sent to /authorize; the URL was also surfaced via notify
    assert len(fake_browser.opened) == 1
    assert "/authorize" in fake_browser.opened[0]
    assert any("/authorize" in notice for notice in notices)
    assert oauth_server.hits == {"register": 1, "authorize": 1, "token": 1, "unauthorized": 1}

    # tokens + client registration persisted, 0600
    token_file = _token_file(tmp_path)
    assert token_file.is_file()
    assert stat.S_IMODE(token_file.stat().st_mode) == 0o600
    data = json.loads(token_file.read_text())
    assert data["tokens"]["access_token"] == "good-token"
    assert data["client_info"]["client_id"] == "lecode-test-client"

    # a fresh manager reuses the stored token: no second dance
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
    assert len(fake_browser.opened) == 1


async def test_login_reruns_the_flow(oauth_server, fake_browser, tmp_path):
    mgr = McpManager(_oauth_config(oauth_server.base))
    await mgr.connect()
    try:
        assert mgr.status()[0].state == "connected"
        assert oauth_server.hits["authorize"] == 1
        status = await mgr.login("auth")
        assert status is not None
        assert status.state == "connected", status.error
        assert oauth_server.hits["authorize"] == 2
    finally:
        await mgr.shutdown()


async def test_logout_deletes_tokens(oauth_server, fake_browser, tmp_path, monkeypatch):
    mgr = McpManager(_oauth_config(oauth_server.base))
    await mgr.connect()
    try:
        assert mgr.status()[0].state == "connected"
        assert _token_file(tmp_path).is_file()

        # with no browser to complete the re-authorization, the reconnect fails
        # fast; the deleted token file must stay deleted
        fake_browser.drive = False
        monkeypatch.setattr("lecode.extras.mcp_client.CONNECT_TIMEOUT_S", 2.0)
        status = await mgr.logout("auth")
        assert status is not None
        assert status.state == "failed"
        assert not _token_file(tmp_path).exists()
    finally:
        await mgr.shutdown()


async def test_oauth_headless_without_tokens_fails(oauth_server, fake_browser, monkeypatch):
    """No browser, no stored tokens: the server fails with an actionable error."""
    fake_browser.drive = False
    monkeypatch.setattr("lecode.extras.mcp_client.CONNECT_TIMEOUT_S", 2.0)
    mgr = McpManager(_oauth_config(oauth_server.base))
    try:
        await mgr.connect()
        status = mgr.status()[0]
        assert status.state == "failed"
        assert "authorize interactively" in (status.error or "")
        assert "/mcp login auth" in (status.error or "")
    finally:
        await mgr.shutdown()
