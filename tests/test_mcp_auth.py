"""OAuth 2.1 support tests: storage, loopback callback, and one end-to-end
flow against a local fake authorization server + MCP server.

The fake AS implements just enough of RFC 8414/7591/6749 for the SDK's
client: resource/server metadata, dynamic registration, an insta-consent
``/authorize`` redirect, and a token endpoint with refresh.
"""

from __future__ import annotations

import asyncio
import stat
import time
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urlencode

import httpx2
import pytest
import uvicorn
from mcp.client.auth import OAuthClientProvider, OAuthFlowError
from mcp.server.mcpserver import MCPServer
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, RedirectResponse

from lecode.agent.builder import build_runtime
from lecode.config.models import Config, McpServerConfig
from lecode.extras import mcp_auth as mcp_auth_mod
from lecode.extras.mcp_auth import (
    FileTokenStorage,
    LoopbackAuthCallback,
    _redirect_port,
    make_oauth_provider,
)
from lecode.extras.mcp_client import MCP_EXTRA, McpManager, ServerStatus, attach_mcp

# -- storage ------------------------------------------------------------------


@pytest.fixture
def cfg_env(tmp_path, monkeypatch):
    """Isolate credential storage from the user's real config dir."""
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "lecode-config"))
    return tmp_path


async def test_storage_round_trip_and_permissions(cfg_env):
    storage = FileTokenStorage("https://auth.example/mcp")
    assert await storage.get_tokens() is None
    assert await storage.get_client_info() is None

    token = OAuthToken(access_token="acc", refresh_token="ref", expires_in=120)
    await storage.set_tokens(token)
    await storage.set_client_info(OAuthClientInformationFull(client_id="cli-1"))

    assert await storage.get_tokens() == token
    assert (await storage.get_client_info()).client_id == "cli-1"
    assert stat.S_IMODE(storage.path.stat().st_mode) == 0o600
    assert stat.S_IMODE(storage.path.parent.stat().st_mode) == 0o700

    # clearing the client registration keeps the tokens
    await storage.clear_client_info()
    assert await storage.get_tokens() == token
    assert await storage.get_client_info() is None

    await storage.clear()
    assert not storage.path.exists()
    assert await storage.get_tokens() is None


async def test_storage_rejects_wrong_endpoint_file(cfg_env):
    storage = FileTokenStorage("https://auth.example/mcp")
    await storage.set_tokens(OAuthToken(access_token="acc"))
    # a file bound to another endpoint must never be served back
    storage.path.write_text(
        '{"server_url": "https://other.example/mcp", "tokens": {"access_token": "x"}}'
    )
    assert await storage.get_tokens() is None


async def test_storage_discards_garbage(cfg_env):
    storage = FileTokenStorage("https://auth.example/mcp")
    storage.path.parent.mkdir(parents=True, exist_ok=True)
    storage.path.write_text("not json at all")
    assert await storage.get_tokens() is None


# -- loopback callback --------------------------------------------------------


@pytest.mark.parametrize(
    "request_line",
    [b"BROKEN\r\n", b"GET http://[invalid/callback HTTP/1.1\r\n", b"x" * 70000 + b"\r\n"],
)
async def test_loopback_malformed_request_closes_connection(cfg_env, caplog, request_line):
    loopauth = await LoopbackAuthCallback.open(FileTokenStorage("https://auth.example/mcp"))
    reader, writer = await asyncio.open_connection(
        "127.0.0.1", httpx2.URL(loopauth.redirect_url).port
    )
    try:
        writer.write(request_line)
        await writer.drain()
        await asyncio.wait_for(reader.read(), timeout=1)
        assert reader.at_eof()
        assert not [record for record in caplog.records if record.levelno >= 40]
        async with httpx2.AsyncClient(trust_env=False) as client:
            response = await client.get(loopauth.redirect_url + "?code=valid&state=s1")
        assert response.status_code == 200
        assert (await loopauth.wait_for_callback()).code == "valid"
    finally:
        writer.close()
        await writer.wait_closed()
        await loopauth.aclose()


async def test_loopback_callback_round_trip(cfg_env):
    storage = FileTokenStorage("https://auth.example/mcp")
    loopauth = await LoopbackAuthCallback.open(storage)
    assert loopauth.redirect_url.startswith("http://127.0.0.1:")
    assert loopauth.redirect_url.endswith("/callback")
    try:

        def hit():
            with httpx2.Client(trust_env=False) as client:
                resp = client.get(
                    loopauth.redirect_url + "?code=c1&state=s1&iss=https%3A%2F%2Fauth.example"
                )
                assert resp.status_code == 200

        task = asyncio.create_task(asyncio.to_thread(hit))
        result = await loopauth.wait_for_callback()
        await task
        assert result.code == "c1"
        assert result.state == "s1"
        assert result.iss == "https://auth.example"
    finally:
        await loopauth.aclose()


async def test_loopback_callback_denied(cfg_env):
    storage = FileTokenStorage("https://auth.example/mcp")
    loopauth = await LoopbackAuthCallback.open(storage)
    try:

        def hit():
            with httpx2.Client(trust_env=False) as client:
                client.get(loopauth.redirect_url + "?error=access_denied&error_description=no")

        task = asyncio.create_task(asyncio.to_thread(hit))
        with pytest.raises(OAuthFlowError, match="access_denied"):
            await loopauth.wait_for_callback()
        await task
    finally:
        await loopauth.aclose()


async def test_loopback_close_cuts_pending_connection(cfg_env):
    """aclose while a browser connected but stalled: EOF now, wait cancelled."""
    loopauth = await LoopbackAuthCallback.open(FileTokenStorage("https://auth.example/mcp"))
    reader, writer = await asyncio.open_connection(
        "127.0.0.1", httpx2.URL(loopauth.redirect_url).port
    )
    try:
        read_task = asyncio.create_task(reader.read())
        await asyncio.sleep(0.1)  # let the server accept the connection
        await asyncio.wait_for(loopauth.aclose(), timeout=2)
        assert await asyncio.wait_for(read_task, timeout=2) == b""  # EOF, not the 60s idle
        with pytest.raises(asyncio.CancelledError):
            await loopauth.wait_for_callback()
    finally:
        writer.close()
        await writer.wait_closed()


def test_loopback_port_is_stable_per_endpoint(cfg_env):
    assert _redirect_port("https://auth.example/mcp") == _redirect_port("https://auth.example/mcp")


# -- provider wiring ----------------------------------------------------------


async def test_make_oauth_provider_attaches_handlers(cfg_env):
    storage = FileTokenStorage("https://auth.example/mcp")
    provider = await make_oauth_provider("https://auth.example/mcp", storage, None)
    assert isinstance(provider, OAuthClientProvider)
    # automatic connections must never open a browser
    assert provider.context.redirect_handler is None
    assert provider.context.callback_handler is None
    redirect = str(provider.context.client_metadata.redirect_uris[0])
    assert redirect == f"http://127.0.0.1:{_redirect_port('https://auth.example/mcp')}/callback"
    # no stored tokens → nothing to seed
    assert provider.context.token_expiry_time is None


async def test_oauth_conflicts_with_static_authorization_header(cfg_env):
    config = Config()
    config.mcp.enable_exa = False
    config.mcp.servers["x"] = McpServerConfig(
        transport="http",
        url="https://auth.example/mcp",
        auth="oauth",
        headers={"Authorization": "Bearer static"},
    )
    manager = McpManager(config)
    try:
        await manager.connect()  # fails fast, before any network I/O
        status = manager.status()[0]
        assert status.state == "failed"
        assert "conflicts" in status.error
    finally:
        await manager.shutdown()


# -- slash commands -----------------------------------------------------------


async def test_mcp_auth_and_logout_commands(tmp_path, monkeypatch):
    from tests.test_mcp import mock_server_config
    from tests.test_tui_app import make_app

    app, _, out = make_app(tmp_path, monkeypatch, [])
    config = Config()
    config.mcp.enable_exa = False
    config.mcp.servers["test"] = mock_server_config()
    manager = McpManager(config)
    await manager.connect()
    app.runtime.ctx.extras[MCP_EXTRA] = manager
    try:

        async def fake_authenticate(name):
            return ServerStatus(name, "connected", tools=4)

        async def fake_logout(name):
            return ServerStatus(name, "auth_required")

        monkeypatch.setattr(manager, "authenticate", fake_authenticate)
        monkeypatch.setattr(manager, "logout", fake_logout)

        await app.handle_command("/mcp auth test")
        assert "test authenticated (4 tools)" in out.getvalue()
        await app.handle_command("/mcp logout test")
        assert "test logged out" in out.getvalue()
        await app.handle_command("/mcp auth nope")
        assert "unknown MCP server: nope" in out.getvalue()
    finally:
        await manager.shutdown()


async def test_mcp_status_lists_auth_required(tmp_path, monkeypatch):
    from tests.test_tui_app import make_app

    app, _, out = make_app(tmp_path, monkeypatch, [])
    config = Config()
    config.mcp.enable_exa = False
    config.mcp.servers["oauth"] = McpServerConfig(
        transport="http", url="https://auth.example/mcp", auth="oauth"
    )
    manager = McpManager(config)

    async def fail_open(self, server):
        raise OAuthFlowError("no redirect handler provided")

    monkeypatch.setattr(McpManager, "_open", fail_open)
    await manager.connect()
    app.runtime.ctx.extras[MCP_EXTRA] = manager
    try:
        status = manager.status()[0]
        assert status.state == "auth_required"
        await app.handle_command("/mcp")
        assert "oauth: authentication required — /mcp auth oauth" in out.getvalue()
    finally:
        await manager.shutdown()


# -- end-to-end against a fake authorization server ---------------------------


class FakeAuth:
    """Issues tokens for one client; guards /mcp with bearer checks.

    Faithful on expiry: a bearer token is only accepted until its own
    ``token_ttl`` elapses (set per-issue via :attr:`token_ttl`), so an
    expired cached token really gets a 401.
    """

    def __init__(self) -> None:
        self.issued_access: list[str] = []
        self.access_expiry: dict[str, float] = {}
        self.issued_refresh: set[str] = set()
        self.deny = False
        self.reject_refresh = False
        self.token_ttl = 3600
        self._handlers: dict[tuple[str, str], Callable[[Request], Awaitable[Any]]] = {}

    @staticmethod
    def _base(request) -> str:
        return f"{request.url.scheme}://{request.url.netloc}"

    async def _protected_resource(self, request):
        base = self._base(request)
        return JSONResponse({"resource": f"{base}/mcp", "authorization_servers": [base]})

    async def _auth_metadata(self, request):
        base = self._base(request)
        return JSONResponse(
            {
                "issuer": base,
                "authorization_endpoint": f"{base}/authorize",
                "token_endpoint": f"{base}/token",
                "registration_endpoint": f"{base}/register",
                "response_types_supported": ["code"],
                "code_challenge_methods_supported": ["S256"],
            }
        )

    async def _register(self, request):
        body = await request.json()
        body["client_id"] = "lecode-test-client"
        body["client_secret"] = "lecode-test-secret"
        return JSONResponse(body, status_code=201)

    async def _authorize(self, request):
        query = request.query_params
        redirect = query["redirect_uri"]
        if self.deny:
            params = {
                "state": query.get("state", ""),
                "error": "access_denied",
                # the description is attacker/server-controlled text; the
                # regression pair below pins that control chars never survive
                "error_description": "user\x1b said\x07 no",
            }
        else:
            params = {
                "state": query.get("state", ""),
                "code": "lecode-auth-code",
                "iss": self._base(request),
            }
        separator = "&" if "?" in redirect else "?"
        return RedirectResponse(f"{redirect}{separator}{urlencode(params)}", status_code=302)

    async def _token(self, request):
        form = await request.form()
        if form.get("grant_type") == "refresh_token":
            assert form.get("refresh_token") in self.issued_refresh
            if self.reject_refresh:
                return JSONResponse({"error": "invalid_grant"}, status_code=400)
        access = f"access-{len(self.issued_access)}"
        refresh = f"refresh-{len(self.issued_access)}"
        self.issued_access.append(access)
        self.access_expiry[access] = time.time() + self.token_ttl
        self.issued_refresh.add(refresh)
        body = {
            "access_token": access,
            "token_type": "Bearer",
            "expires_in": self.token_ttl,
            "refresh_token": refresh,
        }
        if form.get("scope"):
            body["scope"] = form["scope"]
        return JSONResponse(body)

    def build_app(self):
        """The fake AS + MCP server as one ASGI app.

        A plain ASGI wrapper (not a Starlette Mount) keeps the MCP app the
        served app, so uvicorn runs its lifespan — the SDK's session manager
        starts its task group there and refuses requests without it.
        """
        mcp = MCPServer("mock-oauth")

        @mcp.tool()
        def ping() -> str:
            """Ping the server."""
            return "pong"

        inner = mcp.streamable_http_app()
        self._handlers = {
            ("GET", "/.well-known/oauth-protected-resource"): self._protected_resource,
            ("GET", "/.well-known/oauth-authorization-server"): self._auth_metadata,
            ("POST", "/register"): self._register,
            ("GET", "/authorize"): self._authorize,
            ("POST", "/token"): self._token,
        }
        authz = self

        async def app(scope, receive, send):
            if scope["type"] != "http":
                await inner(scope, receive, send)
                return
            request = Request(scope, receive)
            path = request.url.path
            if path.startswith("/mcp"):
                authorization = request.headers.get("authorization", "")
                now = time.time()
                accepted = any(
                    authorization == f"Bearer {t}" and now < authz.access_expiry[t]
                    for t in authz.issued_access
                )
                if not accepted:
                    base = f"{request.url.scheme}://{request.url.netloc}"
                    response = PlainTextResponse(
                        "authentication required",
                        status_code=401,
                        headers={
                            "WWW-Authenticate": (
                                f'Bearer resource_metadata="{base}'
                                '/.well-known/oauth-protected-resource"'
                            )
                        },
                    )
                    await response(scope, receive, send)
                    return
            handler = authz._handlers.get((scope.get("method"), path))
            if handler is not None:
                response = await handler(request)
                await response(scope, receive, send)
                return
            await inner(scope, receive, send)

        return app


@pytest.fixture
async def fake_oauth_server():
    """The fake AS + MCP server on a random localhost port."""
    authz = FakeAuth()
    config = uvicorn.Config(authz.build_app(), host="127.0.0.1", port=0, log_level="error")
    server = uvicorn.Server(config)
    task = asyncio.ensure_future(server.serve())
    deadline = asyncio.get_running_loop().time() + 15
    while not server.started:
        if asyncio.get_running_loop().time() > deadline:
            task.cancel()
            raise RuntimeError("fake OAuth server did not start")
        await asyncio.sleep(0.02)
    port = server.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}", authz
    server.should_exit = True
    await asyncio.wait_for(task, timeout=10)


def _fake_browser(url: str) -> bool:
    """A stand-in for webbrowser.open: follow the AS redirect into lecode."""
    with httpx2.Client(trust_env=False, follow_redirects=False) as client:
        resp = client.get(url)
        assert resp.status_code in (302, 303), f"expected a redirect, got {resp.status_code}"
        location = resp.headers["location"]
        assert location.startswith("http://127.0.0.1:"), "redirect must land on loopback"
        landed = client.get(location)
        assert landed.status_code == 200
    return True


async def test_oauth_end_to_end(tmp_path, monkeypatch, fake_oauth_server):
    base, authz = fake_oauth_server
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "lecode-config"))
    config = Config()
    config.mcp.enable_exa = False
    config.mcp.servers["oauth"] = McpServerConfig(
        transport="http", url=f"{base}/mcp", auth="oauth", timeout_s=5.0
    )

    browser_calls: list[str] = []
    monkeypatch.setattr(
        mcp_auth_mod.webbrowser,
        "open",
        lambda url: (browser_calls.append(url), _fake_browser(url))[1],
    )

    runtime = build_runtime(config, tmp_path, auto_approve=True)
    manager = await attach_mcp(runtime.registry, runtime.ctx)
    try:
        # automatic connect never opens a browser: it reports auth_required
        status = manager.status()[0]
        assert status.state == "auth_required"
        assert browser_calls == []

        # interactive login: browser once, then connected with tools; the AS
        # mints a short-lived access token so the live-session refresh path is
        # exercised next (the SDK only refreshes tokens it minted itself)
        authz.token_ttl = 1
        status = await manager.authenticate("oauth")
        assert status.state == "connected"
        assert status.tools == 1
        assert len(browser_calls) == 1
        assert "mcp:oauth:ping" in runtime.registry.names()

        # a real tool call goes over the authenticated transport
        message = await runtime.registry.dispatch("call-1", "mcp:oauth:ping", "{}", runtime.ctx)
        assert message["content"] == "pong"

        # once the short-lived token expires, the next call refreshes it
        # silently (no browser, no user interaction)
        authz.token_ttl = 3600
        await asyncio.sleep(1.2)
        issued_before = len(authz.issued_access)
        message = await runtime.registry.dispatch("call-2", "mcp:oauth:ping", "{}", runtime.ctx)
        assert message["content"] == "pong"
        assert len(authz.issued_access) > issued_before
        assert len(browser_calls) == 1

        # "restart": a fresh manager connects with cached credentials only
        fresh = McpManager(config)
        await fresh.connect()
        assert fresh.status()[0].state == "connected"
        assert len(browser_calls) == 1
        await fresh.shutdown()

        # logout clears persisted credentials and deregisters the tools
        storage = FileTokenStorage(f"{base}/mcp")
        status = await manager.logout("oauth")
        assert status.state == "auth_required"
        assert await storage.get_tokens() is None
        assert "mcp:oauth:ping" not in runtime.registry.names()

        # after logout, reconnect and call recovery stay non-interactive:
        # they must never open a browser, only report authentication required
        status = await manager.reconnect("oauth")
        assert status.state == "auth_required"
        assert len(browser_calls) == 1
        result = await manager.call("oauth", "ping", {})
        assert result.is_error
        assert "unavailable" in result.content
        assert len(browser_calls) == 1
    finally:
        await manager.shutdown()


async def test_oauth_restart_with_expired_token_refreshes_silently(
    tmp_path, monkeypatch, fake_oauth_server
):
    """Cached credentials past their expiry must refresh on restart, not 401.

    The SDK restores stored tokens but not their absolute expiry (only the
    ones it minted in-session carry that), so without the expiry seeding in
    ``make_oauth_provider`` an expired access token gets attached, the server
    401s it, and the automatic flow dead-ends into auth_required even though
    a perfectly good refresh token sits on disk.
    """
    base, authz = fake_oauth_server
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "lecode-config"))
    config = Config()
    config.mcp.enable_exa = False
    config.mcp.servers["oauth"] = McpServerConfig(
        transport="http", url=f"{base}/mcp", auth="oauth", timeout_s=5.0
    )
    browser_calls: list[str] = []
    monkeypatch.setattr(
        mcp_auth_mod.webbrowser,
        "open",
        lambda url: (browser_calls.append(url), _fake_browser(url))[1],
    )

    manager = McpManager(config)
    await manager.connect()
    assert manager.status()[0].state == "auth_required"
    authz.token_ttl = 1  # the access token dies one second after login
    assert (await manager.authenticate("oauth")).state == "connected"
    await manager.shutdown()

    await asyncio.sleep(1.2)  # lecode is "down" while the access token expires
    authz.token_ttl = 3600
    issued_before = len(authz.issued_access)

    # restart at the attach_mcp seam: connect + tool bridge through the registry
    runtime = build_runtime(config, tmp_path, auto_approve=True)
    fresh = await attach_mcp(runtime.registry, runtime.ctx)
    try:
        assert fresh.status()[0].state == "connected"  # silent refresh, not auth_required
        assert len(authz.issued_access) > issued_before  # a new token was minted
        assert len(browser_calls) == 1  # restart never opens a browser
        message = await runtime.registry.dispatch("call-1", "mcp:oauth:ping", "{}", runtime.ctx)
        assert message["content"] == "pong"
    finally:
        await fresh.shutdown()


async def test_oauth_restart_with_rejected_refresh_stays_noninteractive(
    tmp_path, monkeypatch, fake_oauth_server
):
    """A dead refresh token on restart: auth_required, never a browser."""
    base, authz = fake_oauth_server
    authz.reject_refresh = True
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "lecode-config"))
    config = Config()
    config.mcp.enable_exa = False
    config.mcp.servers["oauth"] = McpServerConfig(
        transport="http", url=f"{base}/mcp", auth="oauth", timeout_s=5.0
    )
    browser_calls: list[str] = []
    monkeypatch.setattr(
        mcp_auth_mod.webbrowser,
        "open",
        lambda url: (browser_calls.append(url), _fake_browser(url))[1],
    )

    manager = McpManager(config)
    await manager.connect()
    authz.token_ttl = 1
    assert (await manager.authenticate("oauth")).state == "connected"
    await manager.shutdown()

    await asyncio.sleep(1.2)

    fresh = McpManager(config)
    await fresh.connect()
    try:
        assert fresh.status()[0].state == "auth_required"
        assert len(browser_calls) == 1  # a failed refresh must not open a browser
    finally:
        await fresh.shutdown()


async def test_oauth_denied_consent(tmp_path, monkeypatch, fake_oauth_server):
    base, authz = fake_oauth_server
    authz.deny = True
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "lecode-config"))
    config = Config()
    config.mcp.enable_exa = False
    config.mcp.servers["oauth"] = McpServerConfig(
        transport="http", url=f"{base}/mcp", auth="oauth", timeout_s=5.0
    )
    monkeypatch.setattr(mcp_auth_mod.webbrowser, "open", _fake_browser)

    manager = McpManager(config)
    try:
        await manager.connect()
        status = await manager.authenticate("oauth")
        assert status.state == "auth_required"
        assert "access_denied" in status.error
        # the AS-controlled description must not carry terminal control chars
        assert "\x1b" not in status.error
        assert "\x07" not in status.error
        assert "user said no" in status.error
    finally:
        await manager.shutdown()


async def test_oauth_no_browser_available(tmp_path, monkeypatch, fake_oauth_server):
    base, _ = fake_oauth_server
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "lecode-config"))
    config = Config()
    config.mcp.enable_exa = False
    config.mcp.servers["oauth"] = McpServerConfig(
        transport="http", url=f"{base}/mcp", auth="oauth", timeout_s=5.0
    )
    monkeypatch.setattr(mcp_auth_mod.webbrowser, "open", lambda url: False)

    manager = McpManager(config)
    try:
        await manager.connect()
        status = await manager.authenticate("oauth")
        assert status.state == "auth_required"
        assert "browser" in status.error
    finally:
        await manager.shutdown()
