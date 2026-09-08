"""OAuth 2.1 support for streamable-HTTP MCP servers.

The ``mcp`` SDK's ``OAuthClientProvider`` implements the whole protocol
(protected-resource + authorization-server discovery, dynamic client
registration, PKCE, token exchange, refresh). This module supplies the two
application pieces it needs:

- :class:`FileTokenStorage`: persists tokens + client registration per
  endpoint under ``<config_dir>/mcp-auth`` (0600 files, atomic replace).
- :class:`LoopbackAuthCallback`: serves the authorization-code callback on
  ``127.0.0.1`` and opens the system browser for interactive ``/mcp auth``.

Non-interactive connections pass no redirect/callback handlers, so the SDK
reuses cached credentials (including refresh) and never blocks on a browser.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import tempfile
import time
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlsplit

from pydantic import AnyUrl

from lecode.config.loader import config_dir

if TYPE_CHECKING:  # deferred like the rest of the codebase; the SDK is heavy
    from mcp.client.auth import AuthorizationCodeResult, OAuthClientProvider
    from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

log = logging.getLogger(__name__)

#: How long one interactive login may sit waiting on the browser (s).
CALLBACK_TIMEOUT_S = 180.0

_AUTH_DIR_NAME = "mcp-auth"

#: Serializes file updates in-process so concurrent set_tokens/set_client_info
#: calls cannot clobber each other's read-modify-write.
#: ponytail: one global lock; per-endpoint locks if two servers ever contend.
_storage_lock = asyncio.Lock()

_REDIRECT_CALLBACK = "/callback"


# -- helpers ------------------------------------------------------------------


def _redirect_port(server_url: str) -> int:
    """Deterministic loopback port per endpoint (stable across sessions).

    A stable port keeps the redirect_uri registered at first login valid on
    every later login. Bind failure is self-healing in
    :meth:`LoopbackAuthCallback.open` (fresh dynamic registration).
    """
    digest = int.from_bytes(hashlib.sha256(server_url.encode()).digest()[:2])
    return 40000 + digest % 20000


def _auth_dir() -> Path:
    return config_dir() / _AUTH_DIR_NAME


def mcp_auth_file(server_url: str) -> Path:
    """The credentials file for one endpoint (public for tests/docs)."""
    digest = hashlib.sha256(server_url.encode()).hexdigest()[:32]
    return _auth_dir() / f"{digest}.json"


class FileTokenStorage:
    """Persistence for the SDK's :class:`TokenStorage` protocol.

    One JSON file per endpoint: ``{server_url, tokens, client_info}``. Files
    are 0600 from creation, written atomically (temp file + replace); the
    directory is 0700. Contents are plaintext JSON.
    """

    def __init__(self, server_url: str) -> None:
        self.server_url = server_url
        self.path = mcp_auth_file(server_url)

    # -- protocol -----------------------------------------------------------

    async def get_tokens(self) -> OAuthToken | None:
        from mcp.shared.auth import OAuthToken

        data = await self._read()
        if data is None or data.get("tokens") is None:
            return None
        try:
            return OAuthToken.model_validate_json(json.dumps(data["tokens"]))
        except (ValueError, TypeError) as e:
            log.debug("mcp-auth: discarding invalid tokens in %s: %s", self.path, e)
            return None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        # Stamp when the tokens were acquired: the SDK does not persist (or
        # restore) the absolute expiry itself, so this is the only way to know
        # later that a stored access token has expired (see token_expiry()).
        await self._update(tokens=tokens.model_dump(mode="json"), tokens_acquired_at=time.time())

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        from mcp.shared.auth import OAuthClientInformationFull

        data = await self._read()
        if data is None or data.get("client_info") is None:
            return None
        try:
            return OAuthClientInformationFull.model_validate_json(json.dumps(data["client_info"]))
        except (ValueError, TypeError) as e:
            log.debug("mcp-auth: discarding invalid client_info in %s: %s", self.path, e)
            return None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        await self._update(client_info=client_info.model_dump(mode="json"))

    # -- management ----------------------------------------------------------

    async def clear(self) -> None:
        """Drop all persisted credentials for this endpoint (logout)."""
        async with _storage_lock:

            def do_unlink() -> bool:
                try:
                    self.path.unlink()
                    return True
                except FileNotFoundError:
                    return False

            removed = await asyncio.to_thread(do_unlink)
            if removed:
                log.debug("mcp-auth: cleared %s", self.path)

    async def clear_client_info(self) -> None:
        """Drop only the client registration (forces a fresh one next login)."""
        await self._update(client_info=None)

    async def token_expiry(self) -> float | None:
        """When the stored access token expires (Unix time); None if unknown.

        None means "no stored tokens", "no expiry info", or a file from before
        acquisition stamping — all map to the SDK's own unknown-expiry
        behavior (attach and let the server judge).
        """
        data = await self._read()
        if data is None:
            return None
        acquired = data.get("tokens_acquired_at")
        expires_in = (data.get("tokens") or {}).get("expires_in")
        if isinstance(acquired, bool) or not isinstance(acquired, (int, float)):
            return None
        try:
            return acquired + int(expires_in)
        except (TypeError, ValueError):
            return None

    # -- internals -----------------------------------------------------------

    async def _read(self) -> dict[str, Any] | None:
        def do_read() -> dict[str, Any] | None:
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                return None
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
                log.debug("mcp-auth: unreadable credentials file %s: %s", self.path, e)
                return None
            if not isinstance(data, dict):
                return None
            # fail-safe rebinding: never serve this endpoint another URL's
            # credentials (only reachable on a sha256 collision).
            if data.get("server_url") != self.server_url:
                log.warning("mcp-auth: %s is bound to a different endpoint — ignoring", self.path)
                return None
            return data

        return await asyncio.to_thread(do_read)

    async def _update(self, **fields: Any) -> None:
        async with _storage_lock:
            data = await self._read()
            merged = data or {}
            merged["server_url"] = self.server_url
            merged.update(fields)

            def do_write() -> None:
                _auth_dir().mkdir(mode=0o700, parents=True, exist_ok=True)
                with contextlib.suppress(OSError):  # pragma: no cover — best effort
                    _auth_dir().chmod(0o700)
                # mkstemp → 0600 by default; os.replace keeps that mode.
                fd, tmp_path = tempfile.mkstemp(dir=_auth_dir(), prefix=".tmp-")
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as fh:
                        json.dump(merged, fh)
                        fh.flush()
                        os.fsync(fh.fileno())
                    os.replace(tmp_path, self.path)
                finally:
                    if os.path.exists(tmp_path):  # pragma: no branch
                        with contextlib.suppress(OSError):  # pragma: no cover
                            os.unlink(tmp_path)

            await asyncio.to_thread(do_write)


# -- interactive loopback callback --------------------------------------------


class LoopbackAuthCallback:
    """Hosts the authorization-code callback on 127.0.0.1 for one login."""

    def __init__(
        self,
        server_url: str,
        server: asyncio.Server,
        redirect_url: str,
        result: asyncio.Future[dict[str, list[str]]],
        accepted: set[asyncio.StreamWriter],
    ) -> None:
        self.server_url = server_url
        self._server = server
        self.redirect_url = redirect_url
        self._result = result
        self._accepted = accepted

    @classmethod
    async def open(cls, storage: FileTokenStorage) -> LoopbackAuthCallback:
        """Bind the callback listener, reusing the port registered previously."""
        port = _redirect_port(storage.server_url)
        client_info = await storage.get_client_info()
        registered = client_info.redirect_uris if client_info is not None else None
        if registered:
            registered_port = registered[0].port
            if registered_port is not None:
                port = registered_port
        loop = asyncio.get_running_loop()
        result: asyncio.Future[dict[str, list[str]]] = loop.create_future()
        accepted: set[asyncio.StreamWriter] = set()

        async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            accepted.add(writer)
            try:
                try:
                    line = await asyncio.wait_for(reader.readline(), timeout=60)
                except (TimeoutError, ValueError, EOFError):  # slow, overlong, or broken client
                    line = b""
                # "GET /callback?... HTTP/1.1" — one request per login; junk 404s
                parts = line.decode(errors="replace").split(" ")
                target = parts[1] if len(parts) > 1 else ""
                try:
                    ok = bool(target) and urlsplit(target).path == _REDIRECT_CALLBACK
                except ValueError:  # e.g. a malformed IPv6 literal in the target
                    ok = False
                body = (
                    b"<html><body><h3>lecode</h3><p>"
                    b"Authorization received. You can close this window and return to lecode."
                    b"</p></body></html>"
                )
                headers = (
                    f"HTTP/1.1 {200 if ok else 404} {'OK' if ok else 'Not Found'}\r\n"
                    "Content-Type: text/html; charset=utf-8\r\n"
                    f"Content-Length: {len(body)}\r\n"
                    "Connection: close\r\n"
                    "\r\n"
                ).encode()
                writer.write(headers + body)
                try:
                    await writer.drain()
                finally:
                    writer.close()
                if ok and not result.done():
                    result.set_result(parse_qs(urlsplit(target).query))
            except Exception:  # pragma: no cover — never raise into asyncio's callback task
                log.debug("mcp-auth: callback connection failed", exc_info=True)
                with contextlib.suppress(Exception):
                    writer.close()
            finally:
                accepted.discard(writer)

        try:
            server = await asyncio.start_server(handle, "127.0.0.1", port)
        except OSError:
            # The stable port is taken (rare) — the old registration is then
            # useless, so drop it and register afresh on a random free port.
            log.debug("mcp-auth: callback port %d busy; re-registering on a free port", port)
            await storage.clear_client_info()
            server = await asyncio.start_server(handle, "127.0.0.1", 0)
        actual_port = server.sockets[0].getsockname()[1]
        redirect_url = f"http://127.0.0.1:{actual_port}{_REDIRECT_CALLBACK}"
        return cls(storage.server_url, server, redirect_url, result, accepted)

    async def open_browser(self, authorization_url: str) -> None:
        """The SDK's redirect_handler: hand the URL to the system browser."""
        from mcp.client.auth import OAuthFlowError

        opened = await asyncio.to_thread(webbrowser.open, authorization_url)
        if not opened:
            raise OAuthFlowError(
                "could not open a browser on this machine — run /mcp auth where one exists"
            )

    async def wait_for_callback(self) -> AuthorizationCodeResult:
        """The SDK's callback_handler: yield the redirect parameters."""
        from mcp.client.auth import AuthorizationCodeResult, OAuthFlowError

        try:
            params = await asyncio.wait_for(self._result, timeout=CALLBACK_TIMEOUT_S)
        except TimeoutError:
            raise OAuthFlowError(
                "timed out waiting for the authorization redirect "
                f"(limit {int(CALLBACK_TIMEOUT_S)}s)"
            ) from None
        error = params.get("error") or [None]
        if error[0]:
            description = params.get("error_description") or ["(no description)"]
            raise OAuthFlowError(f"authorization denied: {error[0]} — {description[0]}")
        code = params.get("code") or [None]
        if not code[0]:
            raise OAuthFlowError("the authorization redirect carried no code")
        return AuthorizationCodeResult(
            code=code[0],
            state=(params.get("state") or [None])[0],
            iss=(params.get("iss") or [None])[0],
        )

    async def aclose(self) -> None:
        """Stop serving callbacks; no-op when the flow already completed."""
        self._server.close()
        # Close any still-open accepted connection so its handler task and the
        # peer both see the shutdown instead of idling until the 60s read timeout.
        for writer in list(self._accepted):
            with contextlib.suppress(Exception):
                writer.close()
        await self._server.wait_closed()
        if not self._result.done():
            self._result.cancel()


async def make_oauth_provider(
    server_url: str,
    storage: FileTokenStorage,
    loopback: LoopbackAuthCallback | None,
) -> OAuthClientProvider:
    """Build the SDK auth handler for one endpoint.

    ``loopback`` is ``None`` for automatic connections (cached credentials
    and refresh only — no browser); interactive ``/mcp auth`` passes one.
    """
    from mcp.client.auth import OAuthClientProvider
    from mcp.shared.auth import OAuthClientMetadata

    redirect_url = (
        loopback.redirect_url
        if loopback is not None
        else (f"http://127.0.0.1:{_redirect_port(server_url)}{_REDIRECT_CALLBACK}")
    )
    metadata = OAuthClientMetadata(client_name="lecode", redirect_uris=[AnyUrl(redirect_url)])
    provider = OAuthClientProvider(
        server_url=server_url,
        client_metadata=metadata,
        storage=storage,
        redirect_handler=loopback.open_browser if loopback is not None else None,
        callback_handler=loopback.wait_for_callback if loopback is not None else None,
    )
    # The SDK restores stored tokens but not their absolute expiry, so an
    # expired access token would be attached, 401'd, and the automatic flow
    # would dead-end into "authentication required" even with a good refresh
    # token on disk. Seed the expiry it failed to restore and its own
    # proactive refresh path runs — silently, browser-free.
    expiry = await storage.token_expiry()
    if expiry is not None and expiry <= time.time():
        provider.context.token_expiry_time = expiry
    return provider
