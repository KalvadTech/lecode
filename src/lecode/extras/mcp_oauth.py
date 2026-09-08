"""OAuth support for remote (http/sse) MCP servers.

Wraps the SDK's ``OAuthClientProvider`` (authorization-code flow + PKCE) with
lecode's concrete pieces:

- :class:`FileTokenStorage` persists tokens and the dynamic client
  registration as JSON under ``<config_dir>/mcp_auth/<server>.json`` (0600).
- :class:`OAuthCallbackListener` is an ephemeral localhost HTTP listener that
  captures the ``?code=…&state=…`` redirect and answers with a "you can close
  this tab" page.
- :func:`browser_redirect_handler` opens the authorization URL in the system
  browser and surfaces it through a caller-provided ``notify`` callback so the
  URL is reachable when no browser opens (headless host, SSH session).

Everything is fail-open: corrupt token files read as empty, and a failed
authorization simply fails that one server (per-server isolation in
``McpManager``). With tokens already stored, headless runs need no
interaction — the SDK refreshes them itself.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import webbrowser
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lecode.config.loader import config_dir

if TYPE_CHECKING:
    from mcp.shared.auth import (
        AuthorizationCodeResult,
        OAuthClientInformationFull,
        OAuthToken,
    )

log = logging.getLogger(__name__)

#: How long the callback listener waits for the browser redirect.
CALLBACK_TIMEOUT_S = 300.0

#: Async ``(text) -> None`` used to surface authorization URLs and progress.
Notify = Callable[[str], Any]


def token_dir() -> Path:
    """Directory holding per-server OAuth token files."""
    return config_dir() / "mcp_auth"


def _safe_name(server_name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in server_name)


def delete_tokens(server_name: str) -> bool:
    """Delete one server's stored tokens/registration; True when a file existed."""
    path = token_dir() / f"{_safe_name(server_name)}.json"
    existed = path.exists()
    with contextlib.suppress(OSError):
        path.unlink()
    return existed


class FileTokenStorage:
    """The SDK's ``TokenStorage`` protocol over one JSON file per server."""

    def __init__(self, server_name: str, base_dir: Path | None = None) -> None:
        directory = base_dir if base_dir is not None else token_dir()
        self._path = directory / f"{_safe_name(server_name)}.json"

    def _read(self) -> dict[str, Any]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _write(self, data: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_name(self._path.name + ".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, indent=2))
        os.replace(tmp, self._path)
        os.chmod(self._path, 0o600)

    async def get_tokens(self) -> OAuthToken | None:
        from mcp.shared.auth import OAuthToken

        raw = self._read().get("tokens")
        if raw is None:
            return None
        try:
            return OAuthToken.model_validate(raw)
        except Exception:
            log.debug("mcp oauth: ignoring invalid stored tokens in %s", self._path)
            return None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        data = self._read()
        data["tokens"] = tokens.model_dump(mode="json")
        self._write(data)

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        from mcp.shared.auth import OAuthClientInformationFull

        raw = self._read().get("client_info")
        if raw is None:
            return None
        try:
            return OAuthClientInformationFull.model_validate(raw)
        except Exception:
            log.debug("mcp oauth: ignoring invalid stored client info in %s", self._path)
            return None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        data = self._read()
        data["client_info"] = client_info.model_dump(mode="json")
        self._write(data)


class OAuthCallbackListener:
    """One-shot localhost HTTP listener for the OAuth redirect.

    Bound at construction so its address can be registered as the client's
    ``redirect_uri``; :meth:`wait` serves exactly one request and returns the
    captured ``code``/``state``.
    """

    def __init__(self, timeout: float = CALLBACK_TIMEOUT_S) -> None:
        self._timeout = timeout
        self._server: asyncio.base_events.Server | None = None
        self._done = asyncio.Event()
        self._params: dict[str, str] = {}

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)

    @property
    def redirect_uri(self) -> str:
        if self._server is None:
            raise RuntimeError("OAuthCallbackListener is not started")
        port = self._server.sockets[0].getsockname()[1]
        return f"http://127.0.0.1:{port}/callback"

    async def wait(self) -> AuthorizationCodeResult:
        """SDK ``callback_handler``: block until the redirect arrives."""
        from mcp.shared.auth import AuthorizationCodeResult

        try:
            async with asyncio.timeout(self._timeout):
                await self._done.wait()
        except TimeoutError:
            raise TimeoutError(
                "timed out waiting for the OAuth redirect — finish the browser "
                "authorization and retry"
            ) from None
        return AuthorizationCodeResult(
            code=self._params.get("code", ""),
            state=self._params.get("state"),
            iss=self._params.get("iss"),
        )

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        from urllib.parse import parse_qsl, urlparse

        ok = False
        try:
            request = await asyncio.wait_for(reader.read(16384), timeout=30)
            target = request.split(b"\r\n", 1)[0].split()
            if len(target) >= 2:
                query = dict(parse_qsl(urlparse(target[1].decode("utf-8", "replace")).query))
                self._params = query
                ok = "code" in query
        except Exception:
            log.debug("mcp oauth: bad callback request", exc_info=True)
        detail = "" if ok else f"?error={self._params.get('error', 'missing code')}"
        body = (
            "<html><body><h1>lecode — authorization received</h1>"
            "<p>You can close this tab and return to the terminal.</p></body></html>"
            if ok
            else "<html><body><h1>lecode — authorization failed</h1>"
            f"<p>No authorization code received{detail}.</p></body></html>"
        )
        response = (
            "HTTP/1.1 200 OK\r\ncontent-type: text/html; charset=utf-8\r\n"
            f"content-length: {len(body.encode())}\r\nconnection: close\r\n\r\n{body}"
        )
        try:
            writer.write(response.encode())
            await writer.drain()
        except (ConnectionError, RuntimeError):
            pass
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
        if ok or self._params:
            self._done.set()


async def browser_redirect_handler(url: str, notify: Notify | None = None) -> None:
    """SDK ``redirect_handler``: open the browser; always surface the URL."""
    opened = False
    try:
        opened = await asyncio.to_thread(webbrowser.open, url)
    except Exception as e:
        log.debug("mcp oauth: webbrowser.open failed: %s", e)
    message = (
        "mcp oauth: opened the authorization page in your browser"
        if opened
        else "mcp oauth: open this URL in a browser to authorize:"
    )
    if notify is not None:
        try:
            result = notify(f"{message}\n{url}")
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            log.debug("mcp oauth: notify failed: %s", e)


async def build_oauth_provider(
    server_name: str,
    server_url: str,
    *,
    notify: Notify | None = None,
    timeout: float = CALLBACK_TIMEOUT_S,
) -> Any:
    """An ``OAuthClientProvider`` wired to file storage + the local listener.

    The listener is bound eagerly so its address can be registered as the
    ``redirect_uri`` (the SDK builds the authorization URL before invoking the
    callback handler). The provider exposes the listener as
    ``callback_listener``; the owner closes it when the connection attempt
    ends.
    """
    from mcp.client.auth import OAuthClientProvider
    from mcp.shared.auth import OAuthClientMetadata

    listener = OAuthCallbackListener(timeout=timeout)
    await listener.start()
    metadata = OAuthClientMetadata(
        client_name="lecode",
        redirect_uris=[listener.redirect_uri],
    )
    provider = OAuthClientProvider(
        server_url=server_url,
        client_metadata=metadata,
        storage=FileTokenStorage(server_name),
        redirect_handler=lambda url: browser_redirect_handler(url, notify),
        callback_handler=listener.wait,
    )
    provider.callback_listener = listener
    return provider
