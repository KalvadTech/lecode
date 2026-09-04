"""The LSP manager: lazy per-root language servers, fail-open diagnostics.

One server per (language, root) pair, spawned on first use. Diagnostics are
pulled with ``textDocument/diagnostic``; when the pull fails, the last
``textDocument/publishDiagnostics`` push for the file is used instead.
**Every** failure mode — missing binary, spawn error, initialize timeout,
request error, server crash — degrades to an empty list with a debug log:
LSP problems never block the agent.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lecode.config.models import Config
from lecode.lsp.jsonrpc import JsonRpcClient, JsonRpcError
from lecode.lsp.registry import ServerSpec, server_for_file

log = logging.getLogger(__name__)

#: LSP DiagnosticSeverity → label (1 error, 2 warning, 3 information, 4 hint).
_SEVERITY = {1: "error", 2: "warning", 3: "info", 4: "info"}


@dataclass(frozen=True)
class Diagnostic:
    line: int  # 1-based
    col: int  # 1-based
    severity: str  # "error" | "warning" | "info"
    message: str
    source: str  # the server's own source tag, else the language name


def _parse_diagnostics(items: list[dict[str, Any]], fallback_source: str) -> list[Diagnostic]:
    parsed: list[Diagnostic] = []
    for item in items:
        try:
            start = (item.get("range") or {}).get("start") or {}
            parsed.append(
                Diagnostic(
                    line=int(start.get("line", 0)) + 1,
                    col=int(start.get("character", 0)) + 1,
                    severity=_SEVERITY.get(item.get("severity"), "info"),
                    message=str(item.get("message", "")).strip(),
                    source=str(item.get("source") or fallback_source),
                )
            )
        except (TypeError, ValueError) as e:
            log.debug("lsp: skipping malformed diagnostic: %s", e)
    return parsed


@dataclass
class _Server:
    spec: ServerSpec
    root: Path
    client: JsonRpcClient
    opened: dict[str, int] = field(default_factory=dict)  # uri → version
    published: dict[str, list[Diagnostic]] = field(default_factory=dict)  # uri → diags


class LspManager:
    """Owns language-server processes; serves :meth:`diagnostics_for`."""

    def __init__(
        self,
        config: Config,
        cwd: Path,
        *,
        init_timeout: float = 5.0,
        request_timeout: float = 2.5,
    ) -> None:
        self._config = config
        self._cwd = Path(cwd)
        self._init_timeout = init_timeout
        self._request_timeout = request_timeout
        self._servers: dict[tuple[str, str], _Server] = {}
        self._closed = False

    async def diagnostics_for(self, path: Path | str) -> list[Diagnostic]:
        """Diagnostics for ``path``; empty on any failure (fail-open)."""
        if not self._config.lsp.enabled or self._closed:
            return []
        try:
            return await self._diagnostics(Path(path))
        except Exception as e:
            log.debug("lsp: diagnostics for %s failed: %s", path, e)
            return []

    async def _diagnostics(self, path: Path) -> list[Diagnostic]:
        spec = server_for_file(str(path), self._config)
        if spec is None:
            return []
        if shutil.which(spec.command[0]) is None:
            log.debug("lsp: server binary not found: %s", spec.command[0])
            return []
        server = await self._server_for(spec, path)
        uri = path.resolve().as_uri()
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            log.debug("lsp: cannot read %s: %s", path, e)
            return []
        await self._sync_document(server, uri, text)
        try:
            result = await asyncio.wait_for(
                server.client.request("textDocument/diagnostic", {"textDocument": {"uri": uri}}),
                timeout=self._request_timeout,
            )
            items = result.get("items", []) if isinstance(result, dict) else []
            return _parse_diagnostics(items, spec.language)
        except (TimeoutError, JsonRpcError) as e:
            log.debug("lsp: pull diagnostics failed, using published: %s", e)
            return server.published.get(uri, [])

    # -- server lifecycle --------------------------------------------------------

    def _root_for(self, spec: ServerSpec, path: Path) -> Path:
        """Nearest ancestor with a root marker, else the manager's cwd."""
        current = path.resolve().parent
        for directory in (current, *current.parents):
            if any((directory / marker).exists() for marker in spec.root_markers):
                return directory
        return self._cwd

    async def _server_for(self, spec: ServerSpec, path: Path) -> _Server:
        """The running server for this spec + root, spawned lazily."""
        root = self._root_for(spec, path)
        key = (spec.language, str(root))
        server = self._servers.get(key)
        if server is not None and server.client.alive:
            return server
        server = await self._spawn(spec, root)
        self._servers[key] = server
        return server

    async def _spawn(self, spec: ServerSpec, root: Path) -> _Server:
        process = await asyncio.create_subprocess_exec(
            *spec.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            cwd=root,
        )
        client = JsonRpcClient(process)
        server = _Server(spec=spec, root=root, client=client)
        client.on_notification(
            "textDocument/publishDiagnostics",
            lambda params: self._on_published(server, params),
        )
        try:
            await asyncio.wait_for(
                client.request(
                    "initialize",
                    {
                        "processId": None,
                        "rootUri": root.as_uri(),
                        "capabilities": {
                            "textDocument": {"diagnostic": {"dynamicRegistration": False}}
                        },
                    },
                ),
                timeout=self._init_timeout,
            )
            await client.notify("initialized", {})
        except Exception:
            await client.close()  # kill the half-initialized server
            raise
        log.debug("lsp: %s server up at %s (pid %s)", spec.language, root, process.pid)
        return server

    def _on_published(self, server: _Server, params: dict[str, Any]) -> None:
        uri = str(params.get("uri") or "")
        if uri:
            server.published[uri] = _parse_diagnostics(
                params.get("diagnostics") or [], server.spec.language
            )

    async def _sync_document(self, server: _Server, uri: str, text: str) -> None:
        """didOpen the document, or didChange with full content when open."""
        version = server.opened.get(uri, 0) + 1
        server.opened[uri] = version
        if version == 1:
            await server.client.notify(
                "textDocument/didOpen",
                {
                    "textDocument": {
                        "uri": uri,
                        "languageId": server.spec.language,
                        "version": version,
                        "text": text,
                    }
                },
            )
        else:
            await server.client.notify(
                "textDocument/didChange",
                {
                    "textDocument": {"uri": uri, "version": version},
                    "contentChanges": [{"text": text}],
                },
            )

    async def shutdown(self) -> None:
        """Ask every server to exit, then kill; idempotent, never raises."""
        self._closed = True
        for server in self._servers.values():
            try:
                await asyncio.wait_for(server.client.request("shutdown"), timeout=2.0)
                await server.client.notify("exit")
            except Exception as e:
                log.debug("lsp: graceful shutdown failed: %s", e)
            try:
                await server.client.close()
            except Exception as e:
                log.debug("lsp: close failed: %s", e)
        self._servers.clear()
