"""A minimal async JSON-RPC client over stdio (LSP wire framing).

Messages are framed with ``Content-Length`` headers per the LSP base
protocol. A background reader task dispatches responses to pending futures
by ``id`` and notifications to registered handlers. Server→client requests
get a ``null`` result (we implement none of them). When the process dies or
the stream breaks, every pending future fails with :class:`JsonRpcError` —
callers stay fail-open.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import Callable
from typing import Any

log = logging.getLogger(__name__)


class JsonRpcError(Exception):
    """A JSON-RPC error response, or a broken/dead transport."""


class JsonRpcClient:
    """One JSON-RPC peer over a subprocess's stdin/stdout pipes."""

    def __init__(self, process: asyncio.subprocess.Process) -> None:
        self._process = process
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._handlers: dict[str, list[Callable[[dict[str, Any]], None]]] = {}
        self._next_id = 0
        self._write_lock = asyncio.Lock()
        self._closed = False
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._reader_task = asyncio.ensure_future(self._read_loop())

    def on_notification(self, method: str, handler: Callable[[dict[str, Any]], None]) -> None:
        """Register ``handler(params)`` for an incoming notification method."""
        self._handlers.setdefault(method, []).append(handler)

    @property
    def alive(self) -> bool:
        """Whether the server process is still running."""
        return not self._closed and self._process.returncode is None

    async def request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        """Send a request and await its result; raises :class:`JsonRpcError`."""
        if self._closed:
            raise JsonRpcError("client is closed")
        self._next_id += 1
        request_id = self._next_id
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            await self._send(
                {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}}
            )
        except Exception as e:
            self._pending.pop(request_id, None)
            raise JsonRpcError(f"failed to send {method}: {e}") from e
        return await future

    async def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        """Send a notification (no response expected); failures are dropped."""
        if self._closed:
            return
        try:
            await self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})
        except Exception as e:
            log.debug("lsp: notify %s failed: %s", method, e)

    async def _send(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        frame = f"Content-Length: {len(body)}\r\n\r\n".encode() + body
        async with self._write_lock:
            self._process.stdin.write(frame)
            await self._process.stdin.drain()

    # -- reader loop -----------------------------------------------------------

    async def _read_loop(self) -> None:
        try:
            while True:
                message = await self._read_message()
                if message is None:
                    break
                self._dispatch(message)
        except Exception as e:  # framing errors, EOF mid-body, …
            log.debug("lsp: reader loop ended: %s", e)
        self._fail_pending(JsonRpcError("language server connection closed"))

    async def _read_message(self) -> dict[str, Any] | None:
        """Read one framed message; ``None`` on clean EOF before any header."""
        stdout = self._process.stdout
        length: int | None = None
        saw_header = False
        while True:
            line = await stdout.readline()
            if not line:
                if saw_header:
                    raise JsonRpcError("EOF in message headers")
                return None
            line = line.strip()
            if not line:
                break  # end of headers
            saw_header = True
            if line.lower().startswith(b"content-length:"):
                length = int(line.split(b":", 1)[1].strip())
        if length is None:
            raise JsonRpcError("missing Content-Length header")
        body = await stdout.readexactly(length)
        return json.loads(body.decode("utf-8"))

    def _dispatch(self, message: dict[str, Any]) -> None:
        if "method" in message and "id" in message:
            # A server→client request: we implement none of them.
            task = asyncio.ensure_future(self._reply_null(message["id"]))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
        elif "method" in message:
            for handler in self._handlers.get(str(message["method"]), []):
                try:
                    handler(message.get("params") or {})
                except Exception as e:
                    log.debug("lsp: notification handler failed: %s", e)
        elif "id" in message:
            future = self._pending.pop(message["id"], None)
            if future is None or future.done():
                return
            if "error" in message:
                error = message["error"] or {}
                future.set_exception(
                    JsonRpcError(f"{error.get('message', 'request failed')} ({error.get('code')})")
                )
            else:
                future.set_result(message.get("result"))

    async def _reply_null(self, request_id: Any) -> None:
        try:
            await self._send({"jsonrpc": "2.0", "id": request_id, "result": None})
        except Exception as e:
            log.debug("lsp: null reply failed: %s", e)

    def _fail_pending(self, error: JsonRpcError) -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_exception(error)
        self._pending.clear()

    # -- lifecycle ---------------------------------------------------------------

    async def close(self) -> None:
        """Kill the process and stop the reader; idempotent, never raises."""
        if self._closed:
            return
        self._closed = True
        try:
            if self._process.returncode is None:
                self._process.kill()
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
        except Exception as e:
            log.debug("lsp: process kill failed: %s", e)
        self._reader_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._reader_task
        self._fail_pending(JsonRpcError("client closed"))
