"""Tests for the async JSON-RPC stdio client (against the mock LSP server)."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

from lecode.lsp.jsonrpc import JsonRpcClient, JsonRpcError

MOCK = str(Path(__file__).parent / "mock_lsp_server.py")


async def spawn(mode: str = "echo") -> tuple[asyncio.subprocess.Process, JsonRpcClient]:
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        MOCK,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        env={**os.environ, "MOCK_LSP_MODE": mode},
    )
    return process, JsonRpcClient(process)


async def test_request_response_round_trip():
    _, client = await spawn()
    params = {"nested": {"list": [1, 2, 3]}, "text": "héllo"}
    assert await client.request("echo", params) == params
    await client.close()


async def test_concurrent_requests_resolve_by_id():
    _, client = await spawn()
    first, second = await asyncio.gather(
        client.request("echo", {"n": 1}), client.request("echo", {"n": 2})
    )
    assert first == {"n": 1}
    assert second == {"n": 2}
    await client.close()


async def test_notification_dispatch():
    _, client = await spawn()
    received = asyncio.Event()
    seen: list[dict] = []
    client.on_notification("poked", lambda params: (seen.append(params), received.set()))
    await client.notify("poke", {"hello": "world"})
    await asyncio.wait_for(received.wait(), timeout=5)
    assert seen == [{"hello": "world"}]
    await client.close()


async def test_server_request_gets_null_reply():
    _, client = await spawn()
    replied = asyncio.Event()
    seen: list[dict] = []
    client.on_notification("got-reply", lambda params: (seen.append(params), replied.set()))
    await client.notify("ask-client")
    await asyncio.wait_for(replied.wait(), timeout=5)
    assert seen == [{"value": None}]  # we implement no server→client requests
    await client.close()


async def test_error_response_raises():
    _, client = await spawn()
    with pytest.raises(JsonRpcError, match="method not found"):
        await client.request("fail")
    await client.close()


async def test_process_death_fails_pending():
    _, client = await spawn()
    with pytest.raises(JsonRpcError):
        await client.request("die")  # the mock exits without answering
    # The connection is broken for later requests too.
    with pytest.raises(JsonRpcError):
        await client.request("echo", {})
    await client.close()


async def test_close_is_idempotent():
    process, client = await spawn()
    await client.close()
    await client.close()
    assert process.returncode is not None
    with pytest.raises(JsonRpcError, match="closed"):
        await client.request("echo", {})
