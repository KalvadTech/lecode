"""Tests for deferred MCP tools behind the tool_search tool."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from lecode.agent.builder import build_runtime
from lecode.config.models import Config, McpServerConfig
from lecode.extras.mcp_client import TOOL_SEARCH_NAME, attach_mcp

MOCK = str(Path(__file__).resolve().parent / "mock_mcp_server.py")


def deferred_config(threshold: int) -> Config:
    """Mock server (4 tools) with a deferral threshold."""
    config = Config()
    config.mcp.tool_search_threshold = threshold
    config.mcp.servers["test"] = McpServerConfig(
        transport="stdio", command=sys.executable, args=[MOCK], timeout_s=5.0
    )
    return config


@pytest.fixture
async def deferred_runtime(tmp_path, monkeypatch):
    """A runtime whose mock server (4 tools) sits over the threshold (2)."""
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    runtime = build_runtime(deferred_config(2), tmp_path, auto_approve=True)
    manager = await attach_mcp(runtime.registry, runtime.ctx)
    yield runtime, manager
    await manager.shutdown()


async def test_over_threshold_defers_everything(deferred_runtime):
    runtime, manager = deferred_runtime
    assert not any(n.startswith("mcp:test:") for n in runtime.registry.names())
    assert TOOL_SEARCH_NAME in runtime.registry.names()
    assert manager.deferred_count("test") == 4


async def test_under_threshold_registers_directly(tmp_path, monkeypatch):
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    runtime = build_runtime(deferred_config(20), tmp_path, auto_approve=True)
    manager = await attach_mcp(runtime.registry, runtime.ctx)
    try:
        assert "mcp:test:echo" in runtime.registry.names()
        assert TOOL_SEARCH_NAME not in runtime.registry.names()
    finally:
        await manager.shutdown()


async def test_threshold_zero_disables_deferral(tmp_path, monkeypatch):
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    runtime = build_runtime(deferred_config(0), tmp_path, auto_approve=True)
    manager = await attach_mcp(runtime.registry, runtime.ctx)
    try:
        assert "mcp:test:echo" in runtime.registry.names()
        assert TOOL_SEARCH_NAME not in runtime.registry.names()
    finally:
        await manager.shutdown()


async def test_search_activates_and_dispatch_works(deferred_runtime):
    runtime, manager = deferred_runtime
    message = await runtime.registry.dispatch(
        "s1", TOOL_SEARCH_NAME, '{"query": "echo"}', runtime.ctx
    )
    assert "activated mcp:test:echo" in message["content"]
    assert "mcp:test:echo" in runtime.registry.names()
    assert manager.deferred_count("test") == 3
    # other tools stay deferred, so tool_search stays around
    assert TOOL_SEARCH_NAME in runtime.registry.names()
    # the activated tool calls through like a directly-registered one
    result = await runtime.registry.dispatch("c1", "mcp:test:echo", '{"text": "hi"}', runtime.ctx)
    assert result["content"] == "echo: hi"


async def test_search_matches_description(deferred_runtime):
    runtime, _ = deferred_runtime
    message = await runtime.registry.dispatch(
        "s1", TOOL_SEARCH_NAME, '{"query": "fails at the tool level"}', runtime.ctx
    )
    assert "activated mcp:test:boom" in message["content"]


async def test_search_no_match(deferred_runtime):
    runtime, _ = deferred_runtime
    message = await runtime.registry.dispatch(
        "s1", TOOL_SEARCH_NAME, '{"query": "zzz-nothing"}', runtime.ctx
    )
    assert "no deferred MCP tools match" in message["content"]
    assert "mcp:test:echo" not in runtime.registry.names()


async def test_tool_search_leaves_when_nothing_deferred(deferred_runtime):
    runtime, manager = deferred_runtime
    for query in ("echo", "boom", "crash", "flaky"):
        await runtime.registry.dispatch(
            "s1", TOOL_SEARCH_NAME, f'{{"query": "{query}"}}', runtime.ctx
        )
    assert manager.deferred_count("test") == 0
    assert TOOL_SEARCH_NAME not in runtime.registry.names()


async def test_disable_enable_cycle_keeps_deferral(deferred_runtime):
    runtime, manager = deferred_runtime
    await manager.set_enabled("test", False)
    assert TOOL_SEARCH_NAME not in runtime.registry.names()
    await manager.set_enabled("test", True)
    assert not any(n.startswith("mcp:test:") for n in runtime.registry.names())
    assert TOOL_SEARCH_NAME in runtime.registry.names()
    assert manager.deferred_count("test") == 4
