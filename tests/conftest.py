"""Shared fixtures for the test suite."""

from __future__ import annotations

import pytest

from lecode.agent.tools.base import ToolContext
from lecode.config.models import Config
from lecode.permission import PermissionChecker


@pytest.fixture
def tool_ctx(tmp_path, monkeypatch) -> ToolContext:
    """A yolo-mode, auto-approving tool context rooted at tmp_path."""
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    config = Config()
    checker = PermissionChecker(config, mode="yolo", cwd=tmp_path)
    return ToolContext(cwd=tmp_path, config=config, permission_checker=checker, auto_approve=True)


@pytest.fixture(autouse=True)
def _reset_sse_starlette_shutdown_state():
    """Reset sse-starlette's process-global shutdown state before each test.

    Its shutdown watcher polls the uvicorn server it captured from the
    process-global SIGTERM handler table. When one test's uvicorn fixture
    sets ``should_exit`` at teardown and the watcher's 0.5s poll lands before
    that event loop closes, it flips the module-global ``AppStatus.should_exit``
    (the library never resets it), and every SSE response in LATER tests is
    cancelled at birth — "SSE stream ended without a response" /
    "ASGI callable returned without completing response". Pinned by the
    shutdown regression pair in tests/test_mcp.py; drop both if sse-starlette
    ever scopes this state per event loop.
    """
    from sse_starlette.sse import AppStatus, _get_shutdown_state

    AppStatus.should_exit = False
    state = _get_shutdown_state()
    state.watcher_started = False
    state.events.clear()
