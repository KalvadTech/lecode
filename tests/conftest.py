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
