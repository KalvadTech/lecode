"""Tests for the lsp_diagnostics tool and the write/edit diagnostics appendix."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from lecode.agent.builder import build_runtime
from lecode.agent.tools import edit, write
from lecode.agent.tools.base import ToolContext
from lecode.config.models import Config, LspServerOverride
from lecode.lsp.manager import LspManager
from lecode.lsp.tool import make_tool
from lecode.permission import Decision, PermissionChecker

MOCK = str(Path(__file__).parent / "mock_lsp_server.py")


@pytest.fixture
async def lsp_ctx(tmp_path, monkeypatch):
    """A yolo, auto-approving tool context with a mock-backed LSP manager."""
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("MOCK_LSP_MODE", "pull")
    config = Config()
    config.lsp.servers["python"] = LspServerOverride(command=[sys.executable, MOCK])
    checker = PermissionChecker(config, mode="yolo", cwd=tmp_path)
    ctx = ToolContext(cwd=tmp_path, config=config, permission_checker=checker, auto_approve=True)
    manager = LspManager(config, tmp_path, init_timeout=2.0, request_timeout=1.5)
    ctx.extras["lsp"] = manager
    yield ctx
    await manager.shutdown()


async def test_write_appends_diagnostics_section(lsp_ctx, tmp_path):
    result = await write.make_tool().run({"path": "a.py", "content": "x = 1\n"}, lsp_ctx)
    assert not result.is_error
    assert "## Diagnostics" in result.content
    assert "1:5 error undefined name 'foo'" in result.content
    assert "3:1 warning unused variable" in result.content


async def test_write_no_section_when_no_diagnostics(lsp_ctx, monkeypatch):
    monkeypatch.setenv("MOCK_LSP_DIAGS", "[]")
    result = await write.make_tool().run({"path": "a.py", "content": "x = 1\n"}, lsp_ctx)
    assert not result.is_error
    assert "## Diagnostics" not in result.content


async def test_write_no_section_on_server_failure(lsp_ctx, monkeypatch):
    monkeypatch.setenv("MOCK_LSP_MODE", "crash")
    result = await write.make_tool().run({"path": "a.py", "content": "x = 1\n"}, lsp_ctx)
    assert not result.is_error  # the write itself succeeded
    assert "## Diagnostics" not in result.content


async def test_edit_appends_diagnostics_section(lsp_ctx, tmp_path):
    target = tmp_path / "a.py"
    target.write_text("x = 1\n")
    await write.make_tool().run({"path": "a.py", "content": "x = 1\n"}, lsp_ctx)
    result = await edit.make_tool().run(
        {"path": "a.py", "old_string": "x = 1", "new_string": "x = 2"}, lsp_ctx
    )
    assert not result.is_error
    assert "## Diagnostics" in result.content
    assert "1:5 error undefined name 'foo'" in result.content


async def test_tool_returns_formatted_list(lsp_ctx, tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    result = await make_tool().run({"path": "a.py"}, lsp_ctx)
    assert not result.is_error
    assert "1:5 error undefined name 'foo'" in result.content
    assert "3:1 warning unused variable" in result.content


async def test_tool_reports_no_diagnostics(lsp_ctx, tmp_path, monkeypatch):
    monkeypatch.setenv("MOCK_LSP_DIAGS", "[]")
    (tmp_path / "a.py").write_text("x = 1\n")
    result = await make_tool().run({"path": "a.py"}, lsp_ctx)
    assert "no diagnostics" in result.content


async def test_tool_no_server_for_filetype(lsp_ctx):
    result = await make_tool().run({"path": "README.md"}, lsp_ctx)
    assert "no LSP server" in result.content


async def test_tool_reports_disabled(tool_ctx):
    result = await make_tool().run({"path": "a.py"}, tool_ctx)  # no lsp extra installed
    assert "LSP is disabled" in result.content


def test_readonly_mode_allows_the_tool():
    checker = PermissionChecker(Config(), mode="readonly")
    assert checker.check("lsp_diagnostics", {"path": "x.py"}).decision == Decision.ALLOW


def test_build_runtime_registers_manager_and_tool(tmp_path, monkeypatch):
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    runtime = build_runtime(Config(), tmp_path)
    assert "lsp_diagnostics" in runtime.registry.names()
    assert isinstance(runtime.ctx.extras.get("lsp"), LspManager)


def test_build_runtime_skips_lsp_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    config = Config()
    config.lsp.enabled = False
    runtime = build_runtime(config, tmp_path)
    assert "lsp_diagnostics" not in runtime.registry.names()
    assert "lsp" not in runtime.ctx.extras
