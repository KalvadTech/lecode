"""Tests for the LSP manager: lazy spawn, handshake, diagnostics, fail-open."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from lecode.config.models import Config, LspServerOverride
from lecode.lsp.manager import LspManager

MOCK = str(Path(__file__).parent / "mock_lsp_server.py")


def lsp_config(mode: str = "pull", command: list[str] | None = None) -> Config:
    """A config whose "python" server is the mock (or a given command)."""
    config = Config()
    config.lsp.servers["python"] = LspServerOverride(command=command or [sys.executable, MOCK])
    return config


@pytest.fixture
def manager(tmp_path, monkeypatch):
    monkeypatch.setenv("MOCK_LSP_MODE", "pull")
    return LspManager(lsp_config(), tmp_path, init_timeout=2.0, request_timeout=1.5)


async def test_lazy_spawn_on_first_use(tmp_path, monkeypatch):
    monkeypatch.setenv("MOCK_LSP_MODE", "pull")
    mgr = LspManager(lsp_config(), tmp_path, init_timeout=2.0, request_timeout=1.5)
    assert mgr._servers == {}
    (tmp_path / "a.py").write_text("x = 1\n")
    await mgr.diagnostics_for(tmp_path / "a.py")
    assert len(mgr._servers) == 1
    await mgr.shutdown()


async def test_initialize_handshake_and_document_sync(tmp_path, monkeypatch):
    log_file = tmp_path / "methods.log"
    monkeypatch.setenv("MOCK_LSP_MODE", "pull")
    monkeypatch.setenv("MOCK_LSP_LOG", str(log_file))
    mgr = LspManager(lsp_config(), tmp_path, init_timeout=2.0, request_timeout=1.5)
    (tmp_path / "a.py").write_text("x = 1\n")
    await mgr.diagnostics_for(tmp_path / "a.py")
    methods = log_file.read_text().splitlines()
    assert methods[:2] == ["initialize", "initialized"]
    assert "textDocument/didOpen" in methods
    assert "textDocument/diagnostic" in methods
    assert methods.index("textDocument/didOpen") < methods.index("textDocument/diagnostic")
    await mgr.shutdown()


async def test_pull_diagnostics_parsed(manager, tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    diagnostics = await manager.diagnostics_for(tmp_path / "a.py")
    assert len(diagnostics) == 2
    error, warning = diagnostics
    assert (error.line, error.col, error.severity) == (1, 5, "error")  # 0-based → 1-based
    assert error.message == "undefined name 'foo'"
    assert error.source == "mock"
    assert (warning.line, warning.severity) == (3, "warning")
    await manager.shutdown()


async def test_one_server_per_root(tmp_path, monkeypatch):
    monkeypatch.setenv("MOCK_LSP_MODE", "pull")
    mgr = LspManager(lsp_config(), tmp_path, init_timeout=2.0, request_timeout=1.5)
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "b.py").write_text("y = 2\n")
    await mgr.diagnostics_for(tmp_path / "a.py")
    await mgr.diagnostics_for(tmp_path / "b.py")
    assert len(mgr._servers) == 1  # same root → shared server
    nested = tmp_path / "pkg"
    (nested / "pyproject.toml").parent.mkdir(exist_ok=True)
    (nested / "pyproject.toml").write_text("[project]\nname = 'pkg'\n")
    (nested / "c.py").write_text("z = 3\n")
    await mgr.diagnostics_for(nested / "c.py")
    assert len(mgr._servers) == 2  # own root marker → own server
    await mgr.shutdown()


async def test_didchange_on_second_query(tmp_path, monkeypatch):
    log_file = tmp_path / "methods.log"
    monkeypatch.setenv("MOCK_LSP_MODE", "pull")
    monkeypatch.setenv("MOCK_LSP_LOG", str(log_file))
    mgr = LspManager(lsp_config(), tmp_path, init_timeout=2.0, request_timeout=1.5)
    target = tmp_path / "a.py"
    target.write_text("x = 1\n")
    await mgr.diagnostics_for(target)
    target.write_text("x = 2\n")
    await mgr.diagnostics_for(target)
    methods = log_file.read_text().splitlines()
    assert methods.count("textDocument/didOpen") == 1
    assert "textDocument/didChange" in methods
    await mgr.shutdown()


async def test_published_diagnostics_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("MOCK_LSP_MODE", "push")  # pull errors; push on didOpen
    mgr = LspManager(lsp_config(), tmp_path, init_timeout=2.0, request_timeout=1.5)
    (tmp_path / "a.py").write_text("x = 1\n")
    diagnostics = await mgr.diagnostics_for(tmp_path / "a.py")
    assert [d.message for d in diagnostics] == ["undefined name 'foo'", "unused variable"]
    await mgr.shutdown()


async def test_crash_mid_request_fails_open(tmp_path, monkeypatch):
    monkeypatch.setenv("MOCK_LSP_MODE", "crash")
    mgr = LspManager(lsp_config(), tmp_path, init_timeout=2.0, request_timeout=1.5)
    (tmp_path / "a.py").write_text("x = 1\n")
    assert await mgr.diagnostics_for(tmp_path / "a.py") == []  # no raise
    await mgr.shutdown()


async def test_missing_binary_fails_open(tmp_path):
    config = Config()
    config.lsp.servers["python"] = LspServerOverride(command=["no-such-binary-lecode-test"])
    mgr = LspManager(config, tmp_path)
    (tmp_path / "a.py").write_text("x = 1\n")
    assert await mgr.diagnostics_for(tmp_path / "a.py") == []
    assert mgr._servers == {}  # never spawned


async def test_unknown_filetype_no_spawn(manager, tmp_path):
    (tmp_path / "README.md").write_text("# hi\n")
    assert await manager.diagnostics_for(tmp_path / "README.md") == []
    assert manager._servers == {}


async def test_disabled_config_is_noop(tmp_path):
    config = lsp_config()
    config.lsp.enabled = False
    mgr = LspManager(config, tmp_path)
    (tmp_path / "a.py").write_text("x = 1\n")
    assert await mgr.diagnostics_for(tmp_path / "a.py") == []
    assert mgr._servers == {}


async def test_initialize_timeout_fails_open(tmp_path, monkeypatch):
    monkeypatch.setenv("MOCK_LSP_MODE", "hang")
    mgr = LspManager(lsp_config(), tmp_path, init_timeout=0.3, request_timeout=0.3)
    (tmp_path / "a.py").write_text("x = 1\n")
    assert await mgr.diagnostics_for(tmp_path / "a.py") == []
    assert mgr._servers == {}  # the failed spawn is not cached
    await mgr.shutdown()


async def test_shutdown_is_idempotent(manager, tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    await manager.diagnostics_for(tmp_path / "a.py")
    assert len(manager._servers) == 1
    await manager.shutdown()
    assert manager._servers == {}
    await manager.shutdown()  # second call is a no-op
    assert await manager.diagnostics_for(tmp_path / "a.py") == []  # closed manager is inert
