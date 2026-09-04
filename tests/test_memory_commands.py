"""Tests for the /memory command handlers and the build_runtime wiring."""

from __future__ import annotations

import pytest

from lecode.agent.builder import build_runtime
from lecode.config.models import Config
from lecode.memory.commands import memory_command
from lecode.memory.store import MemoryStore, memory_root


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path / "mem")


def test_show_empty(store):
    assert memory_command(["show"], store) == "(long-term memory is empty)"


def test_show_content(store):
    store.write_long_term("facts here")
    assert "facts here" in memory_command(["show"], store)


def test_edit_returns_instructions_and_content(store):
    store.write_long_term("current content")
    output = memory_command(["edit"], store)
    assert "Edit MEMORY.md" in output
    assert "current content" in output


def test_search_subcommand(store):
    store.write_long_term("keyword in memory")
    output = memory_command(["search", "keyword"], store)
    assert "MEMORY.md (1 hits):" in output
    assert "keyword" in memory_command(["search", "keyword"], store)
    assert "usage" in memory_command(["search"], store)


def test_log_subcommand(store):
    assert "no daily log" in memory_command(["log"], store)
    store.append_daily("did work", day="2020-03-04")
    assert "did work" in memory_command(["log", "2020-03-04"], store)


def test_notes_subcommand(store):
    assert memory_command(["notes"], store) == "(no notes)"
    store.write_note("a", "1")
    store.write_note("b", "2")
    assert memory_command(["notes"], store) == "- a\n- b"


def test_unknown_subcommand_shows_usage(store):
    assert "usage" in memory_command(["bogus"], store)


def test_no_args_defaults_to_show(store):
    store.write_long_term("shown by default")
    assert "shown by default" in memory_command([], store)


# -- build_runtime wiring -------------------------------------------------------


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("LECODE_SKILLS_DIR", str(tmp_path / "no-skills"))
    return tmp_path


def test_runtime_registers_memory_tools(cwd):
    runtime = build_runtime(Config(), cwd)
    for name in ("memory_write", "memory_edit", "memory_read", "memory_search"):
        assert runtime.registry.get(name) is not None
    assert isinstance(runtime.ctx.extras["memory"], MemoryStore)


def test_runtime_without_memory(cwd):
    config = Config()
    config.memory.enabled = False
    runtime = build_runtime(config, cwd)
    assert runtime.registry.get("memory_read") is None
    assert "memory" not in runtime.ctx.extras
    assert "## Memory" not in runtime.system_prompt


def test_injection_present_when_memory_exists(cwd):
    MemoryStore(memory_root(cwd)).write_long_term("injected fact")
    runtime = build_runtime(Config(), cwd)
    assert "## Memory" in runtime.system_prompt
    assert "injected fact" in runtime.system_prompt


def test_injection_absent_when_memory_empty(cwd):
    runtime = build_runtime(Config(), cwd)
    assert "## Memory" not in runtime.system_prompt
