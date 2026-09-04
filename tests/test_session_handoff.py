"""Tests for session handoff briefs."""

from __future__ import annotations

import pytest

from lecode.session import SessionStore
from lecode.session.handoff import build_handoff_prompt, files_touched, handoff


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    return SessionStore()


@pytest.fixture
def source(store):
    s = store.create("origin", cwd="/tmp/project", model="openai/gpt-5-mini")
    store.append_message(s, {"role": "user", "content": "please refactor the parser"})
    store.append_message(
        s,
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "read", "arguments": '{"path": "src/parser.py"}'},
                },
                {
                    "id": "c2",
                    "type": "function",
                    "function": {"name": "write", "arguments": '{"path": "src/parser.py"}'},
                },
                {
                    "id": "c3",
                    "type": "function",
                    "function": {"name": "edit", "arguments": '{"path": "src/lexer.py"}'},
                },
            ],
        },
    )
    store.append_message(s, {"role": "tool", "tool_call_id": "c1", "content": "file body"})
    store.append_message(s, {"role": "assistant", "content": "refactor done"})
    return s


def test_brief_structure(store, source):
    brief = build_handoff_prompt(store.load_messages(source), goal_hint="finish the refactor")
    assert brief.startswith("# Session handoff")
    assert "finish the refactor" in brief
    assert "**user**: please refactor the parser" in brief
    assert "**assistant**: refactor done" in brief
    assert "`src/parser.py`" in brief
    assert "`src/lexer.py`" in brief


def test_brief_without_goal_hint(store, source):
    brief = build_handoff_prompt(store.load_messages(source))
    assert "(no goal hint provided)" in brief


def test_files_touched_dedupes_and_ignores_other_tools(store, source):
    files = files_touched(store.load_messages(source))
    assert files == ["src/parser.py", "src/lexer.py"]


def test_brief_truncates_long_messages(store):
    s = store.create("long", cwd="/tmp")
    store.append_message(s, {"role": "user", "content": "x" * 500})
    brief = build_handoff_prompt(store.load_messages(s))
    assert "x" * 250 not in brief
    assert "…" in brief


def test_handoff_creates_seeded_session(store, source):
    new = handoff(source, store, "follow-up")
    assert new.name == "follow-up"
    assert new.meta.cwd == source.meta.cwd
    assert new.meta.model == source.meta.model
    messages = store.load_messages(new)
    assert len(messages) == 1
    assert messages[0].role == "user"
    assert "# Session handoff" in messages[0].message["content"]


def test_handoff_name_dedup(store, source):
    handoff(source, store, "follow-up")
    second = handoff(source, store, "follow-up")
    assert second.name == "follow-up-2"
