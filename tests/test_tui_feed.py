"""Tests for the append-only feed rendering."""

from __future__ import annotations

from io import StringIO

import pytest
from rich.console import Console

from lecode.config.models import Config
from lecode.tui.feed import TOOL_CALL_MAX_LEN, Feed
from lecode.tui.themes import load_theme


@pytest.fixture
def theme():
    return load_theme("default", Config())


def make_feed(theme, collapse_thinking=True, force_terminal=False):
    out = StringIO()
    console = Console(
        record=True,
        file=out,
        width=200,
        force_terminal=force_terminal or None,
        color_system="truecolor" if force_terminal else None,
    )
    return Feed(console, theme, collapse_thinking=collapse_thinking), out


def test_user_message(theme):
    feed, out = make_feed(theme)
    feed.user_message("fix the bug")
    assert "> fix the bug" in out.getvalue()


def test_assistant_text_renders_markdown(theme):
    feed, out = make_feed(theme)
    feed.assistant_text("# Title\n\nsome **bold** text")
    rendered = out.getvalue()
    assert "Title" in rendered
    assert "bold" in rendered


def test_streaming_prints_tokens_as_they_arrive(theme):
    feed, out = make_feed(theme)
    feed.stream_start()
    feed.stream_token("Hello, ")
    feed.stream_token("world")
    feed.stream_end()
    assert "Hello, world" in out.getvalue()


def test_stream_tokens_not_interpreted_as_markup(theme):
    feed, out = make_feed(theme)
    feed.stream_start()
    feed.stream_token("**not bold** [not-a-style]")
    feed.stream_end()
    assert "**not bold** [not-a-style]" in out.getvalue()


def test_thinking_collapsed_one_liner(theme):
    feed, out = make_feed(theme, collapse_thinking=True)
    feed.stream_start()
    for _ in range(3):
        feed.stream_token("hmm ", thinking=True)
    feed.stream_token("answer")
    feed.stream_end()
    rendered = out.getvalue()
    assert "▸ thinking (3 tokens)" in rendered
    assert "hmm" not in rendered
    assert "answer" in rendered


def test_thinking_expanded_panel(theme):
    feed, out = make_feed(theme, collapse_thinking=False)
    feed.stream_start()
    feed.stream_token("let me think", thinking=True)
    feed.stream_end()
    rendered = out.getvalue()
    assert "let me think" in rendered
    assert "thinking" in rendered
    assert "tokens)" not in rendered


def test_thinking_only_stream(theme):
    feed, out = make_feed(theme)
    feed.stream_start()
    feed.stream_token("hmm", thinking=True)
    feed.stream_end()
    assert "▸ thinking (1 tokens)" in out.getvalue()


def test_tool_call(theme):
    feed, out = make_feed(theme)
    feed.tool_call("bash", "ls -la")
    assert "⚙ bash(ls -la)" in out.getvalue()


def test_tool_call_truncates_long_args(theme):
    feed, out = make_feed(theme)
    feed.tool_call("bash", "x" * 300)
    line = next(line for line in out.getvalue().splitlines() if line.startswith("⚙"))
    assert line.endswith("…")
    assert len(line) == TOOL_CALL_MAX_LEN


def test_tool_result_elides_long_content(theme):
    feed, out = make_feed(theme)
    content = "\n".join(f"line {i}" for i in range(25))
    feed.tool_result("bash", content)
    rendered = out.getvalue()
    assert "line 0" in rendered
    assert "line 9" in rendered
    assert "line 10" not in rendered
    assert "… (15 more lines)" in rendered


def test_tool_result_short_content_not_elided(theme):
    feed, out = make_feed(theme)
    feed.tool_result("bash", "a\nb\nc")
    rendered = out.getvalue()
    assert "a\nb\nc" in rendered
    assert "more lines" not in rendered


def test_tool_result_error_styled(theme, monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    feed, out = make_feed(theme, force_terminal=True)
    feed.tool_result("bash", "boom", is_error=True)
    rendered = out.getvalue()
    assert "boom" in rendered
    assert "\x1b[" in rendered


def test_error_info_retrying(theme):
    feed, out = make_feed(theme)
    feed.error("something broke")
    feed.info("fyi")
    feed.retrying(2, 1.5)
    rendered = out.getvalue()
    assert "✗ something broke" in rendered
    assert "fyi" in rendered
    assert "retrying (attempt 2) in 1.5s…" in rendered


def test_error_emits_ansi_when_terminal(theme, monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    feed, out = make_feed(theme, force_terminal=True)
    feed.error("kaput")
    rendered = out.getvalue()
    assert "kaput" in rendered
    assert "\x1b[" in rendered
