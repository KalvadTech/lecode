"""Tests for the append-only feed rendering."""

from __future__ import annotations

from io import StringIO

import pytest
from rich.console import Console

from lecode.tui.feed import TOOL_CALL_MAX_LEN, Feed
from lecode.tui.themes import THEME


@pytest.fixture
def theme():
    return THEME


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


# -- logbook stamps ------------------------------------------------------------


def test_lines_carry_timestamps(theme):
    feed, out = make_feed(theme)
    feed.user_message("hi")
    feed.tool_call("bash", "ls")
    feed.tool_result("bash", "ok")
    feed.info("note")
    feed.error("boom")
    feed.turn_stats(
        context_used=100,
        context_window=1000,
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.001,
        session_cost_usd=0.002,
    )
    stamped = [ln for ln in out.getvalue().splitlines() if ln.startswith("[")]
    # user, tool call, tool result, info, error, turn stats
    assert len(stamped) == 6
    assert all(len(ln) >= 10 and ln[1:3].isdigit() and ln[3] == ":" for ln in stamped)


def test_metrics_suffix_on_action_lines(theme):
    feed, out = make_feed(theme)
    feed.metrics = lambda: (12_300, 200_000, 0.0412)
    feed.user_message("hi")
    feed.tool_call("bash", "ls")
    feed.tool_result("bash", "ok")
    rendered = out.getvalue()
    assert rendered.count("ctx 12.3k/200.0k · $0.0412") == 3


def test_no_metrics_suffix_when_unbound(theme):
    feed, out = make_feed(theme)
    feed.user_message("hi")
    assert "ctx" not in out.getvalue()


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
    line = next(line for line in out.getvalue().splitlines() if "⚙" in line)
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


def test_turn_stats_line(theme):
    feed, out = make_feed(theme)
    feed.turn_stats(
        context_used=36_864,
        context_window=204_800,
        input_tokens=4_100,
        output_tokens=900,
        cost_usd=0.0062,
        session_cost_usd=0.0314,
        tool_calls=3,
        turns=2,
        elapsed_s=12.34,
    )
    rendered = out.getvalue()
    assert "ctx 36.9k/204.8k (18%)" in rendered
    assert "↑4.1k in" in rendered
    assert "↓0.9k out" in rendered
    assert "$0.0062 this answer" in rendered
    assert "$0.0314 total" in rendered
    assert "3 tool calls" in rendered
    assert "2 rounds" in rendered
    assert "12.3s" in rendered


def test_turn_stats_line_plain_answer_hides_tool_count(theme):
    feed, out = make_feed(theme)
    feed.turn_stats(
        context_used=1_000,
        context_window=200_000,
        input_tokens=1_000,
        output_tokens=50,
        cost_usd=0.0001,
        session_cost_usd=0.0001,
        turns=1,
        elapsed_s=0.4,
    )
    rendered = out.getvalue()
    assert "tool call" not in rendered
    assert "1 round" in rendered
    assert "0.4s" in rendered


def test_error_emits_ansi_when_terminal(theme, monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    feed, out = make_feed(theme, force_terminal=True)
    feed.error("kaput")
    rendered = out.getvalue()
    assert "kaput" in rendered
    assert "\x1b[" in rendered


# -- transient activity indicator ------------------------------------------------


def test_activity_line_shown(theme):
    feed, out = make_feed(theme)
    feed.activity_start("thinking")
    assert "thinking…" in out.getvalue()


def test_activity_erased_by_next_output(theme):
    feed, out = make_feed(theme)
    feed.activity_start("thinking")
    feed.stream_start()
    feed.stream_token("hello")
    text = out.getvalue()
    assert "\x1b[K" in text  # erase-line emitted
    assert text.rstrip().endswith("hello")


def test_activity_tick_cycles_frames(theme):
    feed, out = make_feed(theme)
    feed.activity_start("thinking")
    feed.activity_tick()
    feed.activity_tick()
    text = out.getvalue()
    assert text.count("thinking…") == 3  # initial draw + two ticks


def test_activity_stop_idempotent(theme):
    feed, out = make_feed(theme)
    feed.activity_stop()  # nothing shown: no-op, no crash
    feed.activity_start("running bash")
    feed.activity_stop()
    feed.activity_stop()
    assert out.getvalue().count("running bash…") == 1


def test_tool_call_erases_activity(theme):
    feed, out = make_feed(theme)
    feed.activity_start("thinking")
    feed.tool_call("bash", "ls")
    assert "⚙ bash(ls)" in out.getvalue()
