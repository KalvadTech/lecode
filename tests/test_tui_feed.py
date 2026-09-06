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
    # Verbatim echo with the prompt marker: no timestamp, no suffix.
    assert out.getvalue().strip() == "> fix the bug"


# -- logbook stamps ------------------------------------------------------------


def test_lines_carry_timestamps(theme):
    feed, out = make_feed(theme)
    feed.user_message("hi")  # verbatim: no timestamp on the input echo
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
    # tool call, tool result, info, error, turn stats (not the user echo)
    assert len(stamped) == 5
    assert all(len(ln) >= 10 and ln[1:3].isdigit() and ln[3] == ":" for ln in stamped)


def test_metrics_suffix_on_action_lines(theme):
    feed, out = make_feed(theme)
    feed.metrics = lambda: (12_300, 200_000, 0.0412)
    feed.user_message("hi")  # verbatim: no metrics suffix on the input echo
    feed.tool_call("bash", "ls")
    feed.tool_result("bash", "ok")
    rendered = out.getvalue()
    assert rendered.count("ctx 12.3k/200.0k · $0.0412") == 2


def test_no_metrics_suffix_when_unbound(theme):
    feed, out = make_feed(theme)
    feed.user_message("hi")
    assert "ctx" not in out.getvalue()


def test_llm_call_logged(theme):
    feed, out = make_feed(theme)
    feed.metrics = lambda: (12_300, 200_000, 0.0412)
    feed.llm_call("openai/gpt-5", 2)
    rendered = out.getvalue()
    assert "→ openai/gpt-5 (round 2)" in rendered
    # the call line is bare: no ctx/cost suffix (that belongs to the response)
    assert "ctx" not in rendered and "$" not in rendered
    # Logbook line: carries a timestamp.
    line = next(ln for ln in rendered.splitlines() if "→ openai/gpt-5" in ln)
    assert line.startswith("[") and line[1:3].isdigit() and line[3] == ":"


def test_llm_response_logged(theme):
    feed, out = make_feed(theme)
    feed.llm_response("openai/gpt-5", 2, 92_400, 15, 0.004)
    rendered = out.getvalue()
    assert "← openai/gpt-5 (round 2) · ↑92.4k in · ↓15 out · $0.0040" in rendered
    line = next(ln for ln in rendered.splitlines() if "← openai/gpt-5" in ln)
    assert line.startswith("[") and line[1:3].isdigit() and line[3] == ":"


def test_llm_response_closes_streamed_line(theme):
    feed, out = make_feed(theme)
    feed.stream_start()
    feed.stream_token("the answer")
    feed.llm_response("m", 1, 10, 5, 0.001)  # must not append onto the answer line
    feed.stream_end()
    lines = out.getvalue().splitlines()
    assert lines[0] == "the answer"
    assert "← m (round 1)" in lines[1]


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
    assert "answer: ↑4.1k in · ↓0.9k out · $0.0062" in rendered
    assert "total: $0.0314" in rendered
    assert "ctx 36.9k/204.8k (18%)" in rendered
    assert "3 tool calls" in rendered
    assert "2 rounds" in rendered
    assert "12.3s" in rendered
    # this answer first, then the session total
    assert rendered.index("answer:") < rendered.index("total:")


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


# -- in-layout live region (stream sink) -----------------------------------------


def test_sink_receives_tokens_and_flush_prints_whole(theme):
    feed, out = make_feed(theme)
    sunk: list[str] = []
    cleared: list[bool] = []
    feed.stream_sink = sunk.append
    feed.stream_clear = lambda: cleared.append(True)
    feed.stream_start()
    feed.stream_token("Hello, ")
    feed.stream_token("world")
    assert sunk == ["Hello, ", "world"]
    assert out.getvalue() == ""  # nothing in the scrollback while streaming
    feed.stream_end()
    assert cleared == [True]
    assert "Hello, world" in out.getvalue()  # flushed whole at stream end


def test_sink_stream_closed_by_llm_response(theme):
    feed, out = make_feed(theme)
    feed.stream_sink = lambda text: None
    feed.stream_clear = lambda: None
    feed.stream_start()
    feed.stream_token("the answer")
    feed.llm_response("m", 1, 10, 5, 0.001)
    feed.stream_end()
    lines = out.getvalue().splitlines()
    assert lines[0] == "the answer"
    assert "← m (round 1)" in lines[1]


def test_sink_flush_is_idempotent(theme):
    feed, out = make_feed(theme)
    cleared: list[bool] = []
    feed.stream_sink = lambda text: None
    feed.stream_clear = lambda: cleared.append(True)
    feed.stream_start()
    feed.stream_token("once")
    feed.llm_response("m", 1, 10, 5, 0.001)
    feed.stream_end()  # must not flush twice
    assert cleared == [True]
    assert out.getvalue().count("once") == 1
