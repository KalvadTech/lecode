"""Tests for the live agent roster and its detail rendering."""

from __future__ import annotations

from lecode.agent.runner import Done, Error, LlmCall, ToolCall, ToolResult
from lecode.extras.subagents import SubagentProgress
from lecode.tui.agents import (
    ROSTER_VISIBLE_ROWS,
    AgentRoster,
    detail_lines,
    roster_lines,
)
from lecode.tui.themes import THEME


def progress(run_id: str, event, *, agent: str = "explore", description: str = "Explore src"):
    return SubagentProgress(run_id=run_id, agent=agent, description=description, event=event)


def _plain(lines) -> str:
    return "\n".join(line.plain for line in lines)


def test_roster_tracks_identity_and_tool_activity():
    roster = AgentRoster()
    roster.observe(progress("r1", LlmCall(model="m", turn=1)))
    roster.observe(progress("r1", ToolCall(id="c1", name="read", arguments='{"path": "a.py"}')))
    roster.observe(
        progress("r1", ToolResult(id="c1", name="read", content="contents", is_error=False))
    )
    run = roster.get("r1")
    assert run is not None
    assert run.agent == "explore"
    assert run.description == "Explore src"
    assert run.status == "running"
    assert run.activity[0].name == "read"
    assert run.activity[0].result == "contents"
    assert run.activity[0].running is False


def test_roster_terminal_events_set_status():
    roster = AgentRoster()
    roster.observe(progress("ok", Done(stop_reason="done", turns=2)))
    roster.observe(progress("bad", Error(message="boom")))
    assert roster.get("ok").status == "ok"
    assert roster.get("bad").status == "error"
    assert roster.get("bad").error == "boom"


def test_roster_visible_prefers_running_runs():
    roster = AgentRoster()
    for index in range(5):
        roster.observe(
            progress(f"r{index}", LlmCall(model="m", turn=1), description=f"run {index}")
        )
    roster.observe(progress("r4", Done(stop_reason="done", turns=1)))
    visible, hidden = roster.visible(limit=4)
    assert [run.run_id for run in visible] == ["r3", "r2", "r1", "r0"]
    assert hidden == 1
    assert len(visible) <= ROSTER_VISIBLE_ROWS


def test_roster_cancel_running_keeps_finished_runs():
    roster = AgentRoster()
    roster.observe(progress("r1", LlmCall(model="m", turn=1)))
    roster.observe(progress("r2", LlmCall(model="m", turn=1)))
    roster.observe(progress("r2", Done(stop_reason="done", turns=1)))
    roster.cancel_running()
    assert roster.get("r1").status == "cancelled"
    assert roster.get("r2").status == "ok"


def test_roster_lines_show_description_state_and_overflow():
    roster = AgentRoster()
    for index in range(6):
        roster.observe(
            progress(f"r{index}", LlmCall(model="m", turn=1), description=f"run {index}")
        )
    text = _plain(roster_lines(roster, THEME, width=80))
    assert "run 5" in text
    assert "+2 more" in text
    assert "running" in text


def test_roster_lines_empty_when_no_runs():
    assert roster_lines(AgentRoster(), THEME, width=80) == []


def test_detail_lines_show_tools_and_answer():
    roster = AgentRoster()
    roster.observe(progress("r1", ToolCall(id="c1", name="read", arguments='{"path": "a.py"}')))
    roster.observe(
        progress(
            "r1", ToolResult(id="c1", name="read", content="line one\nline two", is_error=False)
        )
    )
    roster.observe(progress("r1", Done(stop_reason="done", turns=1)))
    roster.finish("r1", answer="the answer")
    text = _plain(detail_lines(roster.get("r1"), THEME, width=80))
    assert "Explore src" in text
    assert "read" in text
    assert "a.py" in text
    assert "line one" in text
    assert "the answer" in text
