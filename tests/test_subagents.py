"""Tests for subagent dispatch: run_subagent, the task tool, @agent routing."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from prompt_toolkit.formatted_text import to_formatted_text
from tests.fakes import FakeProvider
from tests.test_tui_app import make_app, make_blocking_app, wait_for

from lecode.agent.builder import build_runtime
from lecode.agent.runner import AgentRunner, ToolResult
from lecode.agent.runner import Done as RunDone
from lecode.agent.tools import ToolRegistry
from lecode.agent.tools.task import make_tool
from lecode.config.models import Config
from lecode.extras import subagents
from lecode.extras.subagents import (
    SUBAGENT_RESPONSE_CAP,
    SubagentError,
    run_subagent,
)
from lecode.providers.types import Done, TokenDelta, ToolCallDelta
from lecode.session import SessionStore


def make_runtime(tmp_path, monkeypatch, provider, config=None):
    """A built runtime with the provider seam installed on the context."""
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("LECODE_SKILLS_DIR", str(tmp_path / "global-skills"))
    config = config or Config()
    runtime = build_runtime(config, tmp_path)
    runtime.ctx.extras["provider"] = provider
    return runtime


async def _scripted_stream(entry):
    """One scripted turn as a stream: text, then tool calls, then Done."""
    for text in entry.get("text") or []:
        yield TokenDelta(text=text)
    for index, call in enumerate(entry.get("tool_calls") or []):
        yield ToolCallDelta(
            index=index,
            id=str(call.get("id", f"call_{index}")),
            name=str(call["name"]),
            arguments_chunk=str(call.get("arguments", "{}")),
        )
    yield Done(finish_reason=entry.get("finish_reason") or "stop")


class NeverProvider:
    """Streams nothing, ever (for timeout tests)."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def stream_chat(self, messages, model, tools=None, **kwargs):
        self.requests.append({"messages": [dict(m) for m in messages], "model": model})
        return self._stream()

    async def _stream(self):
        await asyncio.Event().wait()  # never set
        yield  # pragma: no cover — unreachable


class ParallelProvider:
    """Parent turns scripted; child turns block until both are in flight.

    If the parent dispatched subagent calls sequentially, the first child
    would wait forever and the wait_for below would blow up the test.
    """

    def __init__(self, parent_script) -> None:
        self.parent_script = list(parent_script)
        self.requests: list[dict[str, Any]] = []
        self._children = 0
        self._both = asyncio.Event()

    def stream_chat(self, messages, model, tools=None, **kwargs):
        self.requests.append({"messages": [dict(m) for m in messages], "model": model})
        is_child = not any(t["function"]["name"] == "task" for t in tools or [])
        if is_child:
            self._children += 1
            if self._children >= 2:
                self._both.set()
            return self._child_stream()
        return _scripted_stream(self.parent_script.pop(0))

    async def _child_stream(self):
        await asyncio.wait_for(self._both.wait(), timeout=5)
        yield TokenDelta(text="child answer")
        yield Done(finish_reason="stop")


class BlockingChildProvider:
    """Parent turn scripted; the child stream blocks forever after starting."""

    def __init__(self, parent_entry) -> None:
        self.parent_entry = parent_entry
        self.requests: list[dict[str, Any]] = []
        self.child_started = asyncio.Event()

    def stream_chat(self, messages, model, tools=None, **kwargs):
        self.requests.append({"messages": [dict(m) for m in messages], "model": model})
        is_child = not any(t["function"]["name"] == "task" for t in tools or [])
        if is_child:
            return self._child_stream()
        return _scripted_stream(self.parent_entry)

    async def _child_stream(self):
        self.child_started.set()
        await asyncio.Event().wait()  # never set
        yield  # pragma: no cover — unreachable


# -- run_subagent -------------------------------------------------------------


async def test_child_uses_agent_prompt_and_lean_registry(tmp_path, monkeypatch):
    provider = FakeProvider([{"text": "found it"}])
    runtime = make_runtime(tmp_path, monkeypatch, provider)
    outcome = await run_subagent(
        runtime.ctx, runtime.registry, runtime.agents, name="explore", prompt="scan"
    )
    assert outcome.text == "found it"
    request = provider.requests[0]
    assert request["messages"][0]["role"] == "system"
    assert "read-only exploration agent" in request["messages"][0]["content"]
    assert request["messages"][1] == {"role": "user", "content": "scan"}
    tool_names = {t["function"]["name"] for t in request["tools"]}
    assert "task" not in tool_names  # no recursion
    assert "ask_user" not in tool_names  # children decide themselves
    assert {"read", "grep", "list_dir"} <= tool_names


async def test_subagent_progress_carries_run_identity(tmp_path, monkeypatch):
    """Every forwarded child event is tagged with one run's id, agent, description."""
    provider = FakeProvider([{"text": "done"}])
    runtime = make_runtime(tmp_path, monkeypatch, provider)
    seen: list[Any] = []
    outcome = await run_subagent(
        runtime.ctx,
        runtime.registry,
        runtime.agents,
        name="explore",
        prompt="scan the repo",
        description="Explore src",
        on_event=seen.append,
    )
    assert outcome.text == "done"
    assert seen
    run_ids = {progress.run_id for progress in seen}
    assert len(run_ids) == 1
    assert next(iter(run_ids))
    assert {progress.agent for progress in seen} == {"explore"}
    assert {progress.description for progress in seen} == {"Explore src"}
    assert any(isinstance(progress.event, RunDone) for progress in seen)


async def test_completed_run_persists_activity_trail(tmp_path, monkeypatch):
    """A finished child run lands in the session as one bounded record."""
    provider = FakeProvider(
        [
            {"tool_calls": [{"name": "read", "arguments": '{"path": "note.txt"}'}]},
            {"text": "read it"},
        ]
    )
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("LECODE_SKILLS_DIR", str(tmp_path / "global-skills"))
    store = SessionStore(config_dir=tmp_path / "cfg")
    session = store.create("agent-run", tmp_path)
    runtime = build_runtime(Config(), tmp_path, session=session, store=store)
    runtime.ctx.extras["provider"] = provider
    (tmp_path / "note.txt").write_text("hello trail", encoding="utf-8")

    outcome = await run_subagent(
        runtime.ctx,
        runtime.registry,
        runtime.agents,
        name="explore",
        prompt="read note.txt",
        description="Read note",
    )

    assert outcome.run_id
    runs = store.load_agent_runs(session)
    assert len(runs) == 1
    run = runs[0]
    assert run["run_id"] == outcome.run_id
    assert run["agent"] == "explore"
    assert run["description"] == "Read note"
    assert run["status"] == "ok"
    assert run["answer"] == "read it"
    assert run["turns"] == 2
    assert run["tool_calls"][0]["name"] == "read"
    assert "hello trail" in run["tool_calls"][0]["result"]
    assert run["tool_calls"][0]["is_error"] is False


async def test_cancelled_run_persists_cancelled_status(tmp_path, monkeypatch):
    provider = BlockingChildProvider({"text": "unused"})
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("LECODE_SKILLS_DIR", str(tmp_path / "global-skills"))
    store = SessionStore(config_dir=tmp_path / "cfg")
    session = store.create("agent-run", tmp_path)
    runtime = build_runtime(Config(), tmp_path, session=session, store=store)
    runtime.ctx.extras["provider"] = provider

    task = asyncio.ensure_future(
        run_subagent(
            runtime.ctx,
            runtime.registry,
            runtime.agents,
            name="explore",
            prompt="hang",
            description="Cancelled run",
        )
    )
    await asyncio.wait_for(provider.child_started.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5)

    runs = store.load_agent_runs(session)
    assert len(runs) == 1
    assert runs[0]["status"] == "cancelled"
    assert runs[0]["description"] == "Cancelled run"


async def test_failed_run_persists_error_status(tmp_path, monkeypatch):
    monkeypatch.setattr(subagents, "SUBAGENT_TIMEOUT_S", 0.05)
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("LECODE_SKILLS_DIR", str(tmp_path / "global-skills"))
    store = SessionStore(config_dir=tmp_path / "cfg")
    session = store.create("agent-run", tmp_path)
    runtime = build_runtime(Config(), tmp_path, session=session, store=store)
    runtime.ctx.extras["provider"] = NeverProvider()

    with pytest.raises(SubagentError, match="timed out"):
        await run_subagent(
            runtime.ctx,
            runtime.registry,
            runtime.agents,
            name="explore",
            prompt="hang",
            description="Never finishes",
        )

    runs = store.load_agent_runs(session)
    assert len(runs) == 1
    assert runs[0]["status"] == "error"
    assert runs[0]["description"] == "Never finishes"
    assert "timed out" in runs[0]["answer"]


async def test_child_ctx_drops_question_callback(tmp_path, monkeypatch):
    """The child ctx (dataclasses.replace of the parent's) must not inherit the
    interactive question callback even if the parent has one installed."""
    provider = FakeProvider([{"text": "done"}])
    runtime = make_runtime(tmp_path, monkeypatch, provider)
    runtime.ctx.question_callback = object()  # sentinel: must not leak
    captured = {}
    real_runner = subagents.AgentRunner

    class SpyRunner(real_runner):
        def __init__(self, provider, registry, ctx, **kwargs):
            captured["ctx"] = ctx
            captured["registry"] = registry
            super().__init__(provider, registry, ctx, **kwargs)

    monkeypatch.setattr(subagents, "AgentRunner", SpyRunner)
    await run_subagent(runtime.ctx, runtime.registry, runtime.agents, name="explore", prompt="x")
    assert "ask_user" not in captured["registry"].names()
    assert captured["ctx"].question_callback is None


async def test_usage_totals_propagate(tmp_path, monkeypatch):
    provider = FakeProvider(
        [{"text": "answer", "usage": {"input_tokens": 10, "output_tokens": 5, "cost_usd": 0.001}}]
    )
    runtime = make_runtime(tmp_path, monkeypatch, provider)
    outcome = await run_subagent(
        runtime.ctx, runtime.registry, runtime.agents, name="explore", prompt="x"
    )
    assert outcome.turns == 1
    assert outcome.input_tokens == 10
    assert outcome.output_tokens == 5
    assert outcome.cost_usd == pytest.approx(0.001)


async def test_unknown_agent_lists_available(tmp_path, monkeypatch):
    provider = FakeProvider([])
    runtime = make_runtime(tmp_path, monkeypatch, provider)
    with pytest.raises(SubagentError, match=r"unknown subagent: nope.*explore"):
        await run_subagent(runtime.ctx, runtime.registry, runtime.agents, name="nope", prompt="x")
    assert provider.requests == []


async def test_primary_agent_not_invocable(tmp_path, monkeypatch):
    provider = FakeProvider([])
    runtime = make_runtime(tmp_path, monkeypatch, provider)
    with pytest.raises(SubagentError, match="unknown subagent: build"):
        await run_subagent(runtime.ctx, runtime.registry, runtime.agents, name="build", prompt="x")
    assert provider.requests == []


async def test_missing_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("LECODE_SKILLS_DIR", str(tmp_path / "global-skills"))
    runtime = build_runtime(Config(), tmp_path)  # no provider seam installed
    with pytest.raises(SubagentError, match="no provider available"):
        await run_subagent(
            runtime.ctx, runtime.registry, runtime.agents, name="explore", prompt="x"
        )


async def test_response_cap(tmp_path, monkeypatch):
    provider = FakeProvider([{"text": "x" * (SUBAGENT_RESPONSE_CAP + 100)}])
    runtime = make_runtime(tmp_path, monkeypatch, provider)
    outcome = await run_subagent(
        runtime.ctx, runtime.registry, runtime.agents, name="explore", prompt="x"
    )
    assert outcome.text.endswith("… (truncated)")
    assert len(outcome.text) == SUBAGENT_RESPONSE_CAP + len("\n… (truncated)")


async def test_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(subagents, "SUBAGENT_TIMEOUT_S", 0.05)
    provider = NeverProvider()
    runtime = make_runtime(tmp_path, monkeypatch, provider)
    with pytest.raises(SubagentError, match="timed out"):
        await run_subagent(
            runtime.ctx, runtime.registry, runtime.agents, name="explore", prompt="x"
        )


async def test_explore_overlay_denies_write(tmp_path, monkeypatch):
    # yolo base mode would allow the write; the explore overlay still denies.
    config = Config()
    config.permissions.mode = "yolo"
    provider = FakeProvider(
        [
            {
                "tool_calls": [
                    {"name": "write", "arguments": '{"file_path": "x.txt", "content": "hi"}'}
                ]
            },
            {"text": "tried to write"},
        ]
    )
    runtime = make_runtime(tmp_path, monkeypatch, provider, config=config)
    outcome = await run_subagent(
        runtime.ctx, runtime.registry, runtime.agents, name="explore", prompt="write a file"
    )
    assert outcome.text == "tried to write"
    tool_msgs = [m for m in provider.requests[1]["messages"] if m["role"] == "tool"]
    assert tool_msgs and tool_msgs[0]["content"].startswith("denied")
    assert not (tmp_path / "x.txt").exists()


async def test_child_does_not_clobber_parent_extras(tmp_path, monkeypatch):
    provider = FakeProvider([{"text": "ok"}])
    runtime = make_runtime(tmp_path, monkeypatch, provider)
    parent_conversation = [{"role": "user", "content": "parent"}]
    runtime.ctx.extras["conversation"] = parent_conversation
    await run_subagent(runtime.ctx, runtime.registry, runtime.agents, name="explore", prompt="x")
    assert runtime.ctx.extras["conversation"] is parent_conversation
    assert runtime.ctx.extras["provider"] is provider


# -- model selection ------------------------------------------------------------


async def test_subagent_model_defaults_to_main(tmp_path, monkeypatch):
    provider = FakeProvider([{"text": "ok"}])
    runtime = make_runtime(tmp_path, monkeypatch, provider)
    await run_subagent(runtime.ctx, runtime.registry, runtime.agents, name="explore", prompt="x")
    assert provider.requests[0]["model"] == runtime.ctx.config.llm.model


async def test_subagent_model_override(tmp_path, monkeypatch):
    config = Config()
    config.agent.subagent_model = "deepseek/deepseek-r1"
    provider = FakeProvider([{"text": "ok"}])
    runtime = make_runtime(tmp_path, monkeypatch, provider, config=config)
    await run_subagent(runtime.ctx, runtime.registry, runtime.agents, name="explore", prompt="x")
    assert provider.requests[0]["model"] == "deepseek/deepseek-r1"


async def test_agent_frontmatter_model_wins(tmp_path, monkeypatch):
    agents_dir = tmp_path / ".lecode" / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "scout.md").write_text(
        "---\ndescription: scout\nmode: subagent\nmodel: anthropic/custom-x\n---\nScout things.\n",
        encoding="utf-8",
    )
    config = Config()
    config.agent.subagent_model = "deepseek/deepseek-r1"
    provider = FakeProvider([{"text": "ok"}])
    runtime = make_runtime(tmp_path, monkeypatch, provider, config=config)
    await run_subagent(runtime.ctx, runtime.registry, runtime.agents, name="scout", prompt="x")
    assert provider.requests[0]["model"] == "anthropic/custom-x"
    assert "Scout things." in provider.requests[0]["messages"][0]["content"]


# -- hooks ------------------------------------------------------------------------


async def test_subagent_hooks_fire(tmp_path, monkeypatch):
    log = tmp_path / "hooks.log"
    config = Config()
    config.hooks = {
        "SubagentStart": [f"cat >> {log} && echo >> {log}"],
        "SubagentEnd": [f"cat >> {log} && echo >> {log}"],
    }
    provider = FakeProvider([{"text": "done"}])
    runtime = make_runtime(tmp_path, monkeypatch, provider, config=config)
    assert runtime.hooks is not None
    await run_subagent(runtime.ctx, runtime.registry, runtime.agents, name="explore", prompt="scan")
    lines = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [line["event"] for line in lines] == ["SubagentStart", "SubagentEnd"]
    assert all(line["agent"] == "explore" for line in lines)
    assert lines[0]["prompt"] == "scan"
    assert lines[1]["result"] == {"content": "done", "is_error": False}


# -- the task tool ------------------------------------------------------------------


async def test_task_tool_returns_child_text(tmp_path, monkeypatch):
    provider = FakeProvider(
        [{"text": "child answer", "usage": {"input_tokens": 3, "output_tokens": 2}}]
    )
    runtime = make_runtime(tmp_path, monkeypatch, provider)
    message, result = await runtime.registry.dispatch_result(
        "c1", "task", '{"prompt": "scan the repo"}', runtime.ctx
    )
    assert not result.is_error
    assert result.content == "child answer"
    assert message["content"] == "child answer"
    assert result.metadata["agent"] == "explore"
    assert result.metadata["turns"] == 1
    assert result.metadata["input_tokens"] == 3


async def test_task_tool_unknown_agent_is_error_result(tmp_path, monkeypatch):
    provider = FakeProvider([])
    runtime = make_runtime(tmp_path, monkeypatch, provider)
    _, result = await runtime.registry.dispatch_result(
        "c1", "task", '{"prompt": "x", "agent": "nope"}', runtime.ctx
    )
    assert result.is_error
    assert "unknown subagent: nope" in result.content


async def test_task_tool_needs_prompt(tmp_path, monkeypatch):
    provider = FakeProvider([])
    runtime = make_runtime(tmp_path, monkeypatch, provider)
    _, result = await runtime.registry.dispatch_result(
        "c1", "task", '{"agent": "explore"}', runtime.ctx
    )
    assert result.is_error
    assert "needs a prompt" in result.content


async def test_task_tool_without_runtime_seams(tool_ctx):
    registry = ToolRegistry([make_tool()])
    _, result = await registry.dispatch_result("c1", "task", '{"prompt": "x"}', tool_ctx)
    assert result.is_error
    assert "unavailable" in result.content


# -- concurrency / cancellation -----------------------------------------------------


async def test_parallel_task_calls(tmp_path, monkeypatch):
    parent_script = [
        {
            "tool_calls": [
                {"name": "task", "arguments": '{"prompt": "scan a"}'},
                {"name": "task", "arguments": '{"prompt": "scan b"}'},
            ]
        },
        {"text": ["parent summary"]},
    ]
    provider = ParallelProvider(parent_script)
    runtime = make_runtime(tmp_path, monkeypatch, provider)
    runner = AgentRunner(provider, runtime.registry, runtime.ctx)
    result = await runner.run([{"role": "user", "content": "do two things"}])
    assert result.final_text == "parent summary"
    child_requests = [r for r in provider.requests if r["messages"][0]["role"] == "system"]
    assert len(child_requests) == 2


async def test_cancelling_parent_cancels_child(tmp_path, monkeypatch):
    provider = BlockingChildProvider(
        {"tool_calls": [{"name": "task", "arguments": '{"prompt": "scan"}'}]}
    )
    runtime = make_runtime(tmp_path, monkeypatch, provider)
    runner = AgentRunner(provider, runtime.registry, runtime.ctx)
    task = asyncio.ensure_future(runner.run([{"role": "user", "content": "go"}]))
    await asyncio.wait_for(provider.child_started.wait(), timeout=5)
    task.cancel()
    # If the child were not cancelled with the parent, this await would hang.
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5)


# -- TUI integration -------------------------------------------------------------------


async def test_child_events_feed_roster_and_one_summary_line(tmp_path, monkeypatch):
    script = [
        {
            "tool_calls": [
                {
                    "name": "task",
                    "arguments": (
                        '{"prompt": "scan", "agent": "explore", "description": "Scan repo"}'
                    ),
                }
            ]
        },
        {"tool_calls": [{"name": "list_dir", "arguments": '{"path": "."}'}]},  # child
        {"text": "scan result"},  # child final
        {"text": "parent final"},  # parent turn 2
    ]
    app, _, out = make_app(tmp_path, monkeypatch, script)
    await app._submit("please scan")
    await app._turn_task

    runs = app.roster.runs()
    assert len(runs) == 1
    run = runs[0]
    assert run.description == "Scan repo"
    assert run.status == "ok"
    assert run.answer == "scan result"
    assert [entry.name for entry in run.activity] == ["list_dir"]

    rendered = out.getvalue()
    # Per-call child lines no longer flood the transcript; the run gets one
    # attributed summary line and stays inspectable through the roster.
    assert "explore/list_dir" not in rendered
    assert "Scan repo" in rendered
    assert "parent final" in rendered


async def test_task_result_event_carries_run_id_metadata(tmp_path, monkeypatch):
    provider = FakeProvider(
        [
            {
                "tool_calls": [
                    {"name": "task", "arguments": '{"prompt": "scan", "agent": "explore"}'}
                ]
            },
            {"text": "scan result"},
            {"text": "parent final"},
        ]
    )
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("LECODE_SKILLS_DIR", str(tmp_path / "global-skills"))
    store = SessionStore(config_dir=tmp_path / "cfg")
    session = store.create("agent-run", tmp_path)
    runtime = build_runtime(Config(), tmp_path, session=session, store=store)
    runtime.ctx.extras["provider"] = provider
    events: list[Any] = []
    runner = AgentRunner(provider, runtime.registry, runtime.ctx)
    await runner.run([{"role": "user", "content": "go"}], on_event=events.append)

    task_results = [
        event
        for event in events
        if isinstance(event, ToolResult) and event.name == "task" and not event.is_error
    ]
    assert task_results
    run_id = task_results[0].metadata.get("run_id")
    assert run_id
    assert run_id == store.load_agent_runs(session)[0]["run_id"]


async def test_roster_panel_visible_while_child_runs(tmp_path, monkeypatch):
    provider = BlockingChildProvider(
        {
            "tool_calls": [
                {
                    "name": "task",
                    "arguments": (
                        '{"prompt": "scan", "agent": "explore", "description": "Scan repo"}'
                    ),
                }
            ]
        }
    )
    app, _, _ = make_app(tmp_path, monkeypatch, [])
    app._runner.provider = provider
    app._runtime.ctx.extras["provider"] = provider
    await app._submit("go")
    await wait_for(lambda: provider.child_started.is_set())

    assert app._roster_visible()
    rows = "".join(fragment[1] for fragment in to_formatted_text(app._roster_text()))
    assert "Scan repo" in rows

    run_id = app.roster.runs()[0].run_id
    assert app.open_agent_run(run_id)
    detail = "".join(fragment[1] for fragment in to_formatted_text(app._roster_text()))
    assert "Scan repo" in detail

    app._turn_task.cancel()
    await app._turn_task
    assert app.roster.runs()[0].status == "cancelled"
    app.close_agent_run()
    assert not app._roster_visible()


async def test_runs_command_lists_and_opens_detail(tmp_path, monkeypatch):
    script = [
        {
            "tool_calls": [
                {
                    "name": "task",
                    "arguments": (
                        '{"prompt": "scan", "agent": "explore", "description": "Scan repo"}'
                    ),
                }
            ]
        },
        {"tool_calls": [{"name": "list_dir", "arguments": '{"path": "."}'}]},
        {"text": "scan result"},
        {"text": "parent final"},
    ]
    app, _, out = make_app(tmp_path, monkeypatch, script)
    await app._submit("please scan")
    await app._turn_task
    run_id = app.roster.runs()[0].run_id

    await app.handle_command("/runs")
    assert "Scan repo" in out.getvalue()

    await app.handle_command(f"/runs {run_id}")
    assert app.detail_run_id == run_id
    app.close_agent_run()
    assert app.detail_run_id is None


async def test_runs_picker_offers_run_ids(tmp_path, monkeypatch):
    script = [
        {"tool_calls": [{"name": "task", "arguments": '{"prompt": "scan", "agent": "explore"}'}]},
        {"text": "scan result"},
        {"text": "parent final"},
    ]
    app, _, _ = make_app(tmp_path, monkeypatch, script)
    await app._submit("please scan")
    await app._turn_task
    run_id = app.roster.runs()[0].run_id

    from prompt_toolkit.document import Document

    result = app.arg_completion_rows(Document("/runs ", 6))
    assert result is not None
    _, _, _, rows = result
    assert any(insert == run_id for insert, _, _ in rows)


async def test_detail_panel_preserves_draft(tmp_path, monkeypatch):
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    app, _, _ = make_app(tmp_path, monkeypatch, [])
    with create_pipe_input() as inp:
        app._build_app(input=inp, output=DummyOutput())
    app._input_area.buffer.text = "half-written"
    app.open_agent_run("r1")
    app.close_agent_run()
    assert app._input_area.buffer.text == "half-written"


async def test_at_agent_runs_directly(tmp_path, monkeypatch):
    script = [{"text": "42 files", "usage": {"input_tokens": 7, "output_tokens": 3}}]
    app, provider, out = make_app(tmp_path, monkeypatch, script)
    await app._submit("@explore count the files")
    await app._turn_task
    rendered = out.getvalue()
    assert "> @explore count the files" in rendered
    assert "42 files" in rendered
    request = provider.requests[0]
    assert "read-only exploration agent" in request["messages"][0]["content"]
    assert request["messages"][1] == {"role": "user", "content": "count the files"}
    # A side query: no message history, but the run itself is persisted.
    assert app.store.load_messages(app.session) == []
    runs = app.store.load_agent_runs(app.session)
    assert len(runs) == 1
    assert runs[0]["agent"] == "explore"
    assert runs[0]["status"] == "ok"
    assert runs[0]["prompt"] == "count the files"
    assert runs[0]["answer"] == "42 files"
    assert app._last_response == "42 files"
    assert app._status.input_tokens == 7


async def test_at_primary_mention_degrades_to_note(tmp_path, monkeypatch):
    script = [{"text": "noted"}]
    app, provider, _ = make_app(tmp_path, monkeypatch, script)
    await app._submit("@plan sketch it")  # plan is a primary, not a subagent
    await app._turn_task
    user_msg = provider.requests[0]["messages"][-1]
    assert user_msg["role"] == "user"
    assert "no subagent was dispatched" in user_msg["content"]
    assert "@plan" in user_msg["content"]


async def test_at_agent_while_busy_queues_as_note(tmp_path, monkeypatch):
    app, provider, _ = make_blocking_app(tmp_path, monkeypatch)
    await app._submit("first")
    await wait_for(lambda: len(provider.requests) == 1)
    await app._submit("@explore hello")
    assert app._input_queue.qsize() == 1
    assert len(provider.requests) == 1  # no subagent request while busy
    provider.blocked = False
    provider.release.set()
    await wait_for(lambda: not app._turn_running() and app._input_queue.empty())
    user_msgs = [m for req in provider.requests for m in req["messages"] if m["role"] == "user"]
    assert any("no subagent was dispatched" in str(m["content"]) for m in user_msgs)


# -- /model-subagent /models-subagent ---------------------------------------------------


async def test_model_subagent_show_set_reset(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/model-subagent")
    assert "subagent model: (inherits main: deepseek/deepseek-v4-flash)" in out.getvalue()
    await app.handle_command("/model-subagent deepseek/deepseek-r1")
    assert app.config.agent.subagent_model == "deepseek/deepseek-r1"
    await app.handle_command("/model-subagent")
    assert "subagent model: deepseek/deepseek-r1" in out.getvalue()
    await app.handle_command("/model-subagent default")
    assert app.config.agent.subagent_model is None
    assert "(inherits main" in out.getvalue()


async def test_model_subagent_unknown_model(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/model-subagent nope/nope")
    assert "unknown model: nope/nope" in out.getvalue()
    assert app.config.agent.subagent_model is None


async def test_models_subagent_marks_effective(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    app.config.agent.subagent_model = "deepseek/deepseek-r1"
    await app.handle_command("/models-subagent")
    rendered = out.getvalue()
    assert "deepseek/deepseek-r1 (subagent)" in rendered
    assert "/model-subagent default" in rendered
