"""Tests for subagent dispatch: run_subagent, the task tool, @agent routing."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from tests.fakes import FakeProvider
from tests.test_tui_app import make_app, make_blocking_app, wait_for

from lecode.agent.builder import build_runtime
from lecode.agent.runner import AgentRunner
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
    assert "advisor" not in tool_names
    assert {"read", "grep", "list_dir"} <= tool_names


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


async def test_child_events_render_inline(tmp_path, monkeypatch):
    script = [
        {"tool_calls": [{"name": "task", "arguments": '{"prompt": "scan", "agent": "explore"}'}]},
        {"tool_calls": [{"name": "list_dir", "arguments": '{"path": "."}'}]},  # child
        {"text": "scan result"},  # child final
        {"text": "parent final"},  # parent turn 2
    ]
    app, _, out = make_app(tmp_path, monkeypatch, script)
    await app._submit("please scan")
    await app._turn_task
    rendered = out.getvalue()
    assert "⚙ task(" in rendered
    assert "explore/list_dir" in rendered
    assert "explore finished" in rendered
    assert "parent final" in rendered


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
    # A side query: nothing persisted to the session.
    assert app.store.load_messages(app.session) == []
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
    assert "subagent model: (inherits main: openai/gpt-5-mini)" in out.getvalue()
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
