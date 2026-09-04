"""Tests for the advisor tool, handoff mode, and the /advisor command."""

from __future__ import annotations

import asyncio

from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from tests.fakes import FakeProvider
from tests.test_tui_app import make_app, wait_for

from lecode.agent.builder import build_runtime
from lecode.agent.tools.advisor import (
    AdvisorTool,
    serialize_message,
    truncate_context,
)
from lecode.agent.tools.base import ToolContext
from lecode.config.models import Config
from lecode.permission import Decision, PermissionChecker
from lecode.providers.openai_compat import ProviderError


def make_ctx(tmp_path, script, conversation=None, **advisor_overrides) -> ToolContext:
    """A yolo tool context with a scripted provider and conversation extras."""
    config = Config()
    config.advisor.enabled = True
    for key, value in advisor_overrides.items():
        setattr(config.advisor, key, value)
    checker = PermissionChecker(config, mode="yolo", cwd=tmp_path)
    ctx = ToolContext(cwd=tmp_path, config=config, permission_checker=checker, auto_approve=True)
    ctx.extras["provider"] = FakeProvider(script)
    ctx.extras["conversation"] = conversation if conversation is not None else []
    return ctx


def make_advisor_app(tmp_path, monkeypatch, script, **overrides):
    config = Config()
    config.advisor.enabled = True
    for key, value in overrides.items():
        setattr(config.advisor, key, value)
    return make_app(tmp_path, monkeypatch, script, config=config)


def _conversation(n: int, size: int = 100) -> list[dict]:
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"message {i} " + "x" * size}
        for i in range(n)
    ]


# -- tool: model mode -------------------------------------------------------------


async def test_question_round_trip(tmp_path):
    ctx = make_ctx(tmp_path, [{"text": "use a state machine"}])
    tool = AdvisorTool()
    result = await tool.run({"question": "how should I structure this?"}, ctx)
    assert not result.is_error
    assert result.content == "Advisor: use a state machine"
    request = ctx.extras["provider"].requests[0]
    assert request["model"] == ctx.config.llm.model
    assert request["messages"][0]["role"] == "system"
    assert "how should I structure this?" in request["messages"][1]["content"]


async def test_own_system_prompt(tmp_path):
    ctx = make_ctx(tmp_path, [{"text": "ok"}])
    await AdvisorTool().run({"question": "q"}, ctx)
    system = ctx.extras["provider"].requests[0]["messages"][0]["content"]
    assert "senior staff engineer" in system


async def test_custom_advisor_model(tmp_path):
    ctx = make_ctx(tmp_path, [{"text": "ok"}], model="anthropic/claude-opus-4.1")
    await AdvisorTool().run({"question": "q"}, ctx)
    assert ctx.extras["provider"].requests[0]["model"] == "anthropic/claude-opus-4.1"


async def test_focus_hint_included(tmp_path):
    ctx = make_ctx(tmp_path, [{"text": "ok"}])
    await AdvisorTool().run({"question": "q", "focus": "src/cache.py"}, ctx)
    assert "Focus: src/cache.py" in ctx.extras["provider"].requests[0]["messages"][1]["content"]


async def test_conversation_context_in_request(tmp_path):
    conversation = [
        {"role": "user", "content": "build a cache"},
        {"role": "assistant", "content": "done"},
    ]
    ctx = make_ctx(tmp_path, [{"text": "ok"}], conversation=conversation)
    await AdvisorTool().run({"question": "q"}, ctx)
    user = ctx.extras["provider"].requests[0]["messages"][1]["content"]
    assert "## Conversation so far" in user
    assert "user: build a cache" in user


async def test_empty_conversation_marked(tmp_path):
    ctx = make_ctx(tmp_path, [{"text": "ok"}])
    await AdvisorTool().run({"question": "q"}, ctx)
    assert "(empty)" in ctx.extras["provider"].requests[0]["messages"][1]["content"]


async def test_disabled_explains_how_to_enable(tmp_path):
    ctx = make_ctx(tmp_path, [], enabled=False)
    result = await AdvisorTool().run({"question": "q"}, ctx)
    assert result.is_error
    assert "advisor is disabled" in result.content
    assert "/advisor on" in result.content


async def test_missing_question_is_error(tmp_path):
    ctx = make_ctx(tmp_path, [{"text": "ok"}])
    tool = AdvisorTool()
    result = await tool.run({}, ctx)
    assert result.is_error and "needs a question" in result.content
    assert tool.uses == 0  # validation happens before the budget is spent


async def test_max_uses_exhaustion(tmp_path):
    ctx = make_ctx(tmp_path, [{"text": "one"}], max_uses=1)
    tool = AdvisorTool()
    first = await tool.run({"question": "q1"}, ctx)
    assert not first.is_error
    second = await tool.run({"question": "q2"}, ctx)
    assert second.is_error
    assert "advisor budget exhausted (1/1)" in second.content


async def test_uses_counter_across_calls(tmp_path):
    ctx = make_ctx(tmp_path, [{"text": "a"}, {"text": "b"}], max_uses=5)
    tool = AdvisorTool()
    await tool.run({"question": "q1"}, ctx)
    await tool.run({"question": "q2"}, ctx)
    assert tool.uses == 2


async def test_provider_failure_is_error_result(tmp_path):
    ctx = make_ctx(tmp_path, [{"error": ProviderError("boom", retryable=False)}])
    result = await AdvisorTool().run({"question": "q"}, ctx)
    assert result.is_error and "advisor call failed: boom" in result.content


async def test_empty_advisor_response_is_error(tmp_path):
    ctx = make_ctx(tmp_path, [{}])
    result = await AdvisorTool().run({"question": "q"}, ctx)
    assert result.is_error and "empty response" in result.content


async def test_no_provider_is_error(tmp_path):
    ctx = make_ctx(tmp_path, [])
    del ctx.extras["provider"]
    result = await AdvisorTool().run({"question": "q"}, ctx)
    assert result.is_error and "no provider" in result.content


async def test_stream_only_provider_fallback(tmp_path):
    """Providers exposing only ``stream_chat`` work via ``collect``."""

    class StreamOnly:
        def __init__(self):
            self.inner = FakeProvider([{"text": "streamed advice"}])

        def stream_chat(self, messages, model, **kwargs):
            return self.inner.stream_chat(messages, model, **kwargs)

    ctx = make_ctx(tmp_path, [])
    ctx.extras["provider"] = StreamOnly()
    result = await AdvisorTool().run({"question": "q"}, ctx)
    assert result.content == "Advisor: streamed advice"


# -- context serialization / truncation ---------------------------------------------


def test_serialize_tool_calls_one_line():
    message = {
        "role": "assistant",
        "content": "checking",
        "tool_calls": [
            {"function": {"name": "bash", "arguments": '{"command": "ls -la"}'}},
        ],
    }
    line = serialize_message(message)
    assert line == 'assistant: checking [bash({"command": "ls -la"})]'
    assert "\n" not in line


def test_serialize_multimodal_content_parts():
    message = {
        "role": "user",
        "content": [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {}}],
    }
    assert serialize_message(message) == "user: look"


def test_truncation_budget_head_tail_marker():
    lines = [serialize_message(m) for m in _conversation(60, size=100)]
    result = truncate_context(_conversation(60, size=100), 1)
    assert len(result.encode()) <= 1024 + 64  # budget + marker slack
    assert lines[0] in result  # head preserved
    assert lines[-1] in result  # tail preserved
    assert "messages elided" in result


def test_truncation_untouched_when_within_budget():
    conversation = _conversation(4, size=20)
    result = truncate_context(conversation, 32)
    assert "elided" not in result
    assert "message 0" in result and "message 3" in result


def test_truncation_counts_elided_messages():
    result = truncate_context(_conversation(60, size=100), 1)
    marker = next(line for line in result.splitlines() if "elided" in line)
    kept = len([line for line in result.splitlines() if "elided" not in line])
    assert marker == f"[… {60 - kept} messages elided …]"


# -- handoff mode -------------------------------------------------------------------


async def test_handoff_callback_answer_becomes_result(tmp_path):
    seen = []

    async def human(question, focus):
        seen.append((question, focus))
        return "split it into two modules"

    ctx = make_ctx(tmp_path, [], mode="handoff")
    ctx.extras["advisor_handoff"] = human
    result = await AdvisorTool().run({"question": "how?", "focus": "big.py"}, ctx)
    assert not result.is_error
    assert result.content == "Advisor (human): split it into two modules"
    assert seen == [("how?", "big.py")]


async def test_handoff_without_callback_is_explanatory_error(tmp_path):
    ctx = make_ctx(tmp_path, [], mode="handoff")
    result = await AdvisorTool().run({"question": "how?"}, ctx)
    assert result.is_error
    assert "handoff is unavailable" in result.content


async def test_handoff_declined_is_error(tmp_path):
    async def human(question, focus):
        return None

    ctx = make_ctx(tmp_path, [], mode="handoff")
    ctx.extras["advisor_handoff"] = human
    result = await AdvisorTool().run({"question": "how?"}, ctx)
    assert result.is_error and "declined" in result.content


async def test_handoff_counts_against_budget(tmp_path):
    async def human(question, focus):
        return "guidance"

    ctx = make_ctx(tmp_path, [], mode="handoff", max_uses=1)
    ctx.extras["advisor_handoff"] = human
    tool = AdvisorTool()
    assert not (await tool.run({"question": "q1"}, ctx)).is_error
    assert "budget exhausted" in (await tool.run({"question": "q2"}, ctx)).content


# -- permissions / registration ---------------------------------------------------------


def test_advisor_auto_allowed_in_readonly_mode(tmp_path):
    config = Config()
    checker = PermissionChecker(config, mode="readonly", cwd=tmp_path)
    assert checker.check("advisor", {"question": "q"}).decision == Decision.ALLOW


def test_advisor_registered_by_build_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    runtime = build_runtime(Config(), tmp_path)
    assert runtime.registry.get("advisor") is not None


# -- /advisor command -------------------------------------------------------------


async def test_advisor_status(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/advisor")
    rendered = out.getvalue()
    assert "advisor: off" in rendered
    assert "mode: model" in rendered
    assert f"model: {app.config.llm.model}" in rendered
    assert "uses: 0/5" in rendered
    assert "context limit: 32 KB" in rendered


async def test_advisor_on_off(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/advisor on")
    assert app.config.advisor.enabled is True
    await app.handle_command("/advisor off")
    assert app.config.advisor.enabled is False
    assert "advisor: on" in out.getvalue() and "advisor: off" in out.getvalue()


async def test_advisor_handoff_toggles_mode(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/advisor handoff")
    assert app.config.advisor.mode == "handoff"
    await app.handle_command("/advisor handoff")
    assert app.config.advisor.mode == "model"
    assert "advisor mode: handoff" in out.getvalue()


async def test_advisor_model_and_validation(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/advisor model anthropic/claude-opus-4.1")
    assert app.config.advisor.model == "anthropic/claude-opus-4.1"
    await app.handle_command("/advisor model")
    assert "usage: /advisor model" in out.getvalue()


async def test_advisor_max_uses_and_validation(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/advisor max-uses 10")
    assert app.config.advisor.max_uses == 10
    await app.handle_command("/advisor max-uses 0")
    assert "usage: /advisor max-uses" in out.getvalue()
    assert app.config.advisor.max_uses == 10


async def test_advisor_context_limit_and_validation(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/advisor context-limit 64")
    assert app.config.advisor.context_limit_kb == 64
    await app.handle_command("/advisor context-limit nope")
    assert "usage: /advisor context-limit" in out.getvalue()


async def test_advisor_unknown_subcommand(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/advisor bogus")
    assert "usage: /advisor" in out.getvalue()


async def test_advisor_uses_reset_on_session_switch(tmp_path, monkeypatch):
    app, _, _ = make_advisor_app(tmp_path, monkeypatch, [{"text": "a"}])
    tool = app.runtime.registry.get("advisor")
    tool.uses = 3
    await app.handle_command("/new fresh")
    assert app.runtime.registry.get("advisor").uses == 0


# -- end-to-end through the agent loop ---------------------------------------------


async def test_advisor_call_during_turn(tmp_path, monkeypatch):
    script = [
        {"tool_calls": [{"name": "advisor", "arguments": '{"question": "which approach?"}'}]},
        {"text": "go with the simple one"},  # the advisor's answer
        {"text": "final answer"},
    ]
    app, provider, out = make_advisor_app(tmp_path, monkeypatch, script)
    await app._submit("help me decide")
    await app._turn_task
    rendered = out.getvalue()
    assert "Advisor: go with the simple one" in rendered
    assert "final answer" in rendered
    # request 0 = main turn, request 1 = advisor (its own prompt + question)
    advisor_request = provider.requests[1]
    assert "senior staff engineer" in advisor_request["messages"][0]["content"]
    assert "which approach?" in advisor_request["messages"][-1]["content"]
    # the advisor saw the conversation so far
    assert "help me decide" in advisor_request["messages"][-1]["content"]


async def test_advisor_disabled_during_turn(tmp_path, monkeypatch):
    script = [
        {"tool_calls": [{"name": "advisor", "arguments": '{"question": "q"}'}]},
        {"text": "never mind then"},
    ]
    app, _, out = make_app(tmp_path, monkeypatch, script)  # advisor disabled by default
    await app._submit("go")
    await app._turn_task
    assert "advisor is disabled" in out.getvalue()
    assert "never mind then" in out.getvalue()


async def test_handoff_end_to_end_via_pipe(tmp_path, monkeypatch):
    """Handoff mode: the advisor question is answered inline via Enter."""
    script = [
        {"tool_calls": [{"name": "advisor", "arguments": '{"question": "which db?"}'}]},
        {"text": "done with sqlite"},
    ]
    app, _, out = make_advisor_app(tmp_path, monkeypatch, script, mode="handoff")
    with create_pipe_input() as inp:
        inp.send_text("pick a database\n")
        task = asyncio.ensure_future(app.run(input=inp, output=DummyOutput()))
        await wait_for(lambda: "advisor asks: which db?" in out.getvalue())
        inp.send_text("use sqlite\n")
        await wait_for(lambda: "done with sqlite" in out.getvalue())
        inp.send_text("/quit\n")
        assert await task == 0
    rendered = out.getvalue()
    assert "Advisor (human): use sqlite" in rendered
