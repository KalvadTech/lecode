"""Tests for pierre mode: the post-task review (request vs result)."""

from __future__ import annotations

from tests.fakes import FakeProvider, sample_catalog
from tests.test_tui_app import make_app

from lecode.agent.review import review, user_request
from lecode.agent.runner import AgentRunner, Done, Review
from lecode.agent.tools.base import ToolRegistry
from lecode.providers.openai_compat import ProviderError
from lecode.session.model import EventRecord
from lecode.session.storage import SessionStore


def test_user_request_last_user_message():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "the real ask"},
    ]
    assert user_request(messages) == "the real ask"


def test_user_request_content_parts_and_absent():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look at this"},
                {"type": "image_url", "image_url": {"url": "data:..."}},
            ],
        }
    ]
    assert user_request(messages) == "look at this"
    assert user_request([{"role": "system", "content": "sys"}]) == ""


async def test_review_returns_feedback_and_usage():
    provider = FakeProvider(
        [
            {
                "text": "Solid, but tests were not run.",
                "usage": {"input_tokens": 100, "output_tokens": 20},
            }
        ]
    )
    outcome = await review(
        provider, "deepseek/deepseek-v4-flash", request="fix the bug", response="done"
    )
    assert outcome is not None
    assert outcome.feedback == "Solid, but tests were not run."
    assert outcome.model == "deepseek/deepseek-v4-flash"
    assert outcome.usage == {"input_tokens": 100, "output_tokens": 20}
    sent = provider.requests[0]["messages"]
    assert sent[0]["role"] == "system"
    assert "fix the bug" in sent[1]["content"] and "done" in sent[1]["content"]


async def test_review_fail_open():
    down = FakeProvider([{"error": ProviderError("down", status=500)}])
    assert await review(down, "m", request="q", response="a") is None
    blank = FakeProvider([{"text": "  "}])
    assert await review(blank, "m", request="q", response="a") is None
    assert await review(FakeProvider([]), "m", request="", response="a") is None
    assert await review(FakeProvider([]), "m", request="q", response="") is None


# -- runner integration ----------------------------------------------------------


async def test_runner_emits_review_when_pierre_enabled(tool_ctx, tmp_path):
    tool_ctx.config.pierre.enabled = True
    tool_ctx.config.pierre.model = "openai/gpt-5-mini"
    store = SessionStore(config_dir=tmp_path / "cfg")
    session = store.create("pierre", tmp_path, model=tool_ctx.config.llm.model)
    script = [
        {"text": "all fixed", "usage": {"input_tokens": 10, "output_tokens": 5}},
        # the pierre call
        {
            "text": "LGTM, verified against the request.",
            "usage": {"input_tokens": 100, "output_tokens": 20},
        },
    ]
    provider = FakeProvider(script)
    runner = AgentRunner(
        provider,
        ToolRegistry([]),
        tool_ctx,
        session=session,
        store=store,
        catalog=sample_catalog(),
    )
    events: list = []
    result = await runner.run([{"role": "user", "content": "fix it"}], on_event=events.append)

    assert result.review == "LGTM, verified against the request."
    reviews = [e for e in events if isinstance(e, Review)]
    assert len(reviews) == 1
    assert reviews[0].model == "openai/gpt-5-mini"
    # Review comes after Done.
    assert isinstance(events[events.index(reviews[0]) - 1], Done)
    # Pierre's tokens are folded into the run totals.
    assert result.usage_totals.input_tokens == 110
    assert result.usage_totals.output_tokens == 25
    # … and the review is persisted as a session event.
    pierre_events = [
        r for r in store.read_records(session) if isinstance(r, EventRecord) and r.kind == "pierre"
    ]
    assert len(pierre_events) == 1
    assert pierre_events[0].data["feedback"] == result.review
    assert pierre_events[0].data["model"] == "openai/gpt-5-mini"
    # … and session stats (the resume-restore source) include pierre's usage.
    from lecode.session.stats import session_stats

    stats = session_stats(store, session, catalog=sample_catalog())
    assert stats.input_tokens == 110
    assert stats.output_tokens == 25


async def test_runner_no_review_when_disabled(tool_ctx):
    runner = AgentRunner(FakeProvider([{"text": "done"}]), ToolRegistry([]), tool_ctx)
    events: list = []
    result = await runner.run([{"role": "user", "content": "hi"}], on_event=events.append)
    assert result.review is None
    assert not [e for e in events if isinstance(e, Review)]


async def test_runner_review_failure_is_silent(tool_ctx):
    tool_ctx.config.pierre.enabled = True
    script = [
        {"text": "done", "usage": {"input_tokens": 10, "output_tokens": 5}},
        {"error": ProviderError("reviewer down", status=500)},
    ]
    runner = AgentRunner(FakeProvider(script), ToolRegistry([]), tool_ctx)
    result = await runner.run([{"role": "user", "content": "hi"}])
    assert result.stop_reason == "done"
    assert result.review is None
    assert result.usage_totals.input_tokens == 10  # reviewer usage not folded


async def test_runner_skips_review_on_non_done_stop(tool_ctx):
    tool_ctx.config.pierre.enabled = True
    tool_ctx.config.agent.max_turns = 0
    runner = AgentRunner(FakeProvider([]), ToolRegistry([]), tool_ctx)
    result = await runner.run([{"role": "user", "content": "hi"}])
    assert result.stop_reason == "max_turns"
    assert result.review is None


# -- /pierre command ----------------------------------------------------------------


async def test_pierre_command_status_on_off_model(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/pierre")
    assert "pierre: off" in out.getvalue()
    await app.handle_command("/pierre model z-ai/glm-5.2")
    assert app.config.pierre.model == "z-ai/glm-5.2"
    await app.handle_command("/pierre on")
    assert app.config.pierre.enabled is True
    await app.handle_command("/pierre")
    text = out.getvalue()
    assert "pierre: on" in text and "z-ai/glm-5.2" in text
    await app.handle_command("/pierre off")
    assert app.config.pierre.enabled is False


async def test_pierre_on_refused_without_model(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/pierre on")
    assert app.config.pierre.enabled is False
    assert "/pierre model <id>" in out.getvalue()


async def test_pierre_on_refused_when_reviewer_is_main_model(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    app.config.pierre.model = app.config.llm.model
    await app.handle_command("/pierre on")
    assert app.config.pierre.enabled is False
    assert "must differ from the main model" in out.getvalue()


async def test_pierre_feedback_rendered_after_stats(tmp_path, monkeypatch):
    script = [
        {"text": "done", "usage": {"input_tokens": 10, "output_tokens": 5}},
        {"text": "Covers the request."},
    ]
    app, _, out = make_app(tmp_path, monkeypatch, script)
    app.config.pierre.enabled = True
    await app._submit("hi")
    await app._turn_task
    rendered = out.getvalue()
    assert "◆ pierre" in rendered
    assert "Covers the request." in rendered
    # stats line first, then the review
    assert rendered.index("answer:") < rendered.index("◆ pierre")
