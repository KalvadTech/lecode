"""Tests for session stats: counts, tokens, cost reporting."""

from __future__ import annotations

import pytest
from tests.fakes import sample_catalog

from lecode.session import SessionStore
from lecode.session.stats import session_stats

GPT5_MINI = "openai/gpt-5-mini"  # catalog pricing: 0.25 prompt / 2.0 completion per 1M


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("LECODE_SKILLS_DIR", str(tmp_path / "skills"))
    monkeypatch.chdir(tmp_path)
    return SessionStore()


@pytest.fixture
def session(store):
    s = store.create("stats", cwd="/tmp", model=GPT5_MINI)
    store.append_message(s, {"role": "user", "content": "hi"})
    store.append_message(
        s,
        {"role": "assistant", "content": "hello"},
        usage={"input_tokens": 1000, "output_tokens": 500, "cost_usd": 0.01},
    )
    store.append_message(
        s,
        {"role": "assistant", "content": "more"},
        usage={"input_tokens": 1_000_000, "output_tokens": 1_000_000},  # no recorded cost
    )
    return s


def test_counts_and_roles(store, session):
    stats = session_stats(store, session, catalog=sample_catalog())
    assert stats.message_count == 3
    assert stats.role_counts == {"user": 1, "assistant": 2}
    assert stats.input_tokens == 1_001_000
    assert stats.output_tokens == 1_000_500
    assert stats.usage_incomplete is False


def test_cost_combines_recorded_and_catalog_pricing(store, session):
    stats = session_stats(store, session, catalog=sample_catalog())
    # recorded 0.01 + catalog: (1M * 0.25 + 1M * 2.0) / 1M = 2.25
    assert stats.cost_usd == pytest.approx(2.26)


def test_cost_unknown_model_skipped(store):
    s = store.create("mystery", cwd="/tmp", model="no/such-model")
    store.append_message(
        s, {"role": "assistant", "content": "x"}, usage={"input_tokens": 5, "output_tokens": 5}
    )
    stats = session_stats(store, s, catalog=sample_catalog())
    assert stats.cost_usd == 0.0
    assert stats.input_tokens == 5
    assert stats.usage_incomplete


def test_openai_style_usage_keys(store):
    s = store.create("oa", cwd="/tmp", model=GPT5_MINI)
    store.append_message(
        s,
        {"role": "assistant", "content": "x"},
        usage={"prompt_tokens": 1_000_000, "completion_tokens": 0},
    )
    stats = session_stats(store, s, catalog=sample_catalog())
    assert stats.cost_usd == pytest.approx(0.25)


def test_timestamps_and_tombstones(store, session):
    assert session_stats(store, session).tombstone_count == 0
    store.undo(session)
    stats = session_stats(store, session)
    assert stats.tombstone_count == 1
    assert stats.created_at == session.meta.created_at
    assert stats.last_active is not None
    assert stats.last_active >= stats.created_at


def test_empty_session(store):
    s = store.create("empty", cwd="/tmp")
    stats = session_stats(store, s)
    assert stats.message_count == 0
    assert stats.last_active is None
    assert stats.cost_usd == 0.0


def test_context_tokens_from_last_assistant_usage(store, session):
    stats = session_stats(store, session)
    assert stats.context_tokens == 1_000_000  # last assistant turn's input


def test_context_tokens_fall_back_after_undo(store, session):
    store.append_message(session, {"role": "user", "content": "again"})
    store.append_message(
        session,
        {"role": "assistant", "content": "big"},
        usage={"input_tokens": 50_000, "output_tokens": 10},
    )
    assert session_stats(store, session).context_tokens == 50_000
    store.undo(session)  # hides the last user turn and its assistant reply
    assert session_stats(store, session).context_tokens == 1_000_000


def test_context_tokens_zero_without_usage(store):
    s = store.create("plain", cwd="/tmp")
    store.append_message(s, {"role": "assistant", "content": "no usage"})
    assert session_stats(store, s).context_tokens == 0


# -- worker usage ----------------------------------------------------------------


def test_worker_usage_summed_once(store, session):
    store.append_event(
        session,
        "worker_usage",
        {"input_tokens": 100, "output_tokens": 50, "cost_usd": 0.002},
    )
    store.append_event(
        session,
        "worker_usage",
        {"usage": {"input_tokens": 200, "output_tokens": 25, "cost_usd": 0.003}},
    )
    stats = session_stats(store, session, catalog=sample_catalog())
    assert stats.input_tokens == 1_001_000 + 300
    assert stats.output_tokens == 1_000_500 + 75
    assert stats.cost_usd == pytest.approx(2.26 + 0.005)


def test_worker_usage_includes_failed_and_cancelled_dispatches(store, session):
    store.append_event(
        session,
        "worker_usage",
        {"status": "failed", "input_tokens": 10, "output_tokens": 5},
    )
    store.append_event(
        session,
        "worker_usage",
        {"status": "cancelled", "input_tokens": 7, "output_tokens": 3},
    )
    stats = session_stats(store, session, catalog=sample_catalog())
    assert stats.input_tokens == 1_001_000 + 17
    assert stats.output_tokens == 1_000_500 + 8


def test_worker_usage_incomplete_flag(store, session):
    store.append_event(
        session, "worker_usage", {"input_tokens": 1, "output_tokens": 1, "cost_usd": 0}
    )
    assert session_stats(store, session, catalog=sample_catalog()).usage_incomplete is False
    store.append_event(
        session,
        "worker_usage",
        {"input_tokens": 1, "output_tokens": 1, "incomplete": True},
    )
    stats = session_stats(store, session)
    assert stats.usage_incomplete is True
    assert stats.input_tokens == 1_001_000 + 2


@pytest.mark.parametrize("kind", ["message", "pierre", "worker_usage"])
@pytest.mark.parametrize("nested", [False, True])
def test_incomplete_zero_usage_propagates(store, kind, nested):
    session = store.create("incomplete", cwd="/tmp", model=GPT5_MINI)
    usage = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0, "incomplete": True}
    if kind == "message":
        store.append_message(session, {"role": "assistant", "content": "done"}, usage=usage)
    else:
        data = {"usage": usage} if nested else usage
        store.append_event(session, kind, data)
    stats = session_stats(store, session, catalog=sample_catalog())
    assert stats.cost_usd == 0
    assert stats.usage_incomplete


@pytest.mark.parametrize("model", ["missing/model", "openai/gpt-5-", None])
@pytest.mark.parametrize("usage", [None, {"input_tokens": 0}, {"cost_usd": 0}])
def test_missing_usage_and_pricing_are_not_known_zero(store, model, usage):
    session = store.create("zero", cwd="/tmp", model=model)
    store.append_message(session, {"role": "assistant", "content": "done"}, usage=usage)
    stats = session_stats(store, session, catalog=sample_catalog())
    assert stats.cost_usd == 0
    assert stats.usage_incomplete is (usage != {"cost_usd": 0})


@pytest.mark.parametrize("incomplete", [False, True])
async def test_worker_live_usage_checkpoint_and_root_restore_count_once(
    tool_ctx, store, tmp_path, incomplete
):
    from lecode.agent.runner import LlmResponse
    from lecode.extras.workers import Worker, WorkerManager

    root = store.create("root", cwd=tmp_path)
    store.append_message(
        root,
        {"role": "assistant", "content": "root"},
        usage={"input_tokens": 10, "output_tokens": 1, "cost_usd": 0.5},
    )
    child = store.create("child", cwd=tmp_path)
    worker = Worker("child", None, 1, "explore", "delegated", "completed", child, tmp_path)
    worker.dispatch_id = "dispatch"
    manager = WorkerManager(
        tool_ctx.config, cwd=tmp_path, root_ctx=tool_ctx, store=store, session=root
    )
    manager._record(worker)
    stale_snapshot = store.load_events(root, "worker")[-1]
    await manager._event(worker, LlmResponse("model", 1, 5, 2, 0, usage_incomplete=incomplete))
    assert worker.usage_incomplete is incomplete
    assert worker.usage_totals.cost_usd == 0
    manager._record_usage(worker)
    await manager._event(worker, LlmResponse("model", 2, 7, 3, 0.25))
    assert worker.usage_incomplete is incomplete
    checkpoint = store.load_events(child, "worker_usage_checkpoint")[-1]
    assert checkpoint["usage_incomplete"] is incomplete
    # The child checkpoint is durable even if the last root snapshot was lost.
    store.append_event(root, "worker", stale_snapshot)
    await manager.shutdown()

    for _ in range(2):
        restored = WorkerManager(
            tool_ctx.config, cwd=tmp_path, root_ctx=tool_ctx, store=store, session=root
        )
        try:
            loaded = restored.load()[0]
            assert loaded.usage_totals == worker.usage_totals
            assert loaded.usage_incomplete is incomplete
            restored.load()
            stats = session_stats(store, root)
            assert (stats.input_tokens, stats.output_tokens, stats.context_tokens) == (22, 6, 10)
            assert stats.cost_usd == 0.75
            assert stats.usage_incomplete is incomplete
            assert len(store.load_events(root, "worker_usage")) == 2
        finally:
            await restored.shutdown()


@pytest.mark.parametrize("flag", [{}, {"incomplete": True}, {"usage_incomplete": True}])
async def test_worker_checkpoint_completeness_loads_old_and_new_json(
    tool_ctx, store, tmp_path, flag
):
    from lecode.extras.workers import Worker, WorkerManager

    root = store.create("root", cwd=tmp_path)
    child = store.create("child", cwd=tmp_path)
    worker = Worker("child", None, 1, "explore", "delegated", "completed", child, tmp_path)
    manager = WorkerManager(
        tool_ctx.config, cwd=tmp_path, root_ctx=tool_ctx, store=store, session=root
    )
    manager._record(worker)
    snapshot = store.load_events(root, "worker")[-1]
    snapshot["usage_totals"].pop("usage_incomplete")
    store.append_event(root, "worker", snapshot)
    store.append_event(
        child,
        "worker_usage_checkpoint",
        {"dispatch_id": None, "input_tokens": 2, "cost_usd": 0, **flag},
    )
    try:
        loaded = manager.load()[0]
        assert loaded.usage_incomplete is bool(flag)
        stats = session_stats(store, root)
        assert stats.usage_incomplete is bool(flag)
        assert stats.input_tokens == 2
        assert stats.cost_usd == 0
    finally:
        await manager.shutdown()
