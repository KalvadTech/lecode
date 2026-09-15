"""Phase 6 acceptance through compaction, runtime and provider seams; no network."""

import json

import pytest
from tests.fakes import FakeProvider

from lecode.agent.builder import build_runtime
from lecode.config.models import Config
from lecode.session.compaction import compact_session
from lecode.session.storage import SessionStore

PREFERENCE = "For this project, I prefer tabs for indentation."


def candidate(text=PREFERENCE, seqs=None, **updates):
    return {
        "text": text,
        "source_seqs": seqs or [1],
        "kind": "explicit_user_preference",
        "quote": text,
        "conflicts": [],
        "proposal": False,
        **updates,
    }


def extraction(*candidates):
    return {
        "text": json.dumps({"candidates": list(candidates)}),
        "usage": {"input_tokens": 30, "output_tokens": 10, "cost_usd": 0.02},
    }


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    config = Config()
    config.lsp.enabled = False
    config.memory.auto_learn = True
    store = SessionStore(tmp_path / "cfg")
    session = store.create("learn", tmp_path)
    runtime = build_runtime(config, tmp_path, session=session, store=store)
    for text in (PREFERENCE, "task one", "task two"):
        store.append_message(session, {"role": "user", "content": text})
        store.append_message(session, {"role": "assistant", "content": "Acknowledged."})
    yield runtime
    runtime.ctx.extras["facts"].close()


async def compact(runtime, provider):
    ctx = runtime.ctx
    return await compact_session(
        provider,
        ctx.session_store,
        ctx.session,
        "current-model",
        config=ctx.config,
        ctx=ctx,
    )


async def test_opt_in_compaction_promotes_exact_lasting_user_evidence(runtime):
    from lecode.session.stats import session_stats

    provider = FakeProvider(
        [
            {"text": "working summary", "usage": {"input_tokens": 70, "output_tokens": 20}},
            extraction(candidate()),
        ]
    )
    assert await compact(runtime, provider) == "working summary"
    ctx = runtime.ctx
    facts = ctx.extras["facts"].search("prefer")
    assert len(facts) == 1 and facts[0].text == PREFERENCE
    assert ctx.extras["facts"].source(facts[0].id).seqs == (1,)
    assert len(provider.requests) == 2
    request = provider.requests[1]
    assert request["model"] == "current-model"
    assert request["kwargs"]["reasoning_effort"] == ctx.config.llm.thinking
    supplied = json.loads(request["messages"][1]["content"])
    assert [m["seq"] for m in supplied["messages"]] == [1, 2]
    assert "working summary" not in request["messages"][1]["content"]
    stats = session_stats(ctx.session_store, ctx.session)
    assert (stats.input_tokens, stats.output_tokens) == (100, 30)
    assert stats.cost_usd == pytest.approx(0.02)


async def test_runner_auto_compaction_uses_live_context_and_counts_learning_once(runtime):
    from lecode.agent.builder import refresh_system_prompt
    from lecode.agent.runner import AgentRunner
    from lecode.session.stats import session_stats

    ctx = runtime.ctx
    ctx.config.compaction.mid_turn_threshold = 1
    provider = FakeProvider(
        [
            {
                "tool_calls": [{"id": "read", "name": "memory_read", "arguments": "{}"}],
                "usage": {"input_tokens": 10, "output_tokens": 2},
            },
            {"text": "summary", "usage": {"input_tokens": 70, "output_tokens": 20}},
            extraction(candidate()),
            {"text": "done", "usage": {"input_tokens": 15, "output_tokens": 3}},
        ]
    )
    runner = AgentRunner(
        provider,
        runtime.registry,
        ctx,
        session=ctx.session,
        store=ctx.session_store,
        refresh_prompt=lambda: refresh_system_prompt(runtime),
    )
    result = await runner.run(ctx.session_store.load_for_model(ctx.session))
    assert result.final_text == "done"
    assert len(ctx.extras["facts"].search("prefer")) == 1
    assert (result.usage_totals.input_tokens, result.usage_totals.output_tokens) == (125, 35)
    stats = session_stats(ctx.session_store, ctx.session)
    assert (stats.input_tokens, stats.output_tokens) == (125, 35)


async def test_conflicts_are_visible_proposals_never_silent_revisions(runtime):
    ctx = runtime.ctx
    ref = ctx.session_store.source_snapshot(ctx.session.id, 3, 3, project_root=ctx.cwd).ref
    incumbent = ctx.extras["facts"].remember(
        "For this project, I prefer spaces for indentation.",
        ref,
        sessions=ctx.session_store,
        project_root=ctx.cwd,
    )
    provider = FakeProvider([{"text": "summary"}, extraction(candidate(conflicts=[incumbent.id]))])
    assert await compact(runtime, provider) == "summary"
    assert ctx.extras["facts"].get(incumbent.id) == incumbent
    assert len(ctx.extras["facts"].search("prefer")) == 1
    payload = json.loads(provider.requests[-1]["messages"][1]["content"])
    assert payload["existing_facts"][0]["id"] == incumbent.id
    event = ctx.session_store.read_records(ctx.session)[-1]
    assert event.data["proposals"][0]["conflicts"] == [incumbent.id]
    assert event.data["proposals"][0]["text"] == PREFERENCE


@pytest.mark.parametrize(
    "reply",
    [
        extraction(candidate(text="task one", seqs=[3])),
        extraction(candidate(quote="not in the source")),
        extraction(candidate(seqs=[5])),
        extraction(candidate(seqs=[True])),
        extraction(candidate(conflicts=["0" * 64])),
        extraction(candidate(extra="not allowed")),
        extraction(candidate(text="x" * 513)),
        extraction(*[candidate()] * 5),
        {"text": "not JSON"},
        {"text": '{"candidates":[],"candidates":' + json.dumps([candidate()]) + "}"},
        {"text": json.dumps({"candidates": [candidate()]}), "finish_reason": "length"},
    ],
)
async def test_untrusted_or_malformed_output_cannot_promote_and_keeps_summary(runtime, reply):
    provider = FakeProvider([{"text": "usable summary"}, reply])
    assert await compact(runtime, provider) == "usable summary"
    ctx = runtime.ctx
    assert ctx.extras["facts"].search("prefer") == []
    assert ctx.session_store.load_for_model(ctx.session)[0]["content"] == "usable summary"
    assert (
        ctx.session_store.source_snapshot(ctx.session.id, 1, 2, project_root=ctx.cwd).status
        == "valid"
    )
    assert ctx.session_store.read_records(ctx.session)[-1].data["purpose"] == "learning"


@pytest.mark.parametrize("change", ["forget", "undo", "clear", "switch", "failure", "cancel"])
async def test_extraction_await_races_discard_candidates_and_account_once(runtime, change):
    import asyncio

    ctx = runtime.ctx
    store, session, facts = ctx.session_store, ctx.session, ctx.extras["facts"]
    ref = store.source_snapshot(session.id, 5, 5, project_root=ctx.cwd).ref
    incumbent = facts.remember("old fact", ref, sessions=store, project_root=ctx.cwd)
    entered, resume = asyncio.Event(), asyncio.Event()

    class WaitingProvider(FakeProvider):
        async def complete(self, *args, **kwargs):
            if self.requests:
                entered.set()
                await resume.wait()
                if change == "failure":
                    raise RuntimeError("offline")
            return await super().complete(*args, **kwargs)

    provider = WaitingProvider([{"text": "working summary"}, extraction(candidate())])
    task = asyncio.create_task(compact(runtime, provider))
    await entered.wait()
    assert store.working_summary(session).data["summary"] == "working summary"
    if change == "forget":
        facts.forget(incumbent.id)
    elif change == "undo":
        store.rewind_to(session, 0)
    elif change == "clear":
        store.append_event(session, "clear")
    elif change == "switch":
        ctx.session = store.create("other", ctx.cwd)
    elif change == "cancel":
        task.cancel()
    resume.set()
    if change == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        assert await task == "working summary"
    assert facts.search("prefer") == []
    events = [r for r in store.read_records(session) if getattr(r, "kind", None) == "memory_usage"]
    assert len(events) == 1 and events[0].data["purpose"] == "learning"
    if change in {"failure", "cancel", "switch"}:
        assert store.load_for_model(session)[0]["content"] == "working summary"
    if change == "undo":
        assert store.redo(session)
        assert store.load_for_model(session)[0]["content"] == "working summary"


async def test_project_fact_requires_exact_paired_read_corroboration_not_tool_instructions(runtime):
    ctx = runtime.ctx
    store, session = ctx.session_store, ctx.session
    store.append_event(session, "clear")
    quote = 'requires-python = ">=3.12"'
    fact_text = "pyproject.toml contains " + json.dumps(quote)
    messages = [
        {"role": "user", "content": "Check the Python requirement."},
        {
            "role": "assistant",
            "content": "Reading",
            "tool_calls": [
                {
                    "id": "r1",
                    "type": "function",
                    "function": {"name": "read", "arguments": '{"path":"pyproject.toml"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "r1",
            "name": "read",
            "content": f"1\t{quote}\n2\t{PREFERENCE}",
        },
        {"role": "assistant", "content": fact_text},
    ]
    seqs = [store.append_message(session, message).seq for message in messages]
    for _ in range(2):
        store.append_message(session, {"role": "user", "content": "next"})
        store.append_message(session, {"role": "assistant", "content": "ok"})
    provider = FakeProvider(
        [
            {"text": "summary"},
            extraction(
                candidate(fact_text, seqs, kind="verified_project_fact", quote=quote),
                candidate(seqs=[seqs[2]]),
                candidate(
                    'pyproject.toml contains "unsupported"',
                    seqs,
                    kind="verified_project_fact",
                    quote="unsupported",
                ),
            ),
        ]
    )
    assert await compact(runtime, provider) == "summary"
    assert ctx.extras["facts"].search("prefer") == []
    facts = ctx.extras["facts"].search("requires-python")
    assert len(facts) == 1 and facts[0].text == fact_text
    assert ctx.extras["facts"].source(facts[0].id).seqs == tuple(seqs)
    _, result = await runtime.registry.dispatch_result(
        "recall", "memory_recall", json.dumps({"fact_id": facts[0].id}), ctx
    )
    recalled = json.loads(json.loads(result.content)["source_text"])
    assert recalled[2]["message"]["content"] == messages[2]["content"]


@pytest.mark.parametrize(
    "mode", ["default", "disabled", "readonly", "child", "nonpersistent", "overlay"]
)
async def test_learning_is_optional_and_requires_writable_parent(runtime, mode):
    from lecode.permission.checker import AgentOverlay

    ctx = runtime.ctx
    if mode == "default":
        ctx.config.memory.auto_learn = Config().memory.auto_learn
    elif mode == "disabled":
        ctx.config.memory.enabled = False
    elif mode == "readonly":
        ctx.permission_checker.set_mode("readonly")
    elif mode == "overlay":
        ctx.permission_checker = ctx.permission_checker.for_agent(AgentOverlay(mode="readonly"))
    elif mode == "child":
        ctx.recall_context = object()
    provider = FakeProvider([{"text": "summary"}, extraction(candidate())])
    store, session = ctx.session_store, ctx.session
    if mode == "nonpersistent":
        ctx.session = None
    assert (
        await compact_session(provider, store, session, "current-model", config=ctx.config, ctx=ctx)
        == "summary"
    )
    assert len(provider.requests) == 1
    assert ctx.extras["facts"].search("prefer") == []


async def test_six_learning_boundaries_are_incremental_and_new_source_duplicates_do_not_readd(
    runtime,
):
    ctx = runtime.ctx
    seen = set()
    for index in range(6):
        visible = ctx.session_store.visible_messages(ctx.session)
        previous = ctx.session_store.working_summary(ctx.session)
        fresh = [m for m in visible if previous is None or m.seq >= previous.data["keep_from_seq"]]
        preference = next((m for m in fresh[:-4] if m.message.get("content") == PREFERENCE), None)
        provider = FakeProvider(
            [
                {"text": f"summary-{index}"},
                extraction(candidate(seqs=[preference.seq])) if preference else extraction(),
            ]
        )
        assert await compact(runtime, provider) == f"summary-{index}"
        payload = json.loads(provider.requests[-1]["messages"][1]["content"])
        seqs = {m["seq"] for m in payload["messages"]}
        assert seqs and not seqs.intersection(seen)
        seen.update(seqs)
        for _ in range(2):
            ctx.session_store.append_message(ctx.session, {"role": "user", "content": PREFERENCE})
            ctx.session_store.append_message(ctx.session, {"role": "assistant", "content": "ok"})
    assert len(ctx.extras["facts"].search("prefer")) == 1
    assert ctx.session_store.load_for_model(ctx.session)[0]["content"] == "summary-5"


async def test_explicit_correction_is_proposed_even_when_model_requests_promotion(runtime):
    ctx = runtime.ctx
    ctx.session_store.append_event(ctx.session, "clear")
    text = "Correction: For this project, I prefer spaces instead of tabs."
    record = ctx.session_store.append_message(ctx.session, {"role": "user", "content": text})
    ctx.session_store.append_message(ctx.session, {"role": "assistant", "content": "noted"})
    for _ in range(2):
        ctx.session_store.append_message(ctx.session, {"role": "user", "content": "next"})
        ctx.session_store.append_message(ctx.session, {"role": "assistant", "content": "ok"})
    assert (
        await compact(
            runtime, FakeProvider([{"text": "summary"}, extraction(candidate(text, [record.seq]))])
        )
        == "summary"
    )
    assert ctx.extras["facts"].search("prefer") == []
    assert ctx.session_store.read_records(ctx.session)[-1].data["proposals"][0]["text"] == text


async def test_explicitly_temporary_preference_is_not_auto_promoted(runtime):
    ctx = runtime.ctx
    ctx.session_store.append_event(ctx.session, "clear")
    text = "For this project, I prefer tabs for this task only."
    record = ctx.session_store.append_message(ctx.session, {"role": "user", "content": text})
    ctx.session_store.append_message(ctx.session, {"role": "assistant", "content": "ok"})
    for _ in range(2):
        ctx.session_store.append_message(ctx.session, {"role": "user", "content": "next"})
        ctx.session_store.append_message(ctx.session, {"role": "assistant", "content": "ok"})
    await compact(
        runtime, FakeProvider([{"text": "summary"}, extraction(candidate(text, [record.seq]))])
    )
    assert ctx.extras["facts"].search("prefer") == []


async def test_learning_uses_current_catalog_headroom_and_prices_rejected_output(runtime):
    from lecode.providers.catalog import Catalog, ModelInfo
    from lecode.session.compaction import estimate_request
    from lecode.session.stats import session_stats

    ctx = runtime.ctx
    catalog = Catalog(
        [
            ModelInfo.model_validate(
                {
                    "id": "current-small",
                    "name": "small",
                    "context_window": 4000,
                    "max_output": 256,
                    "pricing": {"prompt": 2, "completion": 4},
                    "modalities": {"input": ["text"], "output": ["text"]},
                }
            )
        ]
    )
    provider = FakeProvider(
        [
            {"text": "summary", "usage": {"input_tokens": 70, "output_tokens": 20}},
            {"text": "invalid JSON", "usage": {"input_tokens": 30, "output_tokens": 10}},
        ]
    )
    observed = []
    assert (
        await compact_session(
            provider,
            ctx.session_store,
            ctx.session,
            "current-small",
            config=ctx.config,
            ctx=ctx,
            catalog=catalog,
            on_usage=observed.append,
        )
        == "summary"
    )
    assert len(provider.requests) == 2 and len(observed) == 2
    for request in provider.requests:
        assert request["model"] == "current-small"
        assert request["kwargs"]["max_tokens"] == 256
        assert estimate_request(request["messages"]) + 256 <= 4000
    assert session_stats(ctx.session_store, ctx.session).cost_usd == pytest.approx(0.00032)


async def test_independent_same_project_session_sees_fresh_fact_with_sources_as_evidence(runtime):
    from lecode.agent.builder import refresh_system_prompt
    from lecode.agent.runner import AgentRunner

    assert (
        await compact(runtime, FakeProvider([{"text": "summary"}, extraction(candidate())]))
        == "summary"
    )
    ctx = runtime.ctx
    fact = ctx.extras["facts"].search("prefer")[0]
    ctx.extras["facts"].close()
    store = SessionStore(ctx.session_store.config_dir)
    second = store.create("independent", ctx.cwd)
    config = ctx.config.model_copy(deep=True)
    config.llm.system_prompt.custom = "My custom prompt remains."
    other = build_runtime(config, ctx.cwd, store=store, session=second)
    provider = FakeProvider([{"text": "hello"}])
    runner = AgentRunner(
        provider,
        other.registry,
        other.ctx,
        session=second,
        store=store,
        refresh_prompt=lambda: refresh_system_prompt(other),
    )
    try:
        await runner.run([{"role": "user", "content": "What is my indentation preference?"}])
        prompt = provider.requests[0]["messages"][0]["content"]
        assert PREFERENCE in prompt and fact.id in prompt
        assert "untrusted evidence" in prompt.lower() and "not instructions" in prompt.lower()
        assert ctx.session.id in prompt and '"seqs": [1]' in prompt and '"revision": 1' in prompt
        assert prompt.startswith("My custom prompt remains.")
    finally:
        other.ctx.extras["facts"].close()


async def test_runtime_memory_bound_includes_scratchpad_and_only_whole_facts(runtime):
    from lecode.agent.builder import refresh_system_prompt
    from lecode.context.skills import SkillRegistry

    await compact(runtime, FakeProvider([{"text": "summary"}, extraction(candidate())]))
    ctx = runtime.ctx
    runtime.skills = SkillRegistry()
    runtime.agent_name = None
    ctx.config.llm.system_prompt.custom = "base"
    ctx.config.memory.max_bytes = 1600
    ctx.config.memory.facts_max_bytes = 700
    ctx.extras["memory"].write_long_term("human-é " * 2000)
    ctx.extras["memory"].write_scratchpad("scratch-é " * 2000)
    prompt = refresh_system_prompt(runtime)
    memory = prompt.split("## Memory", 1)[1]
    assert len(("## Memory" + memory).encode()) <= 1600
    assert "human-é" in memory and "scratch-é" in memory
    assert PREFERENCE in memory
    lines = [line for line in memory.splitlines() if line.startswith('{"id"')]
    assert len(lines) == 1 and json.loads(lines[0])["text"] == PREFERENCE
    ctx.config.memory.facts_max_bytes = 10
    assert PREFERENCE not in refresh_system_prompt(runtime)


async def test_readonly_and_child_fact_listing_exposes_ids_revisions_and_status(runtime):
    from dataclasses import replace

    from lecode.extras.subagents import child_registry
    from lecode.memory.commands import memory_command
    from lecode.memory.recall import RecallContext

    ctx = runtime.ctx
    await compact(runtime, FakeProvider([{"text": "summary"}, extraction(candidate())]))
    fact = ctx.extras["facts"].search("prefer")[0]
    recall = RecallContext(ctx.session_store, ctx.extras["facts"], ctx.cwd)
    ctx.permission_checker.set_mode("readonly")
    child = replace(ctx, session=None, session_store=None, recall_context=recall, extras={})
    _, result = await child_registry(runtime.registry).dispatch_result(
        "list", "memory_list", "{}", child
    )
    assert not result.is_error
    data = json.loads(result.content)
    assert data["facts"][0]["id"] == fact.id
    assert data["facts"][0]["status"] == "valid"
    assert data["facts"][0]["sources"][0]["seqs"] == [1]
    assert "memory_correct" not in child_registry(runtime.registry).names()
    assert json.loads(memory_command(["facts"], ctx.extras["memory"], recall=recall)) == data
    ctx.session_store.rewind_to(ctx.session, 0)
    _, result = await runtime.registry.dispatch_result("list2", "memory_list", "{}", ctx)
    hidden = json.loads(result.content)["facts"][0]
    assert hidden["status"] == "hidden" and "text" not in hidden


async def test_fact_inspection_surfaces_proposals_without_replaying_hidden_text(runtime):
    ctx = runtime.ctx
    await compact(
        runtime, FakeProvider([{"text": "summary"}, extraction(candidate(proposal=True))])
    )
    _, result = await runtime.registry.dispatch_result("list", "memory_list", "{}", ctx)
    data = json.loads(result.content)
    assert data["learning"]["status"] == "proposed"
    assert data["learning"]["proposals"][0]["text"] == PREFERENCE
    assert data["facts"] == []
    ctx.session_store.rewind_to(ctx.session, 0)
    _, result = await runtime.registry.dispatch_result("list2", "memory_list", "{}", ctx)
    assert PREFERENCE not in result.content


@pytest.mark.parametrize(
    "change", ["clear", "undo-redo", "missing", "corrupt", "forget", "correct"]
)
async def test_refresh_excludes_invalid_evidence_and_preserves_durable_clear_semantics(
    runtime, change
):
    from lecode.agent.builder import refresh_system_prompt

    ctx = runtime.ctx
    store, session, facts = ctx.session_store, ctx.session, ctx.extras["facts"]
    await compact(runtime, FakeProvider([{"text": "summary"}, extraction(candidate())]))
    fact = facts.search("prefer")[0]
    facts.add("unverified fact must not inject", source_id="missing", source_seq=1)
    assert "unverified fact" not in refresh_system_prompt(runtime)
    if change == "clear":
        store.append_event(session, "clear")
        assert PREFERENCE in refresh_system_prompt(runtime)
    elif change == "undo-redo":
        store.rewind_to(session, 0)
        assert PREFERENCE not in refresh_system_prompt(runtime)
        assert store.redo(session)
        assert PREFERENCE in refresh_system_prompt(runtime)
    elif change == "missing":
        store.delete(session.id)
        assert PREFERENCE not in refresh_system_prompt(runtime)
    elif change == "corrupt":
        with session.path.open("a") as f:
            f.write("corrupt source line\n")
        assert PREFERENCE not in refresh_system_prompt(runtime)
    elif change == "forget":
        facts.forget(fact.id)
        assert PREFERENCE not in refresh_system_prompt(runtime)
    else:
        revised = "For this project, I prefer spaces."
        record = store.append_message(session, {"role": "user", "content": revised})
        ref = store.source_snapshot(session.id, record.seq, record.seq, project_root=ctx.cwd).ref
        facts.correct(
            fact.id, revised, expected_revision=1, ref=ref, sessions=store, project_root=ctx.cwd
        )
        prompt = refresh_system_prompt(runtime)
        assert PREFERENCE not in prompt and revised in prompt and '"revision": 2' in prompt


async def test_runtime_close_releases_fact_database_wal(runtime):
    from pathlib import Path

    await compact(runtime, FakeProvider([{"text": "summary"}, extraction(candidate())]))
    facts = runtime.ctx.extras["facts"]
    wal = Path(str(facts.path) + "-wal")
    assert wal.exists()
    runtime.close()
    assert not wal.exists()


async def test_runner_refreshes_corrected_facts_between_tool_rounds(runtime):
    from dataclasses import asdict

    from lecode.agent.builder import refresh_system_prompt
    from lecode.agent.runner import AgentRunner

    ctx = runtime.ctx
    await compact(runtime, FakeProvider([{"text": "summary"}, extraction(candidate())]))
    fact = ctx.extras["facts"].search("prefer")[0]
    ref = ctx.session_store.source_snapshot(ctx.session.id, 3, 3, project_root=ctx.cwd).ref
    corrected = "For this project, I prefer spaces."
    ctx.config.compaction.enabled = False
    provider = FakeProvider(
        [
            {
                "tool_calls": [
                    {
                        "id": "c",
                        "name": "memory_correct",
                        "arguments": json.dumps(
                            {
                                "fact_id": fact.id,
                                "expected_revision": 1,
                                "text": corrected,
                                "source_snapshot": asdict(ref),
                            }
                        ),
                    }
                ]
            },
            {"text": "done"},
        ]
    )
    runner = AgentRunner(
        provider,
        runtime.registry,
        ctx,
        session=ctx.session,
        store=ctx.session_store,
        refresh_prompt=lambda: refresh_system_prompt(runtime),
    )
    assert (await runner.run(ctx.session_store.load_for_model(ctx.session))).final_text == "done"
    first, second = [request["messages"][0]["content"] for request in provider.requests]
    assert PREFERENCE in first and PREFERENCE not in second
    assert corrected in second


async def test_learning_comparison_does_not_include_orphaned_prior_revisions(runtime):
    ctx = runtime.ctx
    store, facts = ctx.session_store, ctx.extras["facts"]
    await compact(runtime, FakeProvider([{"text": "summary"}, extraction(candidate())]))
    fact = facts.search("prefer")[0]
    original = ctx.session
    ctx.session = store.create("correction", ctx.cwd)
    record = store.append_message(ctx.session, {"role": "user", "content": "new evidence"})
    store.append_message(ctx.session, {"role": "assistant", "content": "ok"})
    ref = store.source_snapshot(ctx.session.id, record.seq, record.seq, project_root=ctx.cwd).ref
    facts.correct(
        fact.id, "revised fact", expected_revision=1, ref=ref, sessions=store, project_root=ctx.cwd
    )
    store.delete(original.id)
    for _ in range(2):
        store.append_message(ctx.session, {"role": "user", "content": "next"})
        store.append_message(ctx.session, {"role": "assistant", "content": "ok"})
    provider = FakeProvider([{"text": "summary"}, extraction()])
    await compact(runtime, provider)
    assert json.loads(provider.requests[-1]["messages"][1]["content"])["existing_facts"] == []


async def test_optional_learning_preparation_failure_cannot_fail_a_working_summary(
    runtime, monkeypatch
):
    def unavailable():
        raise OSError("fact inspection unavailable")

    monkeypatch.setattr(runtime.ctx.extras["facts"], "list", unavailable)
    provider = FakeProvider([{"text": "working summary"}])
    assert await compact(runtime, provider) == "working summary"
    assert len(provider.requests) == 1
    ctx = runtime.ctx
    assert ctx.session_store.load_for_model(ctx.session)[0]["content"] == "working summary"
