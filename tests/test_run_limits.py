"""Shared cost admission, including concurrent and auxiliary model calls."""

import asyncio

import pytest
from tests.fakes import FakeProvider, sample_catalog

from lecode.providers.budget import BudgetedProvider, CostLimitError
from lecode.providers.catalog import Catalog, Pricing
from lecode.providers.openai_compat import ProviderError
from lecode.providers.types import TokenDelta, Usage, collect

MODEL = "deepseek/deepseek-v4-flash"
PROMPT = [{"role": "user", "content": "hello"}]


async def test_stream_and_auxiliary_completions_share_one_ledger():
    provider = FakeProvider(
        [
            {"text": "main", "usage": {"cost_usd": 0.2}},
            {"text": "summary", "usage": {"cost_usd": 0.3}},
        ]
    )
    budget = BudgetedProvider(provider, sample_catalog(), 0.5)
    assert (await collect(budget.stream_chat(PROMPT, MODEL))).content == "main"
    assert (await budget.complete(PROMPT, MODEL)).content == "summary"
    with pytest.raises(CostLimitError, match="reached"):
        await budget.complete(PROMPT, MODEL)
    assert len(provider.requests) == 2
    assert budget.cost_usd == pytest.approx(0.5)


@pytest.mark.parametrize("model", ["missing/model", "openai/gpt-5-"])
async def test_unknown_or_ambiguous_model_never_reaches_provider(model):
    provider = FakeProvider([{"usage": {"cost_usd": 0}}])
    budget = BudgetedProvider(provider, sample_catalog(), 1)
    with pytest.raises(CostLimitError, match="price is unknown"):
        await budget.complete(PROMPT, model)
    assert not provider.requests


@pytest.mark.parametrize(
    "usage",
    [
        None,
        True,
        ["invalid"],
        "invalid",
        {"input_tokens": 1},
        {"cost_usd": -1},
        {"cost_usd": float("nan")},
        {"cost_usd": float("inf")},
        {"cost_usd": "bad"},
        {"cost_usd": False},
        {"input_tokens": float("nan"), "output_tokens": 1},
        {"input_tokens": -1, "output_tokens": 1},
        {"input_tokens": False, "output_tokens": 1},
        {"input_tokens": 0.5, "output_tokens": 1},
        {"cost_usd": 0, "incomplete": True},
    ],
)
async def test_unknown_or_invalid_spend_closes_admission(usage):
    provider = FakeProvider([{"text": "answer", "usage": usage}])
    budget = BudgetedProvider(provider, sample_catalog(), 1)
    with pytest.raises(CostLimitError, match="spend is unknown"):
        await budget.complete(PROMPT, MODEL)
    with pytest.raises(CostLimitError):
        await budget.complete(PROMPT, MODEL)
    assert len(provider.requests) == 1
    assert budget.usage_incomplete


@pytest.mark.parametrize(
    "pricing",
    [
        Pricing(prompt=0, completion=0, known=False),
        Pricing(prompt=-1, completion=0),
        Pricing(prompt=float("nan"), completion=0),
        Pricing(prompt=0, completion=float("inf")),
    ],
)
async def test_invalid_or_unknown_catalog_price_is_rejected(pricing):
    entry = sample_catalog().get(MODEL).model_copy(update={"pricing": pricing})
    provider = FakeProvider([{"usage": {"cost_usd": 0}}])
    budget = BudgetedProvider(provider, Catalog([entry]), 1)
    with pytest.raises(CostLimitError, match="price is unknown"):
        await budget.complete(PROMPT, MODEL)
    assert not provider.requests


async def test_known_free_models_and_token_estimates_remain_valid():
    entry = (
        sample_catalog().get(MODEL).model_copy(update={"pricing": Pricing(prompt=0, completion=0)})
    )
    provider = FakeProvider(
        [
            {"text": "free", "usage": {"input_tokens": 10, "output_tokens": 5}},
            {"text": "free again", "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
        ]
    )
    budget = BudgetedProvider(provider, Catalog([entry]), 0.01)
    await budget.complete(PROMPT, MODEL)
    await budget.complete(PROMPT, MODEL)
    assert budget.cost_usd == 0 and budget.error is None
    assert (budget.input_tokens, budget.output_tokens) == (20, 10)


async def test_accounting_uses_catalog_when_provider_omits_billed_cost():
    provider = FakeProvider([{"usage": {"input_tokens": 1_000_000, "output_tokens": 1_000_000}}])
    budget = BudgetedProvider(provider, sample_catalog(), 1)
    await budget.complete(PROMPT, MODEL)
    assert budget.cost_usd == pytest.approx(0.27)


async def test_admitted_concurrent_calls_can_finish_but_new_calls_stop():
    started = asyncio.Queue()
    finish = asyncio.Event()

    class ConcurrentProvider(FakeProvider):
        async def _stream(self, entry):
            started.put_nowait(None)
            await finish.wait()
            yield Usage({"cost_usd": 0.6})

    provider = ConcurrentProvider([{}, {}])
    budget = BudgetedProvider(provider, sample_catalog(), 0.5)
    calls = [asyncio.create_task(budget.complete(PROMPT, MODEL)) for _ in range(2)]
    await asyncio.wait_for(started.get(), 1)
    await asyncio.wait_for(started.get(), 1)
    finish.set()
    await asyncio.gather(*calls)
    assert budget.cost_usd == pytest.approx(1.2)
    with pytest.raises(CostLimitError):
        await budget.complete(PROMPT, MODEL)
    assert len(provider.requests) == 2


async def test_cancelled_or_failed_stream_with_missing_usage_blocks_retry():
    class FailingProvider(FakeProvider):
        async def _stream(self, entry):
            yield TokenDelta("partial")
            raise ProviderError("connection lost", retryable=True)

    provider = FailingProvider([{}])
    budget = BudgetedProvider(provider, sample_catalog(), 1)
    with pytest.raises(CostLimitError, match="spend is unknown") as error:
        await budget.complete(PROMPT, MODEL)
    assert isinstance(error.value.__cause__, ProviderError)
    with pytest.raises(CostLimitError, match="spend is unknown"):
        await budget.complete(PROMPT, MODEL)
    assert len(provider.requests) == 1


@pytest.mark.parametrize("cancelled", [False, True])
async def test_concurrent_runner_preserves_valid_usage_after_sibling_failure(
    tool_ctx, tmp_path, monkeypatch, cancelled
):
    from lecode.agent.runner import AgentRunner
    from lecode.agent.tools.base import ToolContext, ToolRegistry
    from lecode.session.stats import session_stats
    from lecode.session.storage import SessionStore

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LECODE_SKILLS_DIR", str(tmp_path / "skills"))
    started = asyncio.Queue()
    releases = [asyncio.Event(), asyncio.Event()]
    valid_usage_sent = asyncio.Event()
    usage = {"input_tokens": 10, "output_tokens": 2, "cost_usd": 0.2}

    class ConcurrentProvider(FakeProvider):
        async def _stream(self, entry):
            started.put_nowait(None)
            await releases[entry["index"]].wait()
            yield TokenDelta("valid response" if entry["index"] else "bad usage")
            yield Usage(usage if entry["index"] else {"cost_usd": "invalid"})
            if entry["index"] and cancelled:
                valid_usage_sent.set()
                await asyncio.Event().wait()

    provider = ConcurrentProvider([{"index": 0}, {"index": 1}])
    budget = BudgetedProvider(provider, sample_catalog(), 1)
    store = SessionStore(config_dir=tmp_path / "cfg")
    sessions = [store.create(f"concurrent-{i}", tmp_path, model=MODEL) for i in range(2)]
    runners = [
        AgentRunner(
            budget,
            ToolRegistry(),
            ToolContext(
                cwd=tmp_path,
                config=tool_ctx.config,
                permission_checker=tool_ctx.permission_checker,
                auto_approve=True,
            ),
            store=store,
            session=session,
            catalog=sample_catalog(),
        )
        for session in sessions
    ]
    tasks = [asyncio.create_task(runner.run(PROMPT)) for runner in runners]
    await asyncio.wait_for(started.get(), 1)
    await asyncio.wait_for(started.get(), 1)
    releases[0].set()
    assert (await tasks[0]).stop_reason == "cost_limit"
    releases[1].set()
    if cancelled:
        await asyncio.wait_for(valid_usage_sent.wait(), 1)
        tasks[1].cancel()
        with pytest.raises(asyncio.CancelledError):
            await tasks[1]
    else:
        await tasks[1]

    recorded = store.load_messages(sessions[1])
    assert recorded[0].message["content"] == "valid response"
    assert recorded[0].usage == usage
    stats = session_stats(store, sessions[1])
    assert (stats.input_tokens, stats.output_tokens, stats.cost_usd) == (10, 2, 0.2)
    assert not stats.usage_incomplete
    assert budget.cost_usd == 0.2
    with pytest.raises(CostLimitError):
        await budget.complete(PROMPT, MODEL)
    assert len(provider.requests) == 2


async def test_latest_usage_snapshot_is_counted_once():
    class SnapshotsProvider(FakeProvider):
        async def _stream(self, entry):
            yield Usage({"input_tokens": 10})
            yield Usage({"input_tokens": 10, "output_tokens": 5, "cost_usd": 0.2})

    budget = BudgetedProvider(SnapshotsProvider([{}]), sample_catalog(), 1)
    await budget.complete(PROMPT, MODEL)
    assert budget.cost_usd == 0.2
    assert budget.input_tokens == 10 and budget.output_tokens == 5
