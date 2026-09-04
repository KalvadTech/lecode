"""Tests for retry_async: classification, backoff, on_retry hook."""

from __future__ import annotations

import pytest

from lecode.providers import retry as retry_mod
from lecode.providers.openai_compat import ProviderError
from lecode.providers.retry import retry_async


@pytest.fixture
def fast_sleep(monkeypatch):
    """Deterministic jitter + recorded sleeps (no real waiting)."""
    monkeypatch.setattr(retry_mod, "uniform", lambda lo, hi: hi)
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(retry_mod, "sleep", fake_sleep)
    return sleeps


async def test_success_first_try(fast_sleep):
    calls = 0

    async def fn() -> str:
        nonlocal calls
        calls += 1
        return "ok"

    assert await retry_async(fn) == "ok"
    assert calls == 1
    assert fast_sleep == []


async def test_retryable_error_retries_with_backoff(fast_sleep):
    calls = 0
    retries: list[tuple[int, ProviderError, float]] = []

    async def fn() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ProviderError("rate limited", status=429, retryable=True)
        return "ok"

    result = await retry_async(fn, on_retry=lambda a, e, d: retries.append((a, e, d)))
    assert result == "ok"
    assert calls == 3
    assert [r[0] for r in retries] == [1, 2]
    # Full jitter pinned at the cap: base_delay * 2^(attempt-1).
    assert [r[2] for r in retries] == [0.5, 1.0]
    assert fast_sleep == [0.5, 1.0]


async def test_delay_capped_at_max_delay(fast_sleep):
    calls = 0

    async def fn() -> str:
        nonlocal calls
        calls += 1
        raise ProviderError("still down", status=503, retryable=True)

    with pytest.raises(ProviderError):
        await retry_async(fn, max_attempts=5, base_delay=10.0, max_delay=15.0)
    assert calls == 5
    assert fast_sleep == [10.0, 15.0, 15.0, 15.0]


async def test_non_retryable_error_raises_immediately(fast_sleep):
    calls = 0

    async def fn() -> None:
        nonlocal calls
        calls += 1
        raise ProviderError("bad request", status=400, retryable=False)

    with pytest.raises(ProviderError, match="bad request"):
        await retry_async(fn)
    assert calls == 1
    assert fast_sleep == []


async def test_max_attempts_exhausted_raises_last_error(fast_sleep):
    calls = 0

    async def fn() -> None:
        nonlocal calls
        calls += 1
        raise ProviderError(f"fail {calls}", status=500, retryable=True)

    with pytest.raises(ProviderError, match="fail 3"):
        await retry_async(fn, max_attempts=3)
    assert calls == 3


async def test_async_on_retry_is_awaited(fast_sleep):
    seen: list[int] = []

    async def on_retry(attempt: int, error: ProviderError, delay: float) -> None:
        seen.append(attempt)

    calls = 0

    async def fn() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ProviderError("x", status=500, retryable=True)
        return "ok"

    assert await retry_async(fn, on_retry=on_retry) == "ok"
    assert seen == [1]
