"""Retry with exponential backoff and full jitter for provider calls.

Only retryable failures are retried: :class:`~lecode.providers.openai_compat.ProviderError`
with ``retryable=True`` (which already covers transport/timeout errors, mapped
by the client). Everything else raises immediately.
"""

from __future__ import annotations

import inspect
from asyncio import sleep
from collections.abc import Awaitable, Callable
from random import uniform
from typing import Any

from lecode.providers.openai_compat import ProviderError

#: on_retry(attempt, error, delay) — sync or async; called before each sleep.
OnRetry = Callable[[int, ProviderError, float], Any]


async def retry_async[T](
    fn: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 5,
    base_delay: float = 0.5,
    max_delay: float = 30.0,
    on_retry: OnRetry | None = None,
) -> T:
    """Call ``fn`` until it succeeds, with full-jitter exponential backoff."""
    attempt = 0
    while True:
        attempt += 1
        try:
            return await fn()
        except ProviderError as e:
            if not e.retryable or attempt >= max_attempts:
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay = uniform(0.0, delay)
            if on_retry is not None:
                result = on_retry(attempt, e, delay)
                if inspect.isawaitable(result):
                    await result
            await sleep(delay)
