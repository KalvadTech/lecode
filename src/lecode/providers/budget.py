"""One cost admission threshold shared by every call in a headless run."""

from __future__ import annotations

import math
from collections.abc import AsyncIterator
from typing import Any

from lecode.providers.catalog import AmbiguousModelError, Catalog, ModelNotFoundError
from lecode.providers.openai_compat import ProviderError
from lecode.providers.types import ChatMessage, CompletedMessage, StreamEvent, Usage, collect


class CostLimitError(ProviderError):
    """No further model requests may be admitted in this run."""


class BudgetedProvider:
    """Wrap the shared provider, including child calls and optional completions.

    Calls already admitted may exceed the threshold. Admission and settlement
    are synchronous, so concurrent tasks share the same ledger without a lock.
    """

    def __init__(self, provider: Any, catalog: Catalog, max_cost: float) -> None:
        if not math.isfinite(max_cost) or max_cost <= 0:
            raise ValueError("max_cost must be finite and greater than zero")
        self.provider = provider
        self.catalog = catalog
        self.max_cost = max_cost
        self.cost_usd = 0.0
        self.input_tokens = 0
        self.output_tokens = 0
        self.error: CostLimitError | None = None
        self.usage_incomplete = False

    def check(self, model: str) -> None:
        """Reject unknown pricing before the request, including model overrides."""
        if self.error is not None:
            raise self.error
        try:
            pricing = self.catalog.get(model).pricing
        except (ModelNotFoundError, AmbiguousModelError):
            pricing = None
        if (
            pricing is None
            or not pricing.known
            or not all(
                math.isfinite(price) and price >= 0
                for price in (pricing.prompt, pricing.completion)
            )
        ):
            self.error = CostLimitError(f"cost limit: price is unknown for model '{model}'")
            raise self.error

    def usage_totals(self, model: str, usage: dict[str, Any] | None) -> tuple[int, int, float]:
        """Validate one request's usage independently of the shared admission state."""
        if not isinstance(usage, dict) or not usage or usage.get("incomplete"):
            raise ValueError("missing usage")
        tokens = []
        for primary, alias in (
            ("input_tokens", "prompt_tokens"),
            ("output_tokens", "completion_tokens"),
        ):
            value = usage.get(primary, usage.get(alias))
            if value is None and usage.get("cost_usd") is not None:
                value = 0
            if isinstance(value, bool):
                raise ValueError("invalid tokens")
            number = float(value)
            if not math.isfinite(number) or number < 0 or not number.is_integer():
                raise ValueError("invalid tokens")
            tokens.append(int(number))
        if usage.get("cost_usd") is not None:
            if isinstance(usage["cost_usd"], bool):
                raise ValueError("invalid cost")
            cost = float(usage["cost_usd"])
        else:
            pricing = self.catalog.get(model).pricing
            cost = (tokens[0] * pricing.prompt + tokens[1] * pricing.completion) / 1e6
        if not math.isfinite(cost) or cost < 0:
            raise ValueError("invalid cost")
        return tokens[0], tokens[1], cost

    def _settle(self, model: str, usage: dict[str, Any] | None) -> bool:
        try:
            input_tokens, output_tokens, cost = self.usage_totals(model, usage)
        except (TypeError, ValueError, OverflowError):
            self.usage_incomplete = True
            self.error = self.error or CostLimitError("cost limit: request spend is unknown")
            return False
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cost_usd += cost
        if self.cost_usd >= self.max_cost:
            self.error = self.error or CostLimitError(f"cost limit reached (${self.max_cost:g})")
        return True

    async def stream_chat(
        self, messages: list[ChatMessage], model: str, **kwargs: Any
    ) -> AsyncIterator[StreamEvent]:
        self.check(model)
        usage = None
        stream = None
        stream_error = None
        try:
            stream = self.provider.stream_chat(messages, model=model, **kwargs)
            async for event in stream:
                if isinstance(event, Usage):
                    usage = event.usage
                yield event
        except Exception as exc:
            stream_error = exc
        finally:
            usage_valid = self._settle(model, usage)
            aclose = getattr(stream, "aclose", None)
            if aclose is not None:
                await aclose()
        if not usage_valid:
            raise self.error from stream_error
        if stream_error is not None:
            raise stream_error

    async def complete(
        self, messages: list[ChatMessage], model: str, **kwargs: Any
    ) -> CompletedMessage:
        return await collect(self.stream_chat(messages, model=model, **kwargs))

    async def aclose(self) -> None:
        aclose = getattr(self.provider, "aclose", None)
        if aclose is not None:
            await aclose()
