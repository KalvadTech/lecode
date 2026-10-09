"""The model-catalog table shared by ``/models`` and the setup wizard.

Cost cells run green (cheapest) through yellow and orange to red (priciest);
context cells run light violet (smallest window) to logo deep purple
(largest). Both ramps are computed dynamically from the rows shown: red is
the most expensive *in this table*. Prices and windows span orders of
magnitude, so both ramps are log-scaled.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from rich import box
from rich.table import Table
from rich.text import Text

from lecode.providers.catalog import ModelInfo
from lecode.tui.statusline import human_tokens
from lecode.tui.themes import THEME, Theme

#: Cost ramp stops: green, yellow, orange, red (cheapest to priciest).
_COST_STOPS = (
    (0x22 / 255, 0xE6 / 255, 0x22 / 255),
    (0xE6 / 255, 0xE6 / 255, 0x22 / 255),
    (0xE6 / 255, 0x8A / 255, 0x22 / 255),
    (0xE6 / 255, 0x22 / 255, 0x22 / 255),
)

#: Context ramp stops: light violet to logo deep purple, from the brand
#: palette (THEME.tool, THEME.accent, and the splash logo's #7c3aed).
_CTX_STOPS = (
    (0xC4 / 255, 0xB5 / 255, 0xFD / 255),
    (0xA7 / 255, 0x8B / 255, 0xFA / 255),
    (0x7C / 255, 0x3A / 255, 0xED / 255),
)


def _hex(rgb: tuple[float, ...]) -> str:
    return "#" + "".join(f"{round(c * 255):02x}" for c in rgb)


def _t(value: float, lo: float, hi: float) -> float:
    """``value`` normalized to [0, 1] over [lo, hi]; 0 when the ramp is flat."""
    if hi <= lo:
        return 0.0
    return min(max((value - lo) / (hi - lo), 0.0), 1.0)


def _ramp(t: float, stops: tuple[tuple[float, ...], ...]) -> str:
    """Piecewise RGB interpolation across ``stops``; ``t`` clamps to [0, 1]."""
    pos = min(max(t, 0.0), 1.0) * (len(stops) - 1)
    i = min(int(pos), len(stops) - 2)
    frac = pos - i
    a, b = stops[i], stops[i + 1]
    return _hex(tuple(x + (y - x) * frac for x, y in zip(a, b, strict=True)))


def cost_style(value: float, lo: float, hi: float) -> str:
    """Green (``lo``) through yellow and orange to red (``hi``), log-scaled."""
    t = _t(math.log1p(value - lo), 0.0, math.log1p(hi - lo))
    return _ramp(t, _COST_STOPS)


def context_style(window: int, lo: float, hi: float) -> str:
    """Light violet (``lo``) to logo deep purple (``hi``) on a log scale."""
    return _ramp(_t(math.log(window), lo, hi), _CTX_STOPS)


def build_model_table(
    entries: Iterable[ModelInfo],
    *,
    numbered: bool = False,
    marked_id: str | None = None,
    marker: str = "(current)",
    theme: Theme = THEME,
) -> Table:
    """A table of catalog entries with gradient-colored cost/context cells.

    ``numbered`` adds a 1-based ``#`` column (the wizard's selection menu);
    ``marked_id`` gets a ``marker`` suffix in the accent color. Entries with
    unknown pricing or no context window show muted ``?`` cells and stay out
    of the ramps.
    """
    rows = list(entries)
    prices = [p for e in rows if e.pricing.known for p in (e.pricing.prompt, e.pricing.completion)]
    price_lo, price_hi = (min(prices), max(prices)) if prices else (0.0, 0.0)
    windows = [e.context_window for e in rows if e.context_window > 0]
    ctx_lo, ctx_hi = (math.log(min(windows)), math.log(max(windows))) if windows else (0.0, 0.0)

    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False, header_style=theme.muted)
    if numbered:
        table.add_column("#", justify="right", style=theme.muted)
    table.add_column("Model")
    table.add_column("Context", justify="right")
    table.add_column("In $/M", justify="right")
    table.add_column("Out $/M", justify="right")

    for i, entry in enumerate(rows, start=1):
        model = Text(entry.id)
        if entry.id == marked_id:
            model.append(f" {marker}", style=theme.accent)
        if entry.context_window > 0:
            ctx = Text(
                human_tokens(entry.context_window),
                style=context_style(entry.context_window, ctx_lo, ctx_hi),
            )
        else:
            ctx = Text("?", style=theme.muted)
        if entry.pricing.known:
            cost_in = Text(
                f"${entry.pricing.prompt:g}",
                style=cost_style(entry.pricing.prompt, price_lo, price_hi),
            )
            cost_out = Text(
                f"${entry.pricing.completion:g}",
                style=cost_style(entry.pricing.completion, price_lo, price_hi),
            )
        else:
            cost_in = cost_out = Text("?", style=theme.muted)
        row: list[Text] = [model, ctx, cost_in, cost_out]
        if numbered:
            row.insert(0, Text(str(i)))
        table.add_row(*row)
    return table
