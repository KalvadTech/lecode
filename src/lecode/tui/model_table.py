"""The model-catalog table shared by ``/models`` and the setup wizard.

Cost cells run green (cheapest) → red (priciest); context cells run light
blue (smallest window) → dark blue (largest). Both ramps are relative to the
rows shown — red is the most expensive *in this table*. Context windows span
orders of magnitude, so that ramp is logarithmic; price is linear.
"""

from __future__ import annotations

import colorsys
import math
from collections.abc import Iterable

from rich import box
from rich.table import Table
from rich.text import Text

from lecode.providers.catalog import ModelInfo
from lecode.tui.statusline import human_tokens
from lecode.tui.themes import THEME, Theme

#: Context ramp endpoints: light blue (smallest) → dark blue (largest).
_CTX_LIGHT = (0xA6 / 255, 0xD8 / 255, 0xFF / 255)
_CTX_DARK = (0x1E / 255, 0x3A / 255, 0x8A / 255)


def _hex(rgb: tuple[float, ...]) -> str:
    return "#" + "".join(f"{round(c * 255):02x}" for c in rgb)


def _t(value: float, lo: float, hi: float) -> float:
    """``value`` normalized to [0, 1] over [lo, hi]; 0 when the ramp is flat."""
    if hi <= lo:
        return 0.0
    return min(max((value - lo) / (hi - lo), 0.0), 1.0)


def cost_style(value: float, lo: float, hi: float) -> str:
    """Green (``lo``) → yellow → red (``hi``), as a Rich hex style."""
    hue = (1 / 3) * (1 - _t(value, lo, hi))  # 120° green → 0° red
    return _hex(colorsys.hsv_to_rgb(hue, 0.85, 0.9))


def context_style(window: int, lo: float, hi: float) -> str:
    """Light blue (``lo``) → dark blue (``hi``) on a log scale."""
    t = _t(math.log(window), lo, hi)
    channels = (lo_c + (hi_c - lo_c) * t for lo_c, hi_c in zip(_CTX_LIGHT, _CTX_DARK, strict=True))
    return _hex(tuple(channels))


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
