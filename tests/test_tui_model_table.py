"""Tests for the gradient-colored model catalog table."""

from __future__ import annotations

import math
from io import StringIO

import pytest
from rich.console import Console

from lecode.providers.catalog import Modalities, ModelInfo, Pricing
from lecode.tui.model_table import _t, build_model_table, context_style, cost_style


def entry(
    model_id: str,
    context_window: int,
    prompt: float,
    completion: float,
    *,
    known: bool = True,
) -> ModelInfo:
    return ModelInfo(
        id=model_id,
        name=model_id,
        context_window=context_window,
        pricing=Pricing(prompt=prompt, completion=completion, known=known),
        modalities=Modalities(input=["text"], output=["text"]),
    )


def render(table) -> str:
    out = StringIO()
    Console(file=out, width=200).print(table)
    return out.getvalue()


def test_cost_ramp_runs_green_to_red():
    assert cost_style(0.0, 0.0, 15.0) == "#22e622"  # cheapest: green
    assert cost_style(15.0, 0.0, 15.0) == "#e62222"  # priciest: red
    r, g, b = (int(cost_style(7.5, 0.0, 15.0)[i : i + 2], 16) for i in (1, 3, 5))
    assert r == g > b  # the midpoint is yellow


def test_context_ramp_runs_light_to_dark_blue():
    lo, hi = math.log(32_000), math.log(2_000_000)
    assert context_style(32_000, lo, hi) == "#a6d8ff"  # smallest: light blue
    assert context_style(2_000_000, lo, hi) == "#1e3a8a"  # largest: dark blue


def test_context_ramp_is_logarithmic():
    """A doubling of the window moves the same amount anywhere on the ramp."""
    lo, hi = math.log(8_000), math.log(1_024_000)
    low_step = _t(math.log(16_000), lo, hi) - _t(math.log(8_000), lo, hi)
    high_step = _t(math.log(512_000), lo, hi) - _t(math.log(256_000), lo, hi)
    assert low_step == pytest.approx(high_step)


def test_table_marks_current_and_shows_prices():
    entries = [
        entry("cheap/model", 128_000, 0.05, 0.4),
        entry("pricey/model", 1_000_000, 3.0, 15.0),
    ]
    table = build_model_table(entries, marked_id="cheap/model", marker="(current)")
    text = render(table)
    assert "cheap/model (current)" in text
    assert "pricey/model" in text and "(current)" not in text.split("pricey/model")[1]
    assert "$0.05" in text and "$15" in text and "128.0k" in text
    # the priciest out-price cell is red, the cheapest in-price cell green
    pricey_out = list(table.columns[3].cells)[1]
    cheap_in = next(iter(table.columns[2].cells))
    assert pricey_out.style == "#e62222"
    assert cheap_in.style == "#22e622"


def test_unknown_pricing_and_window_show_question_marks():
    entries = [
        entry("plain/model", 128_000, 1.0, 2.0),
        entry("mystery/model", 0, 0.0, 0.0, known=False),
    ]
    table = build_model_table(entries, marked_id="", marker="")
    cells = [list(col.cells) for col in table.columns]
    assert all(cell.plain == "?" for cell in (cells[1][1], cells[2][1], cells[3][1]))
    # the unknown row stays out of the ramps: the known row spans them alone
    assert cells[2][0].style == "#22e622"
    assert cells[3][0].style == "#e62222"


def test_numbered_menu_matches_selection_order():
    entries = [entry("b/model", 128_000, 1.0, 2.0), entry("a/model", 64_000, 0.5, 1.0)]
    text = render(build_model_table(entries, numbered=True, marked_id=None, marker=""))
    assert "#" in text.splitlines()[0]
    lines = [ln for ln in text.splitlines() if "model" in ln]
    assert [ln.split()[0] for ln in lines] == ["1", "2"]


def test_single_entry_does_not_divide_by_zero():
    text = render(build_model_table([entry("only/model", 128_000, 1.0, 1.0)]))
    assert "only/model" in text and "$1" in text
