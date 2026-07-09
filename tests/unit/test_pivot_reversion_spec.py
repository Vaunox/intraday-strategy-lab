"""Tests for the pivot-reversion spec (P3.5): classic pivot S/R fade.

Pins the fade-at-level entry, the hold-until-pivot-target exit, the first-day-flat warmup,
and -- crucially -- the NO-LOOKAHEAD prefix-invariance property (the study's hard
precondition: the prior-day levels must not leak same-day or future data).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from lab.core.types import BarInterval, Candle, Side
from lab.research.strategies.pivot_reversion import PivotReversionSpec, pivot_reversion_spec

IST = ZoneInfo("Asia/Kolkata")
D1 = datetime(2024, 7, 15, 9, 15, tzinfo=IST)
D2 = datetime(2024, 7, 16, 9, 15, tzinfo=IST)


def _bar(ts: datetime, high: float, low: float, close: float) -> Candle:
    return Candle("X", BarInterval.MIN_5, ts, close, high, low, close, 1000)


def _day(rows: list[tuple[float, float, float]], day: datetime) -> list[Candle]:
    return [_bar(day + timedelta(minutes=5 * i), h, low, c) for i, (h, low, c) in enumerate(rows)]


# Day 1 -> HLC (110, 100, 106): P = 105.3333, R1 = 110.6667, S1 = 100.6667.
_DAY1 = _day([(108, 100, 105), (110, 102, 106)], D1)


def test_shorts_at_resistance_holds_then_exits_at_pivot() -> None:
    spec = PivotReversionSpec()  # entry_band 0.001, exit_band 0.001
    # Day-2 bar 1 pokes to 111 (>= R1 110.667) -> SHORT; held while above the pivot; exits
    # when a later bar reverts down to within exit_band of P (105.333).
    candles = _DAY1 + _day(
        [(107, 104, 105), (111, 108, 109), (110, 108, 109), (108, 105, 105.3)], D2
    )
    signals = spec.generate_signals(candles)
    assert signals[0].strength == 0.0 and signals[1].strength == 0.0  # day 1: no prior pivot
    assert signals[2].strength == 0.0  # day-2 open, level not reached
    assert signals[3].side is Side.SHORT and signals[3].strength == 1.0  # reached resistance R1
    assert signals[4].side is Side.SHORT and signals[4].strength == 1.0  # held (above the pivot)
    assert signals[5].strength == 0.0  # reverted to the pivot P -> target hit, exit


def test_longs_at_support() -> None:
    spec = PivotReversionSpec()
    # Day-2 bar 1 pokes to low 100 (<= S1 100.667) -> LONG (support fade).
    candles = _DAY1 + _day([(107, 104, 105), (105, 100, 101)], D2)
    signals = spec.generate_signals(candles)
    assert signals[3].side is Side.LONG and signals[3].strength == 1.0


def test_first_day_is_flat_no_prior_pivot() -> None:
    spec = PivotReversionSpec()
    signals = spec.generate_signals(_DAY1)  # first day only -> no prior-day levels -> all flat
    assert all(s.strength == 0.0 for s in signals)


def test_no_lookahead_prefix_invariance() -> None:
    """HARD PRECONDITION: signals for any prefix (incl. mid-day-2) equal the prefix of full.

    Proves the prior-day levels leak neither same-day nor future data: a bar's signal never
    changes when later bars are appended.
    """
    spec = PivotReversionSpec()
    candles = _DAY1 + _day(
        [(107, 104, 105), (111, 108, 109), (110, 108, 109), (108, 105, 105.3), (101, 100, 100.5)],
        D2,
    )
    full = spec.generate_signals(candles)
    for k in range(1, len(candles) + 1):
        prefix = spec.generate_signals(candles[:k])
        assert [(s.side, s.strength) for s in prefix] == [(s.side, s.strength) for s in full[:k]]


def test_factory_reads_params_and_validates() -> None:
    spec = pivot_reversion_spec({"entry_band": 0.002, "exit_band": 0.0015})
    assert spec.entry_band == 0.002
    assert spec.exit_band == 0.0015
    assert spec.name == "pivot_reversion"
    assert spec.interval is BarInterval.MIN_5

    with pytest.raises(ValueError, match="entry_band"):
        PivotReversionSpec(entry_band=-0.1)
    with pytest.raises(ValueError, match="exit_band"):
        PivotReversionSpec(exit_band=-0.1)


def test_factory_defaults_when_params_absent() -> None:
    spec = pivot_reversion_spec({})
    assert spec.entry_band == 0.001
    assert spec.exit_band == 0.001
