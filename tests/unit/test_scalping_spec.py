"""Tests for the scalping specs (P3.13): micro mean-reversion vs momentum (both-owed).

Verifies the definitional divergence (MR and momentum take EXACT opposite sides on every
triggered bar), the same-day-return guard (no scalp on a day's first bar), the point-in-time
prefix invariance (the hard no-lookahead precondition), and the factories.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from lab.core.types import BarInterval, Candle, Side
from lab.research.strategies.scalping import (
    ScalpMeanReversionSpec,
    ScalpMomentumSpec,
    scalp_mean_reversion_spec,
    scalp_momentum_spec,
)

IST = ZoneInfo("Asia/Kolkata")
_BARS_PER_DAY = 75


def _oscillating() -> list[Candle]:
    # per-bar moves well above the 0.2% threshold in both directions, across several IST days
    day0 = datetime(2024, 7, 1, 9, 15, tzinfo=IST)
    out: list[Candle] = []
    for i in range(300):
        c = 100.0 * (1.0 + 0.01 * math.sin(i / 1.7))  # ~1% swings -> big per-bar returns
        d, b = divmod(i, _BARS_PER_DAY)
        ts = day0 + timedelta(days=d, minutes=5 * b)
        out.append(Candle("X", BarInterval.MIN_5, ts, c, c + 0.5, c - 0.5, c, 1000))
    return out


_CANDLES = _oscillating()


def test_mr_and_momentum_take_exact_opposite_sides() -> None:
    mr = ScalpMeanReversionSpec().generate_signals(_CANDLES)
    mom = ScalpMomentumSpec().generate_signals(_CANDLES)
    triggered = 0
    for a, b in zip(mr, mom, strict=True):
        assert a.strength == b.strength  # they trigger on the same bars (same |return| gate)
        if a.strength == 1.0:
            assert a.side is not b.side  # exact opposites: fade vs chase
            assert {a.side, b.side} == {Side.LONG, Side.SHORT}
            triggered += 1
    assert triggered > 0  # the oscillating fixture actually triggers scalps


def test_no_scalp_on_the_first_bar_of_a_day() -> None:
    # same-day-return guard: the day's first bar has no valid prior-bar return -> flat.
    signals = ScalpMomentumSpec().generate_signals(_CANDLES)
    for i, s in enumerate(signals):
        if i % _BARS_PER_DAY == 0:
            assert s.strength == 0.0


def test_scalp_prefix_invariance_both_specs() -> None:
    # HARD no-lookahead precondition (both directions): per-bar targets are prefix-invariant.
    for spec in (ScalpMeanReversionSpec(), ScalpMomentumSpec()):
        full = spec.generate_signals(_CANDLES)
        for k in range(1, len(_CANDLES) + 1):
            prefix = spec.generate_signals(_CANDLES[:k])
            assert [(s.side, s.strength) for s in prefix] == [
                (s.side, s.strength) for s in full[:k]
            ]


def test_factories_read_params_and_validate() -> None:
    assert scalp_mean_reversion_spec({"entry_threshold": 0.003}).entry_threshold == 0.003
    assert scalp_mean_reversion_spec({}).entry_threshold == 0.002
    assert scalp_mean_reversion_spec({}).name == "scalp_mean_reversion"
    assert scalp_momentum_spec({"entry_threshold": 0.001}).entry_threshold == 0.001
    assert scalp_momentum_spec({}).name == "scalp_momentum"
    with pytest.raises(ValueError, match="entry_threshold"):
        ScalpMeanReversionSpec(entry_threshold=0.0)
    with pytest.raises(ValueError, match="entry_threshold"):
        ScalpMomentumSpec(entry_threshold=-0.001)
