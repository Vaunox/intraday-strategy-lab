"""Tests for the intraday-reset Donchian channel (the P3.2 breakout range).

Pins the day-boundary reset (the whole point: an early-session bar sees only TODAY's
range, never a level carried across the overnight gap) and strict causality.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from lab.core.types import BarInterval, Candle
from lab.data.features import indicators
from lab.data.features.ohlcv import OHLCV

IST = ZoneInfo("Asia/Kolkata")


def _two_day_candles() -> list[Candle]:
    """Two IST trading days; day 1 highs reach 120, day 2 opens lower (108-111)."""
    candles: list[Candle] = []
    days = (datetime(2024, 7, 15, 9, 15, tzinfo=IST), datetime(2024, 7, 16, 9, 15, tzinfo=IST))
    for d, day in enumerate(days):
        rows = (
            [(110, 100, 105), (120, 108, 115), (115, 110, 112)]
            if d == 0
            else [(108, 100, 104), (109, 103, 106), (111, 105, 110)]
        )
        for i, (high, low, close) in enumerate(rows):
            candles.append(
                Candle(
                    "X",
                    BarInterval.MIN_5,
                    day + timedelta(minutes=5 * i),
                    float(close),
                    float(high),
                    float(low),
                    float(close),
                    1000,
                )
            )
    return candles


def test_intraday_donchian_resets_at_day_boundary_ignoring_prior_day() -> None:
    ohlcv = OHLCV.from_candles(_two_day_candles())  # idx 0..2 = day 1, 3..5 = day 2
    upper, lower = indicators.intraday_donchian(ohlcv, period=3)
    # Day 2's FIRST bar (idx 3) has no prior intraday bar -> NaN (nothing to break yet).
    assert math.isnan(upper[3]) and math.isnan(lower[3])
    # Day 2's SECOND bar (idx 4) sees ONLY day 2 bar 0 (high 108), NOT day 1's high of 120.
    assert upper[4] == pytest.approx(108.0)
    # Day 2's THIRD bar (idx 5) sees day 2 bars 0..1 (highs 108, 109) -> 109, still not 120.
    assert upper[5] == pytest.approx(109.0)
    # Contrast: the GLOBAL donchian carries a prior-day high across the overnight gap --
    # this is exactly the causality the intraday reset removes.
    global_upper, _ = indicators.donchian(ohlcv, period=3)
    assert global_upper[4] == pytest.approx(115.0)  # max(high[2..4]) includes day 1 bar 2
    assert global_upper[4] != pytest.approx(108.0)


def test_intraday_donchian_is_causal_prefix_invariant() -> None:
    # Causality == prefix-invariance: the value at bar i must not change when later bars
    # are appended (it uses only bars 0..i-1 of the same day).
    candles = _two_day_candles()
    full_upper, full_lower = indicators.intraday_donchian(OHLCV.from_candles(candles), period=2)
    for k in range(2, len(candles) + 1):
        pre_upper, pre_lower = indicators.intraday_donchian(
            OHLCV.from_candles(candles[:k]), period=2
        )
        assert np.allclose(pre_upper, full_upper[:k], equal_nan=True)
        assert np.allclose(pre_lower, full_lower[:k], equal_nan=True)
