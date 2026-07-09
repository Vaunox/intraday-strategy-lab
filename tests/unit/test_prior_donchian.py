"""Tests for prior_donchian (P3.6): global prior-N-bar channel, excludes current bar, causal.

Pins that it excludes the current bar (so a close can genuinely break it), that it is
GLOBAL (spans the overnight gap -- the distinction from intraday_donchian), and -- load
bearing -- that it is strictly causal (prefix-invariant, reads only prior bars).
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


def _bar(ts: datetime, high: float, low: float, close: float) -> Candle:
    return Candle("X", BarInterval.MIN_5, ts, close, high, low, close, 1000)


def _day(rows: list[tuple[float, float, float]], day: datetime) -> list[Candle]:
    return [_bar(day + timedelta(minutes=5 * i), h, low, c) for i, (h, low, c) in enumerate(rows)]


def test_prior_donchian_excludes_current_bar_and_is_global_across_days() -> None:
    d1 = datetime(2024, 7, 15, 9, 15, tzinfo=IST)
    d2 = datetime(2024, 7, 16, 9, 15, tzinfo=IST)
    candles = _day([(10, 9, 10), (12, 11, 12), (11, 10, 11)], d1) + _day(
        [(13, 12, 13), (9, 8, 9), (14, 13, 14)], d2
    )
    upper, lower = indicators.prior_donchian(OHLCV.from_candles(candles), period=3)
    assert math.isnan(upper[0]) and math.isnan(upper[1]) and math.isnan(upper[2])  # warmup
    # Bar 3 (day 2's first bar): channel = max high of bars 0..2 = 12 -- GLOBAL, spans the gap,
    # references DAY-1 highs; excludes bar 3 itself.
    assert upper[3] == pytest.approx(12.0)
    assert lower[3] == pytest.approx(9.0)  # min low of bars 0..2
    assert upper[4] == pytest.approx(13.0)  # max high of bars 1..3
    # Contrast: intraday_donchian RESETS at day 2 -> bar 3 has no intraday prior -> NaN.
    intra_upper, _ = indicators.intraday_donchian(OHLCV.from_candles(candles), period=3)
    assert math.isnan(intra_upper[3])  # the global-vs-intraday distinction, load-bearing for P3.6


def test_prior_donchian_causal_prefix_invariant() -> None:
    day = datetime(2024, 7, 15, 9, 15, tzinfo=IST)
    candles = _day(
        [(10, 9, 10), (12, 11, 12), (11, 10, 11), (13, 12, 13), (9, 8, 9), (14, 13, 14)], day
    )
    full_u, full_l = indicators.prior_donchian(OHLCV.from_candles(candles), period=3)
    for k in range(1, len(candles) + 1):
        u, low = indicators.prior_donchian(OHLCV.from_candles(candles[:k]), period=3)
        assert np.allclose(u, full_u[:k], equal_nan=True)
        assert np.allclose(low, full_l[:k], equal_nan=True)
