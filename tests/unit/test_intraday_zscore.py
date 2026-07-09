"""Tests for the intraday-reset rolling z-score (the P3.3 mean-reversion signal).

Pins: the definition (population-std z over the current-day window including bar i), the
day-boundary reset (the whole point — an overnight gap must NOT read as a huge z, which
would turn the fade into a gap-fade), the full-window warmup, and strict causality.
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


def _bar(ts: datetime, close: float) -> Candle:
    # Only close feeds the z-score; H/L straddle it by a hair.
    return Candle("X", BarInterval.MIN_5, ts, close, close + 0.1, close - 0.1, close, 1000)


def _day(prices: list[float], day: datetime) -> list[Candle]:
    return [_bar(day + timedelta(minutes=5 * i), p) for i, p in enumerate(prices)]


def test_intraday_zscore_matches_definition_within_day() -> None:
    day = datetime(2024, 7, 15, 9, 15, tzinfo=IST)
    closes = [100.0, 102.0, 101.0, 105.0, 103.0]
    z = indicators.intraday_zscore(OHLCV.from_candles(_day(closes, day)), period=3)
    assert math.isnan(z[0]) and math.isnan(z[1])  # warmup: fewer than 3 current-day bars
    for i in range(2, 5):
        w = np.array(closes[i - 2 : i + 1])
        expected = (closes[i] - w.mean()) / w.std()  # population std (ddof=0)
        assert z[i] == pytest.approx(expected)


def test_intraday_zscore_resets_each_day_and_ignores_overnight_gap() -> None:
    d1 = datetime(2024, 7, 15, 9, 15, tzinfo=IST)
    d2 = datetime(2024, 7, 16, 9, 15, tzinfo=IST)
    # Day 2 gaps up ~100% (100s -> 200s). A trailing window would read that as an enormous
    # z at the open and fade against the gap; the intraday reset makes it warmup-NaN instead.
    candles = _day([100.0, 101.0, 102.0, 103.0], d1) + _day([200.0, 201.0, 202.0], d2)
    z = indicators.intraday_zscore(OHLCV.from_candles(candles), period=3)
    assert math.isnan(z[4]) and math.isnan(z[5])  # day 2's warmup -- NOT a gap-driven extreme
    # idx 6 uses ONLY day-2 closes [200,201,202]; a small, finite z, blind to the gap.
    w = np.array([200.0, 201.0, 202.0])
    assert z[6] == pytest.approx((202.0 - w.mean()) / w.std())
    assert abs(z[6]) < 2.0  # modest, not the huge value a cross-gap window would produce


def test_intraday_zscore_flat_window_is_nan() -> None:
    day = datetime(2024, 7, 15, 9, 15, tzinfo=IST)
    z = indicators.intraday_zscore(OHLCV.from_candles(_day([100.0] * 5, day)), period=3)
    assert all(math.isnan(v) for v in z)  # a flat window has std 0 -> z undefined -> NaN


def test_intraday_zscore_is_causal_prefix_invariant() -> None:
    day = datetime(2024, 7, 15, 9, 15, tzinfo=IST)
    candles = _day([100.0, 102.0, 99.0, 105.0, 98.0, 101.0, 103.0], day)
    full = indicators.intraday_zscore(OHLCV.from_candles(candles), period=3)
    for k in range(1, len(candles) + 1):
        prefix = indicators.intraday_zscore(OHLCV.from_candles(candles[:k]), period=3)
        assert np.allclose(prefix, full[:k], equal_nan=True)
