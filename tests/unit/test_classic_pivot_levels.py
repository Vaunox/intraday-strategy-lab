"""Tests for classic pivot R1/S1 (P3.5): prior-completed-session levels, strictly causal.

Pins the classic formula (R1 = 2P - prevLow, S1 = 2P - prevHigh) and -- load-bearing --
that the levels read ONLY the prior completed day, so there is no same-day or future leak
(prefix-invariance, including a prefix ending mid-day-2).
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


def _two_days() -> list[Candle]:
    d1 = datetime(2024, 7, 15, 9, 15, tzinfo=IST)
    d2 = datetime(2024, 7, 16, 9, 15, tzinfo=IST)
    # Day 1 aggregates to HLC = (110, 100, 106): P = 105.3333, R1 = 110.6667, S1 = 100.6667.
    return _day([(108, 100, 105), (110, 102, 106)], d1) + _day(
        [(107, 103, 105), (108, 104, 106)], d2
    )


def test_classic_pivot_levels_from_prior_day_hlc() -> None:
    r1, s1 = indicators.classic_pivot_levels(OHLCV.from_candles(_two_days()))
    assert math.isnan(r1[0]) and math.isnan(s1[0]) and math.isnan(r1[1]) and math.isnan(s1[1])
    assert r1[2] == pytest.approx(110.6667, abs=1e-3)  # 2P - prevLow
    assert s1[2] == pytest.approx(100.6667, abs=1e-3)  # 2P - prevHigh
    assert r1[3] == pytest.approx(r1[2]) and s1[3] == pytest.approx(s1[2])  # constant within day


def test_classic_pivot_levels_causal_prefix_invariant_no_same_day_leak() -> None:
    # A prefix ending mid-day-2 yields the SAME day-2 levels as the full series (they depend
    # only on the COMPLETE day 1), and appending later bars never changes an earlier level.
    candles = _two_days()
    full_r1, full_s1 = indicators.classic_pivot_levels(OHLCV.from_candles(candles))
    for k in range(1, len(candles) + 1):
        r1, s1 = indicators.classic_pivot_levels(OHLCV.from_candles(candles[:k]))
        assert np.allclose(r1, full_r1[:k], equal_nan=True)
        assert np.allclose(s1, full_s1[:k], equal_nan=True)
