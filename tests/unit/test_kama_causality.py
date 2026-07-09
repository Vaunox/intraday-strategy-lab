"""Load-bearing causality test for KAMA (P3.7 hard precondition): talib.KAMA reads past-only.

The P3.7 adaptive-MA specs use the TRAILING KAMA. Trailing is legitimate (it reflects past
data), but that must be PROVEN, not assumed: this pins that KAMA is prefix-invariant -- the
value at bar i is byte-identical whether or not later bars exist, so there is no forward
leak in the recursion. If this fails, P3.7 must not run.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np

from lab.core.types import BarInterval, Candle
from lab.data.features.indicators import kama
from lab.data.features.ohlcv import OHLCV

IST = ZoneInfo("Asia/Kolkata")


def _candles(prices: list[float]) -> list[Candle]:
    day = datetime(2024, 7, 15, 9, 15, tzinfo=IST)
    return [
        Candle("X", BarInterval.MIN_5, day + timedelta(minutes=5 * i), p, p + 0.5, p - 0.5, p, 1000)
        for i, p in enumerate(prices)
    ]


def test_kama_is_causal_prefix_invariant() -> None:
    # A trend+chop series so the KAMA efficiency ratio genuinely varies (a flat series would
    # make prefix-invariance trivial).
    prices = [100.0 + 5.0 * math.sin(i / 7.0) + 0.1 * i for i in range(80)]
    candles = _candles(prices)
    full = kama(OHLCV.from_candles(candles), 10)
    # Sanity: the indicator actually produces finite values (not an all-NaN degenerate case).
    assert np.isfinite(full[20:]).all()
    for k in range(1, len(candles) + 1):
        prefix = kama(OHLCV.from_candles(candles[:k]), 10)
        assert np.allclose(prefix, full[:k], equal_nan=True), f"KAMA forward leak at prefix k={k}"
