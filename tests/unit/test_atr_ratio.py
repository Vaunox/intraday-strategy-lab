"""Load-bearing causality test for atr_ratio (P3.8 shared-regime hard precondition).

Both P3.8 studies (C1, C2) depend on the ATR-ratio regime primitive; it must be
prefix-invariant -- reads only completed prior bars, no forward leak. If this fails, neither
study runs. Also pins the definition (short ATR / long ATR).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np

from lab.core.types import BarInterval, Candle
from lab.data.features.indicators import atr, atr_ratio
from lab.data.features.ohlcv import OHLCV

IST = ZoneInfo("Asia/Kolkata")


def _candles(n: int) -> list[Candle]:
    day = datetime(2024, 7, 15, 9, 15, tzinfo=IST)
    rng = np.random.default_rng(2)
    close = 100.0 + np.cumsum(rng.normal(0, 0.4, n))
    out: list[Candle] = []
    for i, c in enumerate(close):
        rng_range = abs(float(rng.normal(0, 0.3))) + 0.05
        out.append(
            Candle(
                "X",
                BarInterval.MIN_5,
                day + timedelta(minutes=5 * i),
                float(c),
                float(c) + rng_range,
                float(c) - rng_range,
                float(c),
                1000,
            )
        )
    return out


def test_atr_ratio_is_causal_prefix_invariant() -> None:
    candles = _candles(160)
    full = atr_ratio(OHLCV.from_candles(candles), 20, 100)
    assert np.isfinite(full[100:]).any()  # produces finite values past warmup
    for k in range(1, len(candles) + 1):
        prefix = atr_ratio(OHLCV.from_candles(candles[:k]), 20, 100)
        assert np.allclose(prefix, full[:k], equal_nan=True), f"atr_ratio forward leak at k={k}"


def test_atr_ratio_is_short_over_long() -> None:
    candles = _candles(160)
    ohlcv = OHLCV.from_candles(candles)
    ratio = atr_ratio(ohlcv, 20, 100)
    short_atr, long_atr = atr(ohlcv, 20), atr(ohlcv, 100)
    for i in range(len(candles)):
        if np.isfinite(short_atr[i]) and np.isfinite(long_atr[i]) and long_atr[i] > 0:
            assert ratio[i] == short_atr[i] / long_atr[i]
