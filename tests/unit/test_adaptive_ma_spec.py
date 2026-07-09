"""Tests for the adaptive-MA specs (P3.7): KAMA fast/slow cross (V1) and slope (V2).

Verifies V1's position against the fast/slow KAMA cross, V2's against a single KAMA's slope,
that the two GENUINELY DIVERGE (the reason both are owed -- and the property a price-vs-single
-KAMA cross would have LACKED), point-in-time prefix invariance, and the factories.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from lab.core.types import BarInterval, Candle, Side
from lab.data.features.indicators import kama
from lab.data.features.ohlcv import OHLCV
from lab.research.strategies.adaptive_ma import (
    AdaptiveMaCrossSpec,
    AdaptiveMaSlopeSpec,
    adaptive_ma_cross_spec,
    adaptive_ma_slope_spec,
)

IST = ZoneInfo("Asia/Kolkata")


def _candles(prices: list[float]) -> list[Candle]:
    day = datetime(2024, 7, 15, 9, 15, tzinfo=IST)
    return [
        Candle("X", BarInterval.MIN_5, day + timedelta(minutes=5 * i), p, p + 0.5, p - 0.5, p, 1000)
        for i, p in enumerate(prices)
    ]


def _zigzag() -> list[float]:
    # 4 up/down cycles (15 bars each) -> clear reversals where the fast KAMA leads the slow,
    # so the fast/slow cross diverges from a single KAMA's slope.
    prices: list[float] = []
    value = 100.0
    for _ in range(4):
        for _ in range(15):
            value += 1.0
            prices.append(value)
        for _ in range(15):
            value -= 1.0
            prices.append(value)
    return prices


_PRICES = _zigzag()


def test_cross_position_follows_fast_vs_slow_kama() -> None:
    candles = _candles(_PRICES)
    fast = kama(OHLCV.from_candles(candles), 10)
    slow = kama(OHLCV.from_candles(candles), 30)
    signals = AdaptiveMaCrossSpec(fast_period=10, slow_period=30).generate_signals(candles)
    saw_long = saw_short = False
    for i, s in enumerate(signals):
        if not (math.isfinite(float(fast[i])) and math.isfinite(float(slow[i]))):
            assert s.strength == 0.0
        elif float(fast[i]) > float(slow[i]):
            assert s.side is Side.LONG and s.strength == 1.0
            saw_long = True
        elif float(fast[i]) < float(slow[i]):
            assert s.side is Side.SHORT and s.strength == 1.0
            saw_short = True
    assert saw_long and saw_short  # the series exercises both sides of the cross


def test_slope_position_follows_kama_direction() -> None:
    candles = _candles(_PRICES)
    ma = kama(OHLCV.from_candles(candles), 10)
    signals = AdaptiveMaSlopeSpec(kama_period=10).generate_signals(candles)
    saw_long = saw_short = False
    for i, s in enumerate(signals):
        if i == 0 or not (math.isfinite(float(ma[i])) and math.isfinite(float(ma[i - 1]))):
            assert s.strength == 0.0
        elif float(ma[i]) > float(ma[i - 1]):
            assert s.side is Side.LONG and s.strength == 1.0
            saw_long = True
        elif float(ma[i]) < float(ma[i - 1]):
            assert s.side is Side.SHORT and s.strength == 1.0
            saw_short = True
    assert saw_long and saw_short


def test_cross_and_slope_genuinely_diverge() -> None:
    # The whole reason both are owed (and the property the degenerate price-vs-single-KAMA
    # 'cross' LACKED): on some bar the fast/slow cross and the single-KAMA slope disagree.
    candles = _candles(_PRICES)
    cross = AdaptiveMaCrossSpec(fast_period=10, slow_period=30).generate_signals(candles)
    slope = AdaptiveMaSlopeSpec(kama_period=10).generate_signals(candles)
    disagree = sum(
        1
        for c, s in zip(cross, slope, strict=True)
        if c.strength == 1.0 and s.strength == 1.0 and c.side is not s.side
    )
    assert disagree > 0  # genuinely different bets -- proven, not asserted


def test_cross_prefix_invariance() -> None:
    spec = AdaptiveMaCrossSpec(fast_period=10, slow_period=30)
    candles = _candles(_PRICES)
    full = spec.generate_signals(candles)
    for k in range(1, len(candles) + 1):
        prefix = spec.generate_signals(candles[:k])
        assert [(x.side, x.strength) for x in prefix] == [(x.side, x.strength) for x in full[:k]]


def test_slope_prefix_invariance() -> None:
    spec = AdaptiveMaSlopeSpec(kama_period=10)
    candles = _candles(_PRICES)
    full = spec.generate_signals(candles)
    for k in range(1, len(candles) + 1):
        prefix = spec.generate_signals(candles[:k])
        assert [(x.side, x.strength) for x in prefix] == [(x.side, x.strength) for x in full[:k]]


def test_factories_read_params_and_validate() -> None:
    cross = adaptive_ma_cross_spec({"fast_period": 8, "slow_period": 40})
    assert cross.fast_period == 8 and cross.slow_period == 40
    assert cross.name == "adaptive_ma_cross" and cross.interval is BarInterval.MIN_5
    assert (
        adaptive_ma_cross_spec({}).fast_period == 10
        and adaptive_ma_cross_spec({}).slow_period == 30
    )
    assert adaptive_ma_slope_spec({"kama_period": 20}).kama_period == 20
    assert (
        adaptive_ma_slope_spec({}).kama_period == 10
        and adaptive_ma_slope_spec({}).name == "adaptive_ma_slope"
    )
    with pytest.raises(ValueError, match="fast_period"):
        AdaptiveMaCrossSpec(fast_period=1)
    with pytest.raises(ValueError, match="slow_period must be > fast_period"):
        AdaptiveMaCrossSpec(fast_period=20, slow_period=10)
    with pytest.raises(ValueError, match="kama_period"):
        AdaptiveMaSlopeSpec(kama_period=1)
