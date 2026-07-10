"""Tests for the MA-crossover spec (P3.14): position = sign(fast SMA - slow SMA).

Verifies the crossover gating (LONG while fast > slow, SHORT while below), the point-in-time
prefix invariance (the hard no-lookahead precondition), and the factory.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from lab.core.types import BarInterval, Candle, Side
from lab.data.features.indicators import sma
from lab.data.features.ohlcv import OHLCV
from lab.research.strategies.ma_crossover import MaCrossoverSpec, ma_crossover_spec

IST = ZoneInfo("Asia/Kolkata")
_BARS_PER_DAY = 75


def _sessions(closes: list[float]) -> list[Candle]:
    day0 = datetime(2024, 7, 1, 9, 15, tzinfo=IST)
    out: list[Candle] = []
    for i, c in enumerate(closes):
        d, b = divmod(i, _BARS_PER_DAY)
        ts = day0 + timedelta(days=d, minutes=5 * b)
        out.append(Candle("X", BarInterval.MIN_5, ts, c, c + 0.5, c - 0.5, c, 1000))
    return out


def _up_then_down() -> list[Candle]:
    # rise then fall, so the fast SMA crosses the slow both ways (LONG regime then SHORT regime)
    up = [100.0 + 0.30 * i for i in range(150)]
    down = [up[-1] - 0.30 * i for i in range(1, 151)]
    return _sessions(up + down)


_CANDLES = _up_then_down()


def test_position_is_the_sign_of_fast_minus_slow_both_regimes() -> None:
    spec = MaCrossoverSpec()
    ohlcv = OHLCV.from_candles(_CANDLES)
    fast, slow = sma(ohlcv, spec.fast_period), sma(ohlcv, spec.slow_period)
    signals = spec.generate_signals(_CANDLES)
    longs = shorts = 0
    for i, s in enumerate(signals):
        if s.strength == 0.0:
            continue
        if s.side is Side.LONG:
            assert float(fast[i]) > float(slow[i])
            longs += 1
        else:
            assert float(fast[i]) < float(slow[i])
            shorts += 1
    assert longs > 0 and shorts > 0  # both regimes exercised by the up/down halves


def test_ma_crossover_prefix_invariance() -> None:
    # HARD no-lookahead precondition: per-bar target for 0..k-1 identical on the length-k prefix.
    spec = MaCrossoverSpec()
    full = spec.generate_signals(_CANDLES)
    for k in range(1, len(_CANDLES) + 1):
        prefix = spec.generate_signals(_CANDLES[:k])
        assert [(s.side, s.strength) for s in prefix] == [(s.side, s.strength) for s in full[:k]]


def test_factory_reads_params_and_validates() -> None:
    spec = ma_crossover_spec({"fast_period": 15, "slow_period": 40})
    assert spec.fast_period == 15 and spec.slow_period == 40
    assert spec.name == "ma_crossover" and spec.interval is BarInterval.MIN_5
    assert ma_crossover_spec({}).fast_period == 20
    assert ma_crossover_spec({}).slow_period == 50
    with pytest.raises(ValueError, match="fast_period < slow_period"):
        MaCrossoverSpec(fast_period=50, slow_period=20)
