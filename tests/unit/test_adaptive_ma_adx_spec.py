"""Tests for P4.4 adaptive_ma_adx: KAMA trend taken only when ADX confirms strength.

Covers the no-lookahead prefix-invariance precondition (hard gate), the ADX gating (positions
only in confirmed trends, direction = KAMA slope), and the factory.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from lab.core.types import BarInterval, Candle, Side
from lab.data.features.indicators import adx, kama
from lab.data.features.ohlcv import OHLCV
from lab.research.strategies.adaptive_ma_adx import AdaptiveMaAdxSpec, adaptive_ma_adx_spec

IST = ZoneInfo("Asia/Kolkata")
_BARS_PER_DAY = 75


def _sessions(closes: list[float]) -> list[Candle]:
    day0 = datetime(2024, 7, 1, 9, 15, tzinfo=IST)
    out: list[Candle] = []
    for i, c in enumerate(closes):
        d, b = divmod(i, _BARS_PER_DAY)
        ts = day0 + timedelta(days=d, minutes=5 * b)
        out.append(Candle("X", BarInterval.MIN_5, ts, c, c + 0.4, c - 0.4, c, 1000))
    return out


def _strong_trend() -> list[Candle]:
    # a strong steady up-trend then down-trend -> ADX climbs well above 25; KAMA slope flips
    up = [100.0 + 0.5 * i for i in range(150)]
    down = [up[-1] - 0.5 * i for i in range(1, 151)]
    return _sessions(up + down)


_CANDLES = _strong_trend()


def test_positions_only_in_confirmed_trends_direction_is_kama_slope() -> None:
    spec = AdaptiveMaAdxSpec()
    ohlcv = OHLCV.from_candles(_CANDLES)
    k, a = kama(ohlcv, spec.kama_period), adx(ohlcv, spec.adx_period)
    signals = spec.generate_signals(_CANDLES)
    longs = shorts = 0
    for i, s in enumerate(signals):
        if s.strength == 0.0:
            continue
        assert float(a[i]) > spec.adx_threshold  # only in a confirmed trend
        if s.side is Side.LONG:
            assert float(k[i]) > float(k[i - 1])
            longs += 1
        else:
            assert float(k[i]) < float(k[i - 1])
            shorts += 1
    assert longs > 0 and shorts > 0  # both trend halves exercised
    gated = sum(  # the ADX gate blocks some bars (warmup / chop / reversal) -> not a no-op
        1
        for i in range(1, len(_CANDLES))
        if not (math.isfinite(float(a[i])) and float(a[i]) > spec.adx_threshold)
    )
    assert gated > 0


def test_adaptive_ma_adx_prefix_invariance() -> None:
    spec = AdaptiveMaAdxSpec()
    full = spec.generate_signals(_CANDLES)
    for k in range(1, len(_CANDLES) + 1):
        prefix = spec.generate_signals(_CANDLES[:k])
        assert [(s.side, s.strength) for s in prefix] == [(s.side, s.strength) for s in full[:k]]


def test_factory_reads_params_and_validates() -> None:
    spec = adaptive_ma_adx_spec({"kama_period": 15, "adx_threshold": 20.0})
    assert spec.kama_period == 15 and spec.adx_threshold == 20.0
    assert spec.name == "adaptive_ma_adx" and spec.adx_period == 14
    assert (
        adaptive_ma_adx_spec({}).kama_period == 10
        and adaptive_ma_adx_spec({}).adx_threshold == 25.0
    )
    with pytest.raises(ValueError, match="adx_threshold"):
        AdaptiveMaAdxSpec(adx_threshold=0.0)
