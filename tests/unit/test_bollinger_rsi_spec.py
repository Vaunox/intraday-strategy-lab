"""Tests for P4.3 bollinger_rsi: fade a Bollinger-band touch confirmed by an RSI extreme.

Covers the no-lookahead prefix-invariance precondition (hard gate), the confluence gating
(enters only when both band + RSI agree), and the factory.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from lab.core.types import BarInterval, Candle, Side
from lab.data.features.indicators import bollinger, rsi
from lab.data.features.ohlcv import OHLCV
from lab.research.strategies.bollinger_rsi import BollingerRsiSpec, bollinger_rsi_spec

IST = ZoneInfo("Asia/Kolkata")
_BARS_PER_DAY = 75


def _sessions(closes: list[float]) -> list[Candle]:
    day0 = datetime(2024, 7, 1, 9, 15, tzinfo=IST)
    out: list[Candle] = []
    for i, c in enumerate(closes):
        d, b = divmod(i, _BARS_PER_DAY)
        ts = day0 + timedelta(days=d, minutes=5 * b)
        out.append(Candle("X", BarInterval.MIN_5, ts, c, c + 0.3, c - 0.3, c, 1000))
    return out


def _sawtooth() -> list[Candle]:
    # slow drift up, periodic SHARP drops -> close punches below the lower band with a low RSI
    closes: list[float] = []
    price = 100.0
    for i in range(300):
        price += 0.15 if i % 20 else -4.0  # a sharp -4 drop every 20 bars
        closes.append(price)
    return _sessions(closes)


_CANDLES = _sawtooth()


def test_entries_require_band_and_rsi_confluence() -> None:
    spec = BollingerRsiSpec(bb_num_std=1.5, rsi_oversold=40.0)  # gentler so the fixture fires
    ohlcv = OHLCV.from_candles(_CANDLES)
    upper, _mid, lower = bollinger(ohlcv, spec.bb_period, spec.bb_num_std)
    r = rsi(ohlcv, spec.rsi_period)
    signals = spec.generate_signals(_CANDLES)
    entries, prev = 0, 0.0
    for i, s in enumerate(signals):
        if s.strength == 1.0 and prev == 0.0:  # entered from flat
            if s.side is Side.LONG:
                assert (
                    float(_CANDLES[i].close) < float(lower[i]) and float(r[i]) < spec.rsi_oversold
                )
            else:
                assert (
                    float(_CANDLES[i].close) > float(upper[i])
                    and float(r[i]) > 100.0 - spec.rsi_oversold
                )
            entries += 1
        prev = s.strength
    assert entries > 0  # the sharp drops trigger oversold-band + oversold-RSI fades


def test_bollinger_rsi_prefix_invariance() -> None:
    spec = BollingerRsiSpec()
    full = spec.generate_signals(_CANDLES)
    for k in range(1, len(_CANDLES) + 1):
        prefix = spec.generate_signals(_CANDLES[:k])
        assert [(s.side, s.strength) for s in prefix] == [(s.side, s.strength) for s in full[:k]]


def test_factory_reads_params_and_validates() -> None:
    spec = bollinger_rsi_spec({"bb_num_std": 2.5, "rsi_oversold": 25.0})
    assert spec.bb_num_std == 2.5 and spec.rsi_oversold == 25.0
    assert spec.name == "bollinger_rsi" and spec.bb_period == 20 and spec.rsi_period == 14
    assert bollinger_rsi_spec({}).bb_num_std == 2.0 and bollinger_rsi_spec({}).rsi_oversold == 30.0
    with pytest.raises(ValueError, match="rsi_oversold"):
        BollingerRsiSpec(rsi_oversold=60.0)
    assert math.isclose(bollinger_rsi_spec({}).bb_num_std, 2.0)
