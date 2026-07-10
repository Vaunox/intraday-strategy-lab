"""Tests for P4.5 pivot_macd: a MACD crossover near a classic pivot level.

Covers the no-lookahead prefix-invariance precondition (hard gate), the confluence gating (any
entry IS near the level with the confirming crossover), and the factory. The confluence is
deliberately restrictive (a possible degeneracy) — its real trade count is measured in the §6
landscape, so the behaviour test only asserts VALIDITY of whatever fires, not a firing count.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from lab.core.types import BarInterval, Candle, Side
from lab.data.features.indicators import classic_pivot_levels, macd
from lab.data.features.ohlcv import OHLCV
from lab.research.strategies.pivot_macd import PivotMacdSpec, pivot_macd_spec

IST = ZoneInfo("Asia/Kolkata")
_BARS_PER_DAY = 40


def _oscillating() -> list[Candle]:
    day0 = datetime(2024, 7, 1, 9, 15, tzinfo=IST)
    out: list[Candle] = []
    for i in range(200):  # 5 days x 40 bars, swinging so price passes near the prior-day pivots
        c = 100.0 + 3.0 * math.sin(i / 6.0) + 0.5 * math.sin(i / 1.3)
        d, b = divmod(i, _BARS_PER_DAY)
        ts = day0 + timedelta(days=d, minutes=5 * b)
        out.append(Candle("X", BarInterval.MIN_5, ts, c, c + 0.4, c - 0.4, c, 1000))
    return out


_CANDLES = _oscillating()


def test_any_entry_is_a_valid_level_plus_crossover_confluence() -> None:
    spec = PivotMacdSpec(entry_band=0.01)  # a wider band to encourage firing on the fixture
    ohlcv = OHLCV.from_candles(_CANDLES)
    r1, s1 = classic_pivot_levels(ohlcv)
    line, sig, _hist = macd(ohlcv, 12, 26, 9)
    for s in spec.generate_signals(_CANDLES):
        i = next(j for j, c in enumerate(_CANDLES) if c.timestamp == s.asof)
        close = float(_CANDLES[i].close)
        if s.side is Side.LONG:  # near support with a bullish crossover
            assert abs(close - float(s1[i])) <= spec.entry_band * float(s1[i])
            assert float(line[i - 1]) <= float(sig[i - 1]) and float(line[i]) > float(sig[i])
        else:  # near resistance with a bearish crossover
            assert abs(close - float(r1[i])) <= spec.entry_band * float(r1[i])
            assert float(line[i - 1]) >= float(sig[i - 1]) and float(line[i]) < float(sig[i])


def test_pivot_macd_prefix_invariance() -> None:
    spec = PivotMacdSpec()
    full = spec.generate_signals(_CANDLES)
    for k in range(1, len(_CANDLES) + 1):
        cutoff = _CANDLES[k - 1].timestamp
        prefix = spec.generate_signals(_CANDLES[:k])
        expected = [s for s in full if s.asof <= cutoff]
        assert [(s.asof, s.side) for s in prefix] == [(s.asof, s.side) for s in expected]


def test_factory_reads_params_and_validates() -> None:
    spec = pivot_macd_spec({"entry_band": 0.005})
    assert spec.entry_band == 0.005 and spec.name == "pivot_macd"
    assert pivot_macd_spec({}).entry_band == 0.002
    with pytest.raises(ValueError, match="entry_band"):
        PivotMacdSpec(entry_band=0.0)
