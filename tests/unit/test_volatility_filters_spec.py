"""Tests for the volatility-filter specs (P3.8): C1 expansion-breakout, C2 contraction-reversion.

Verifies each strategy's REGIME GATING (C1 breakouts only in expanding vol, C2 fades entered
only in contracting vol) and that the gate is not a no-op, plus point-in-time prefix
invariance and the factories.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from lab.core.types import BarInterval, Candle
from lab.data.features.indicators import atr_ratio, intraday_donchian
from lab.data.features.ohlcv import OHLCV
from lab.research.strategies.volatility_filters import (
    VolContractionReversionSpec,
    VolExpansionBreakoutSpec,
    vol_contraction_reversion_spec,
    vol_expansion_breakout_spec,
)

IST = ZoneInfo("Asia/Kolkata")


def _vol_varying() -> list[Candle]:
    # calm -> volatile -> calm, so ATR(20)/ATR(100) crosses 1 in both directions and breakouts
    # / z-extremes occur in each regime (one day; the intraday indicators reset per day).
    day = datetime(2024, 7, 15, 9, 15, tzinfo=IST)
    rng = np.random.default_rng(7)
    close = 100.0
    rows: list[tuple[float, float, float]] = []
    for n, vol in ((60, 0.15), (60, 1.2), (60, 0.15)):
        for _ in range(n):
            close += float(rng.normal(0, vol))
            rng_range = abs(float(rng.normal(0, vol))) + 0.05
            rows.append((close, close + rng_range, close - rng_range))
    return [
        Candle("X", BarInterval.MIN_5, day + timedelta(minutes=5 * i), c, hi, lo, c, 1000)
        for i, (c, hi, lo) in enumerate(rows)
    ]


_CANDLES = _vol_varying()
_IDX = {c.timestamp: i for i, c in enumerate(_CANDLES)}


def test_c1_signals_only_in_expanding_regime_and_gate_is_not_a_noop() -> None:
    spec = VolExpansionBreakoutSpec()
    ohlcv = OHLCV.from_candles(_CANDLES)
    ratio = atr_ratio(ohlcv, spec.atr_short, spec.atr_long)
    upper, lower = intraday_donchian(ohlcv, spec.breakout_lookback)
    signals = spec.generate_signals(_CANDLES)
    for s in signals:
        assert float(ratio[_IDX[s.asof]]) > 1.0  # every signal is in the expanding-vol regime
    ungated = sum(
        1
        for i, c in enumerate(_CANDLES)
        if (math.isfinite(float(upper[i])) and c.close > float(upper[i]))
        or (math.isfinite(float(lower[i])) and c.close < float(lower[i]))
    )
    assert 0 < len(signals) < ungated  # gating suppressed SOME breakouts, not none and not all


def test_c2_fades_entered_only_in_contracting_regime() -> None:
    spec = VolContractionReversionSpec()
    ratio = atr_ratio(OHLCV.from_candles(_CANDLES), spec.atr_short, spec.atr_long)
    signals = spec.generate_signals(_CANDLES)
    entries = 0
    prev_strength = 0.0
    for s in signals:
        if s.strength == 1.0 and prev_strength == 0.0:  # entered a fade from flat
            assert float(ratio[_IDX[s.asof]]) < 1.0  # only while volatility is contracting
            entries += 1
        prev_strength = s.strength
    assert entries > 0  # some fades were entered in the contracting regime


def test_c1_prefix_invariance() -> None:
    spec = VolExpansionBreakoutSpec()
    full = spec.generate_signals(_CANDLES)
    for k in range(1, len(_CANDLES) + 1):
        prefix = spec.generate_signals(_CANDLES[:k])
        cutoff = _CANDLES[k - 1].timestamp
        expected = [s for s in full if s.asof <= cutoff]
        assert [(s.asof, s.side, s.strength) for s in prefix] == [
            (s.asof, s.side, s.strength) for s in expected
        ]


def test_c2_prefix_invariance() -> None:
    spec = VolContractionReversionSpec()
    full = spec.generate_signals(_CANDLES)
    for k in range(1, len(_CANDLES) + 1):
        prefix = spec.generate_signals(_CANDLES[:k])
        assert [(s.side, s.strength) for s in prefix] == [(s.side, s.strength) for s in full[:k]]


def test_factories_read_params_and_validate() -> None:
    c1 = vol_expansion_breakout_spec({"breakout_lookback": 15, "atr_long": 80})
    assert c1.breakout_lookback == 15 and c1.atr_long == 80 and c1.atr_short == 20
    assert c1.name == "vol_expansion_breakout" and c1.interval is BarInterval.MIN_5
    assert vol_expansion_breakout_spec({}).breakout_lookback == 20
    assert vol_expansion_breakout_spec({}).atr_long == 100
    c2 = vol_contraction_reversion_spec({"entry_z": 2.5, "atr_long": 120})
    assert c2.entry_z == 2.5 and c2.atr_long == 120 and c2.exit_z == 0.5 and c2.lookback == 20
    assert c2.name == "vol_contraction_reversion"
    with pytest.raises(ValueError, match="atr_short < atr_long"):
        VolExpansionBreakoutSpec(atr_short=100, atr_long=20)
    with pytest.raises(ValueError, match="exit_z"):
        VolContractionReversionSpec(entry_z=0.5, exit_z=1.0)
    with pytest.raises(ValueError, match="atr_short < atr_long"):
        VolContractionReversionSpec(atr_short=100, atr_long=20)
