"""Tests for the mean-reversion spec (P3.3): intraday-reset z-score fade.

Verifies fade direction, wide-entry / narrow-exit hysteresis, daily reset, point-in-time
prefix invariance, and the parameter factory. Small test params (lookback 5, entry_z 1.0,
exit_z 0.3) keep the crossings tractable; the frozen prereg params (2.0 / 0.5 / 20) are
pinned in test_registry.py. Does NOT run the study/kill-gate.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from lab.core.types import BarInterval, Candle, Side
from lab.data.features import indicators
from lab.data.features.ohlcv import OHLCV
from lab.research.strategies.mean_reversion import MeanReversionSpec, mean_reversion_spec

IST = ZoneInfo("Asia/Kolkata")


def _bar(ts: datetime, price: float) -> Candle:
    return Candle("RELIANCE", BarInterval.MIN_5, ts, price, price + 0.1, price - 0.1, price, 1000)


def _day(prices: list[float], day: datetime) -> list[Candle]:
    return [_bar(day + timedelta(minutes=5 * i), p) for i, p in enumerate(prices)]


def test_fades_stretch_above_mean_short_holds_then_exits_on_reversion() -> None:
    spec = MeanReversionSpec(entry_z=1.0, exit_z=0.3, lookback=5)
    day = datetime(2024, 7, 15, 9, 15, tzinfo=IST)
    candles = _day([100.0, 100.0, 100.0, 100.0, 100.0, 103.0, 102.0, 100.5, 100.0], day)
    z = indicators.intraday_zscore(OHLCV.from_candles(candles), spec.lookback)
    signals = spec.generate_signals(candles)

    assert z[5] > 1.0  # decisive stretch above the intraday mean
    assert signals[5].side is Side.SHORT and signals[5].strength == 1.0  # fade the stretch
    assert signals[6].side is Side.SHORT and signals[6].strength == 1.0  # hold through the band
    assert z[7] <= 0.3  # reverted to within the exit band of the mean
    assert signals[7].strength == 0.0  # exited


def test_fades_stretch_below_mean_long() -> None:
    spec = MeanReversionSpec(entry_z=1.0, exit_z=0.3, lookback=5)
    day = datetime(2024, 7, 15, 9, 15, tzinfo=IST)
    candles = _day([100.0, 100.0, 100.0, 100.0, 100.0, 97.0, 98.0, 99.5, 100.0], day)
    z = indicators.intraday_zscore(OHLCV.from_candles(candles), spec.lookback)
    signals = spec.generate_signals(candles)

    assert z[5] < -1.0
    assert signals[5].side is Side.LONG and signals[5].strength == 1.0


def test_position_resets_flat_each_day() -> None:
    spec = MeanReversionSpec(entry_z=1.0, exit_z=0.3, lookback=5)
    d1 = datetime(2024, 7, 15, 9, 15, tzinfo=IST)
    d2 = datetime(2024, 7, 16, 9, 15, tzinfo=IST)
    # Day 1 ends holding a short (sustained above the mean); day 2 must open flat.
    candles = _day([100.0, 100.0, 100.0, 100.0, 100.0, 103.0, 103.0, 103.0], d1) + _day(
        [100.0, 100.0, 100.0], d2
    )
    signals = spec.generate_signals(candles)

    assert signals[7].side is Side.SHORT and signals[7].strength == 1.0  # still short at day-1 end
    assert signals[8].strength == 0.0  # first bar of day 2 -> flat, no overnight carry


def test_point_in_time_prefix_invariance() -> None:
    """No lookahead: signals for any prefix equal the prefix of the full-series signals."""
    spec = MeanReversionSpec(entry_z=1.0, exit_z=0.3, lookback=5)
    day = datetime(2024, 7, 15, 9, 15, tzinfo=IST)
    candles = _day([100.0, 101.0, 99.0, 100.0, 100.0, 104.0, 101.0, 98.0, 100.0, 100.0], day)
    full = spec.generate_signals(candles)

    for k in range(1, len(candles) + 1):
        prefix = spec.generate_signals(candles[:k])
        assert [(s.side, s.strength) for s in prefix] == [(s.side, s.strength) for s in full[:k]]


def test_factory_reads_params_and_validates_band() -> None:
    spec = mean_reversion_spec({"entry_z": 2.5, "exit_z": 0.7, "lookback": 30})
    assert spec.entry_z == 2.5
    assert spec.exit_z == 0.7
    assert spec.lookback == 30
    assert spec.name == "mean_reversion"
    assert spec.interval is BarInterval.MIN_5

    # exit_z must sit strictly inside entry_z
    with pytest.raises(ValueError, match="exit_z"):
        mean_reversion_spec({"entry_z": 0.5, "exit_z": 1.0})


def test_factory_defaults_when_params_absent() -> None:
    spec = mean_reversion_spec({})
    assert spec.entry_z == 2.0
    assert spec.exit_z == 0.5
    assert spec.lookback == 20


def test_spec_rejects_degenerate_lookback() -> None:
    with pytest.raises(ValueError, match="lookback"):
        MeanReversionSpec(lookback=1)
