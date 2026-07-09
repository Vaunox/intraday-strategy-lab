"""Tests for the VWAP specs (P3.1): the mean-reversion fade (V1) and the decisive
cross (V2) -- entry direction, hysteresis / band hold, daily reset, point-in-time
prefix invariance, and the parameter factories.

These verify the specs' signal logic; they do NOT run the study/kill-gate.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from lab.core.types import BarInterval, Candle, Side
from lab.data.features.indicators import vwap_deviation
from lab.data.features.ohlcv import OHLCV
from lab.research.strategies.vwap import (
    VwapCrossSpec,
    VwapMeanReversionSpec,
    vwap_cross_spec,
    vwap_mean_reversion_spec,
)

IST = ZoneInfo("Asia/Kolkata")


def _bar(ts: datetime, price: float, volume: int = 1000) -> Candle:
    # H/L straddle C by a hair so the typical price (H+L+C)/3 == close; VWAP is then
    # the (equal-)volume-weighted mean of close, which the tests can reason about.
    return Candle("RELIANCE", BarInterval.MIN_5, ts, price, price + 0.1, price - 0.1, price, volume)


def _day(prices: list[float], day: datetime) -> list[Candle]:
    return [_bar(day + timedelta(minutes=5 * i), p) for i, p in enumerate(prices)]


def test_fades_extension_above_vwap_short_then_exits_on_reversion() -> None:
    spec = VwapMeanReversionSpec(entry_threshold=0.004, exit_threshold=0.001)
    day = datetime(2024, 7, 15, 9, 15, tzinfo=IST)
    candles = _day([100.0, 100.0, 100.0, 105.0, 105.0, 101.0, 100.0], day)
    dev = vwap_deviation(OHLCV.from_candles(candles))
    signals = spec.generate_signals(candles)

    # Engineered path sanity: bar 3 is well above VWAP; by bar 5 price has reverted.
    assert dev[3] > 0.004
    assert dev[5] <= 0.001

    assert signals[0].strength == 0.0  # at VWAP -> flat
    assert signals[3].side is Side.SHORT and signals[3].strength == 1.0  # fade the extension
    assert signals[4].side is Side.SHORT and signals[4].strength == 1.0  # hold through the band
    assert signals[5].strength == 0.0  # reverted within exit band -> flat


def test_fades_extension_below_vwap_long() -> None:
    spec = VwapMeanReversionSpec(entry_threshold=0.004, exit_threshold=0.001)
    day = datetime(2024, 7, 15, 9, 15, tzinfo=IST)
    candles = _day([100.0, 100.0, 100.0, 95.0, 95.0], day)
    dev = vwap_deviation(OHLCV.from_candles(candles))
    signals = spec.generate_signals(candles)

    assert dev[3] < -0.004
    assert signals[3].side is Side.LONG and signals[3].strength == 1.0


def test_position_resets_flat_each_day() -> None:
    spec = VwapMeanReversionSpec()
    d1 = datetime(2024, 7, 15, 9, 15, tzinfo=IST)
    d2 = datetime(2024, 7, 16, 9, 15, tzinfo=IST)
    # Day 1 ends holding a short (still extended above VWAP); day 2 must open flat.
    candles = _day([100.0, 100.0, 105.0, 105.0], d1) + _day([100.0, 100.0], d2)
    signals = spec.generate_signals(candles)

    assert signals[3].side is Side.SHORT and signals[3].strength == 1.0  # still short at day-1 end
    assert signals[4].strength == 0.0  # first bar of day 2 -> flat, no overnight carry


def test_point_in_time_prefix_invariance() -> None:
    """No lookahead: signals for any prefix equal the prefix of the full-series signals."""
    spec = VwapMeanReversionSpec()
    day = datetime(2024, 7, 15, 9, 15, tzinfo=IST)
    candles = _day([100.0, 101.0, 100.5, 105.0, 104.0, 100.0, 99.0, 100.0], day)
    full = spec.generate_signals(candles)

    for k in range(1, len(candles) + 1):
        prefix = spec.generate_signals(candles[:k])
        assert [(s.side, s.strength) for s in prefix] == [(s.side, s.strength) for s in full[:k]]


def test_factory_reads_params_and_validates_band() -> None:
    spec = vwap_mean_reversion_spec({"entry_threshold": 0.006, "exit_threshold": 0.002})
    assert spec.entry_threshold == 0.006
    assert spec.exit_threshold == 0.002
    assert spec.name == "vwap_mean_reversion"
    assert spec.interval is BarInterval.MIN_5

    # exit must sit strictly inside entry
    with pytest.raises(ValueError, match="exit_threshold"):
        vwap_mean_reversion_spec({"entry_threshold": 0.002, "exit_threshold": 0.004})


def test_factory_defaults_when_params_absent() -> None:
    spec = vwap_mean_reversion_spec({})
    assert spec.entry_threshold == 0.004
    assert spec.exit_threshold == 0.001


# --- V2: decisive VWAP cross (trend / continuation) ------------------------------- #


def test_cross_goes_long_on_decisive_upside_cross_and_holds_through_band() -> None:
    spec = VwapCrossSpec(cross_threshold=0.002)
    day = datetime(2024, 7, 15, 9, 15, tzinfo=IST)
    candles = _day([100.0, 100.0, 100.0, 101.0, 101.0, 100.4], day)
    dev = vwap_deviation(OHLCV.from_candles(candles))
    signals = spec.generate_signals(candles)

    # Engineered path: flat at VWAP, decisive upside cross at bar 3, back near VWAP by bar 5.
    assert dev[0] == pytest.approx(0.0)
    assert dev[3] > 0.002
    assert -0.002 <= dev[5] <= 0.002

    assert signals[0].strength == 0.0  # at VWAP -> flat
    assert signals[3].side is Side.LONG and signals[3].strength == 1.0  # long the decisive cross
    assert signals[4].side is Side.LONG and signals[4].strength == 1.0  # ride the continuation
    assert signals[5].side is Side.LONG and signals[5].strength == 1.0  # hold through the band


def test_cross_goes_short_on_decisive_downside_cross() -> None:
    spec = VwapCrossSpec(cross_threshold=0.002)
    day = datetime(2024, 7, 15, 9, 15, tzinfo=IST)
    candles = _day([100.0, 100.0, 100.0, 99.0], day)
    dev = vwap_deviation(OHLCV.from_candles(candles))
    signals = spec.generate_signals(candles)

    assert dev[3] < -0.002
    assert signals[3].side is Side.SHORT and signals[3].strength == 1.0


def test_cross_holds_through_small_deviation_no_whipsaw() -> None:
    # Once long, a small deviation INSIDE the +/-band must not flip or flatten the position.
    spec = VwapCrossSpec(cross_threshold=0.002)
    day = datetime(2024, 7, 15, 9, 15, tzinfo=IST)
    candles = _day([100.0, 100.0, 100.0, 101.0, 100.1], day)
    dev = vwap_deviation(OHLCV.from_candles(candles))
    signals = spec.generate_signals(candles)

    assert signals[3].side is Side.LONG and signals[3].strength == 1.0  # entered long on the cross
    assert -0.002 < dev[4] < 0.002  # bar 4 sits inside the whipsaw band (slightly negative here)
    assert signals[4].side is Side.LONG and signals[4].strength == 1.0  # still long -- no whipsaw


def test_cross_flips_on_opposite_decisive_cross() -> None:
    spec = VwapCrossSpec(cross_threshold=0.002)
    day = datetime(2024, 7, 15, 9, 15, tzinfo=IST)
    candles = _day([100.0, 100.0, 100.0, 101.0, 101.0, 98.0], day)
    dev = vwap_deviation(OHLCV.from_candles(candles))
    signals = spec.generate_signals(candles)

    assert signals[3].side is Side.LONG and signals[3].strength == 1.0  # long the upside cross
    assert dev[5] < -0.002  # a decisive downside cross (price crossed back through VWAP)
    assert signals[5].side is Side.SHORT and signals[5].strength == 1.0  # flips short


def test_cross_position_resets_flat_each_day() -> None:
    spec = VwapCrossSpec()
    d1 = datetime(2024, 7, 15, 9, 15, tzinfo=IST)
    d2 = datetime(2024, 7, 16, 9, 15, tzinfo=IST)
    # Day 1 ends holding a long (still above VWAP); day 2 must open flat.
    candles = _day([100.0, 100.0, 101.0, 101.0], d1) + _day([100.0, 100.0], d2)
    signals = spec.generate_signals(candles)

    assert signals[3].side is Side.LONG and signals[3].strength == 1.0  # still long at day-1 end
    assert signals[4].strength == 0.0  # first bar of day 2 -> flat, no overnight carry


def test_cross_point_in_time_prefix_invariance() -> None:
    """No lookahead: cross signals for any prefix equal the prefix of the full signals."""
    spec = VwapCrossSpec()
    day = datetime(2024, 7, 15, 9, 15, tzinfo=IST)
    candles = _day([100.0, 101.0, 100.5, 102.0, 99.0, 100.0, 101.5, 100.0], day)
    full = spec.generate_signals(candles)

    for k in range(1, len(candles) + 1):
        prefix = spec.generate_signals(candles[:k])
        assert [(s.side, s.strength) for s in prefix] == [(s.side, s.strength) for s in full[:k]]


def test_cross_factory_reads_params_and_validates() -> None:
    spec = vwap_cross_spec({"cross_threshold": 0.003})
    assert spec.cross_threshold == 0.003
    assert spec.name == "vwap_cross"
    assert spec.interval is BarInterval.MIN_5

    # the whipsaw-guard band must be strictly positive
    with pytest.raises(ValueError, match="cross_threshold"):
        vwap_cross_spec({"cross_threshold": 0.0})


def test_cross_factory_defaults_when_params_absent() -> None:
    spec = vwap_cross_spec({})
    assert spec.cross_threshold == 0.002
