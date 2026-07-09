"""Tests for the Donchian breakout spec (P3.6): global multi-session channel breakout.

Pins the break entry (LONG above the prior high, SHORT below the prior low), the
no-signal-inside-channel case, the GLOBAL (spans-the-overnight-gap) property that
distinguishes it from the dead P3.2 intraday breakout, and -- the hard precondition -- the
NO-LOOKAHEAD prefix-invariance property.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from lab.core.types import BarInterval, Candle, Side
from lab.research.strategies.donchian_breakout import DonchianBreakoutSpec, donchian_breakout_spec

IST = ZoneInfo("Asia/Kolkata")
DAY = datetime(2024, 7, 15, 9, 15, tzinfo=IST)
D2 = DAY + timedelta(days=1)


def _bar(ts: datetime, high: float, low: float, close: float) -> Candle:
    return Candle("X", BarInterval.MIN_5, ts, close, high, low, close, 1000)


def _day(rows: list[tuple[float, float, float]], day: datetime) -> list[Candle]:
    return [_bar(day + timedelta(minutes=5 * i), h, low, c) for i, (h, low, c) in enumerate(rows)]


def test_longs_breakout_of_prior_global_high() -> None:
    spec = DonchianBreakoutSpec(channel_lookback=3)
    # Prior 3-bar high (excluding bar 3) = 12; bar 3 closes 13 > 12 -> LONG.
    candles = _day([(10, 9, 10), (12, 11, 12), (11, 10, 11), (13, 12, 13)], DAY)
    signals = spec.generate_signals(candles)
    assert len(signals) == 1 and signals[0].asof == candles[3].timestamp
    assert signals[0].side is Side.LONG and signals[0].strength == 1.0


def test_shorts_breakout_of_prior_global_low() -> None:
    spec = DonchianBreakoutSpec(channel_lookback=3)
    # Prior 3-bar low (excluding bar 3) = 9; bar 3 closes 8 < 9 -> SHORT.
    candles = _day([(11, 10, 11), (12, 11, 12), (13, 9, 10), (9, 8, 8)], DAY)
    signals = spec.generate_signals(candles)
    assert len(signals) == 1 and signals[0].side is Side.SHORT and signals[0].strength == 1.0


def test_no_signal_inside_channel() -> None:
    spec = DonchianBreakoutSpec(channel_lookback=3)
    # Bar 3 closes 11, inside [prior_low 9, prior_high 12] -> no breakout, no signal.
    candles = _day([(10, 9, 10), (12, 11, 12), (11, 10, 11), (11.5, 10.5, 11)], DAY)
    assert spec.generate_signals(candles) == []


def test_channel_is_global_a_day2_bar_breaks_a_prior_session_high() -> None:
    spec = DonchianBreakoutSpec(channel_lookback=3)
    # Day 2's first bar closes 13, breaking DAY-1's global 3-bar high (12). Only possible
    # because the channel is GLOBAL (does not reset) -- an intraday-reset spec would see NaN.
    candles = _day([(10, 9, 10), (12, 11, 12), (11, 10, 11)], DAY) + _day([(13, 12, 13)], D2)
    signals = spec.generate_signals(candles)
    assert any(s.asof == candles[3].timestamp and s.side is Side.LONG for s in signals)


def test_no_lookahead_prefix_invariance() -> None:
    """HARD PRECONDITION: a breakout signal at bar i must not change when later bars appear."""
    spec = DonchianBreakoutSpec(channel_lookback=3)
    candles = _day(
        [(10, 9, 10), (12, 11, 12), (11, 10, 11), (13, 12, 13), (9, 8, 8), (14, 13, 14)], DAY
    )
    full = spec.generate_signals(candles)
    for k in range(1, len(candles) + 1):
        prefix = spec.generate_signals(candles[:k])
        cutoff = candles[k - 1].timestamp
        expected = [s for s in full if s.asof <= cutoff]
        assert [(s.asof, s.side, s.strength) for s in prefix] == [
            (s.asof, s.side, s.strength) for s in expected
        ]


def test_factory_reads_params_and_validates() -> None:
    spec = donchian_breakout_spec({"channel_lookback": 40})
    assert spec.channel_lookback == 40
    assert spec.name == "donchian_breakout"
    assert spec.interval is BarInterval.MIN_5
    with pytest.raises(ValueError, match="channel_lookback"):
        DonchianBreakoutSpec(channel_lookback=1)


def test_factory_defaults_when_params_absent() -> None:
    assert donchian_breakout_spec({}).channel_lookback == 55
