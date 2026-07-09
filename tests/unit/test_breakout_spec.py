"""Tests for the volume-filtered breakout spec (P3.2): break direction, the volume
filter, no-break silence, point-in-time prefix invariance, and the factory.

These verify the spec's signal logic; they do NOT run the study/kill-gate (that
awaits sign-off on the pre-registration).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from lab.core.types import BarInterval, Candle, Side
from lab.research.strategies.breakout import BreakoutSpec, breakout_spec

IST = ZoneInfo("Asia/Kolkata")

# Small lookbacks so a handful of bars exercise the logic (defaults are 20/20).
SPEC = BreakoutSpec(breakout_lookback=3, volume_period=3, volume_mult=1.5)


def _bar(i: int, high: float, low: float, close: float, volume: int) -> Candle:
    ts = datetime(2024, 7, 15, 9, 15, tzinfo=IST) + timedelta(minutes=5 * i)
    return Candle("RELIANCE", BarInterval.MIN_5, ts, close, high, low, close, volume)


def _range3() -> list[Candle]:
    # Three bars establishing a [99.5, 100.5] range on baseline volume 1000.
    return [_bar(i, 100.5, 99.5, 100.0, 1000) for i in range(3)]


def test_filtered_upside_break_goes_long() -> None:
    # Bar 3 closes 101.5 > prior-3-bar high 100.5, on 2x volume -> filtered LONG.
    candles = [*_range3(), _bar(3, 102.0, 100.0, 101.5, 2000)]
    signals = SPEC.generate_signals(candles)
    assert len(signals) == 1
    assert signals[0].side is Side.LONG
    assert signals[0].strength == 1.0
    assert signals[0].asof == candles[3].timestamp


def test_filtered_downside_break_goes_short() -> None:
    # Bar 3 closes 98.5 < prior-3-bar low 99.5, on 2x volume -> filtered SHORT.
    candles = [*_range3(), _bar(3, 100.0, 98.0, 98.5, 2000)]
    signals = SPEC.generate_signals(candles)
    assert len(signals) == 1
    assert signals[0].side is Side.SHORT


def test_volume_filter_blocks_a_low_volume_break() -> None:
    # Same price break as the LONG case, but on baseline volume (rel-vol 1.0 <= 1.5).
    candles = [*_range3(), _bar(3, 102.0, 100.0, 101.5, 1000)]
    assert SPEC.generate_signals(candles) == []


def test_no_break_emits_no_signal_even_on_high_volume() -> None:
    # Close 100.0 stays inside the range; high volume alone is not a break.
    candles = [*_range3(), _bar(3, 100.5, 99.5, 100.0, 5000)]
    assert SPEC.generate_signals(candles) == []


def test_point_in_time_prefix_invariance() -> None:
    """No lookahead: signals for any prefix equal the full-series signals whose bar
    is within that prefix (breaks are sparse, so compare by timestamp)."""
    candles = [
        *_range3(),
        _bar(3, 102.0, 100.0, 101.5, 2000),  # LONG break
        _bar(4, 101.5, 100.5, 101.0, 1000),  # low rel-vol -> no signal
        _bar(5, 101.0, 98.5, 99.0, 3000),  # SHORT break on volume
    ]
    full = SPEC.generate_signals(candles)
    assert [s.side for s in full] == [Side.LONG, Side.SHORT]

    for k in range(1, len(candles) + 1):
        prefix_asofs = {c.timestamp for c in candles[:k]}
        expected = [(s.asof, s.side) for s in full if s.asof in prefix_asofs]
        got = [(s.asof, s.side) for s in SPEC.generate_signals(candles[:k])]
        assert got == expected


def test_factory_reads_params_and_validates() -> None:
    spec = breakout_spec({"breakout_lookback": 25.0, "volume_mult": 2.0})
    assert spec.breakout_lookback == 25  # rounded to int
    assert spec.volume_mult == 2.0
    assert spec.volume_period == 20  # fixed default, not swept
    assert spec.name == "breakout"
    assert spec.interval is BarInterval.MIN_5

    with pytest.raises(ValueError, match="breakout_lookback"):
        BreakoutSpec(breakout_lookback=1)
    with pytest.raises(ValueError, match="volume_mult"):
        BreakoutSpec(volume_mult=0.0)


def test_factory_defaults_when_params_absent() -> None:
    spec = breakout_spec({})
    assert spec.breakout_lookback == 20
    assert spec.volume_mult == 1.5
