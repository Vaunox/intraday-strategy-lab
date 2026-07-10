"""Tests for the opening-range-breakout spec (P3.11): break of the first-N-min range.

Verifies entries fire only AFTER the opening-range window closes and only on a genuine break,
the point-in-time prefix invariance (the hard no-lookahead precondition), and the factory.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from lab.core.types import BarInterval, Candle, Side
from lab.research.strategies.opening_range_breakout import (
    OpeningRangeBreakoutSpec,
    opening_range_breakout_spec,
)

IST = ZoneInfo("Asia/Kolkata")


def _mk(days: list[list[tuple[float, float, float]]]) -> list[Candle]:
    """Lay per-day (close, high, low) rows onto 5-min IST sessions."""
    day0 = datetime(2024, 7, 1, 9, 15, tzinfo=IST)
    out: list[Candle] = []
    for d, rows in enumerate(days):
        for b, (c, hi, lo) in enumerate(rows):
            ts = day0 + timedelta(days=d, minutes=5 * b)
            out.append(Candle("X", BarInterval.MIN_5, ts, c, hi, lo, c, 1000))
    return out


# 6 range bars (09:15-09:40, the 30-min opening range = [99, 101]) then a break; day 3 no break.
_RANGE = [(100.0, 101.0, 99.0)] * 6
_CANDLES = _mk(
    [
        _RANGE + [(102.0, 102.5, 101.5)] + [(103.0, 103.5, 102.5)] * 3,  # break UP
        _RANGE + [(98.0, 98.5, 97.5)] + [(97.0, 97.5, 96.5)] * 3,  # break DOWN
        _RANGE + [(100.0, 101.0, 99.0)] * 4,  # stays inside -> no break
    ]
)


def test_entries_only_after_window_close_and_on_a_genuine_break() -> None:
    spec = OpeningRangeBreakoutSpec()  # 30-min window
    signals = spec.generate_signals(_CANDLES)
    assert [s.side for s in signals] == [Side.LONG, Side.SHORT]  # day1 up, day2 down, day3 none
    for s in signals:
        # entered only once the opening-range window (30 min) had closed
        session_open = s.asof.replace(hour=9, minute=15, second=0, microsecond=0)
        assert s.asof >= session_open + timedelta(minutes=30)


def test_opening_range_breakout_prefix_invariance() -> None:
    # HARD no-lookahead precondition (sparse form): signals up to each cutoff are stable.
    spec = OpeningRangeBreakoutSpec()
    full = spec.generate_signals(_CANDLES)
    for k in range(1, len(_CANDLES) + 1):
        cutoff = _CANDLES[k - 1].timestamp
        prefix = spec.generate_signals(_CANDLES[:k])
        expected = [s for s in full if s.asof <= cutoff]
        assert [(s.asof, s.side) for s in prefix] == [(s.asof, s.side) for s in expected]


def test_factory_reads_params_and_validates() -> None:
    spec = opening_range_breakout_spec({"opening_range_minutes": 15, "break_buffer": 0.002})
    assert spec.opening_range_minutes == 15 and spec.break_buffer == 0.002
    assert spec.name == "opening_range_breakout" and spec.interval is BarInterval.MIN_5
    assert opening_range_breakout_spec({}).opening_range_minutes == 30
    assert opening_range_breakout_spec({}).break_buffer == 0.001
    with pytest.raises(ValueError, match="break_buffer"):
        OpeningRangeBreakoutSpec(break_buffer=-0.001)
