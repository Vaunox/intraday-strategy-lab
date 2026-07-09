"""Tests for the reversal spec (P3.4): swing-failure (failed-breakout) fade.

Pins the failed-break entry logic, the load-bearing ``break_buffer`` (a marginal graze must
NOT signal), the break-holds no-signal case, the day-boundary reset (no cross-gap swing
level), and -- crucially -- the NO-LOOKAHEAD prefix-invariance property. That last test is
the study's hard precondition: a swing signal that peeked at future bars would manufacture a
phantom edge, so if it fails the study must not run.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from lab.core.types import BarInterval, Candle, Side
from lab.research.strategies.reversal import ReversalSpec, reversal_spec

IST = ZoneInfo("Asia/Kolkata")


def _bar(ts: datetime, high: float, low: float, close: float) -> Candle:
    # open is unused by the spec (only prior swing level + this bar's H/L/close matter).
    return Candle("X", BarInterval.MIN_5, ts, close, high, low, close, 1000)


def _day(rows: list[tuple[float, float, float]], day: datetime) -> list[Candle]:
    return [_bar(day + timedelta(minutes=5 * i), h, low, c) for i, (h, low, c) in enumerate(rows)]


DAY = datetime(2024, 7, 15, 9, 15, tzinfo=IST)


def test_shorts_a_failed_upside_breakout() -> None:
    spec = ReversalSpec(swing_lookback=3, break_buffer=0.001)
    # prior swing high over bars 0..2 = 102; bar 3 pokes to 103 (> 102.102) but closes 101.5 < 102.
    candles = _day([(100, 99, 100), (101, 99.5, 101), (102, 100, 102), (103, 101, 101.5)], DAY)
    signals = spec.generate_signals(candles)
    assert len(signals) == 1
    assert signals[0].asof == candles[3].timestamp
    assert signals[0].side is Side.SHORT and signals[0].strength == 1.0


def test_longs_a_failed_downside_breakout() -> None:
    spec = ReversalSpec(swing_lookback=3, break_buffer=0.001)
    # prior swing low over bars 0..2 = 96; bar 3 pokes to 95 (< 95.904) but closes 96.5 > 96.
    candles = _day([(100, 98, 98), (99.5, 97, 97), (98, 96, 96), (97, 95, 96.5)], DAY)
    signals = spec.generate_signals(candles)
    assert len(signals) == 1
    assert signals[0].side is Side.LONG and signals[0].strength == 1.0


def test_break_buffer_is_load_bearing_marginal_graze_does_not_signal() -> None:
    # Bar 3 pokes to 102.05 -- above the prior swing high 102, but by < break_buffer (0.1%),
    # so NOT a genuine trap-inducing break -> no signal. The SAME bar WOULD signal at buffer 0.
    candles = _day([(100, 99, 100), (101, 99.5, 101), (102, 100, 102), (102.05, 101, 101.5)], DAY)
    # With the 0.1% buffer, the marginal graze (102.05, ~0.05% above 102) is NOT a genuine break.
    assert ReversalSpec(swing_lookback=3, break_buffer=0.001).generate_signals(candles) == []
    # With no buffer, the marginal graze DOES fail-and-signal -> proves the buffer suppressed it.
    zero_buf = ReversalSpec(swing_lookback=3, break_buffer=0.0).generate_signals(candles)
    assert len(zero_buf) == 1 and zero_buf[0].side is Side.SHORT


def test_break_that_holds_does_not_signal() -> None:
    spec = ReversalSpec(swing_lookback=3, break_buffer=0.001)
    # Bar 3 pokes to 103 AND closes 103 (above the swing high 102): the break HELD -> no reversal.
    candles = _day([(100, 99, 100), (101, 99.5, 101), (102, 100, 102), (103, 101, 103)], DAY)
    assert spec.generate_signals(candles) == []


def test_no_lookahead_prefix_invariance() -> None:
    """HARD PRECONDITION: a signal at bar i must not change when later bars are appended."""
    spec = ReversalSpec(swing_lookback=3, break_buffer=0.001)
    candles = _day(
        [
            (100, 99, 100),
            (101, 99, 101),
            (102, 100, 102),
            (103, 101, 101.5),  # failed upside break -> SHORT
            (104, 102, 102.5),
            (103, 99, 99.5),
            (105, 103, 103.5),
            (101, 98, 100),
        ],
        DAY,
    )
    full = spec.generate_signals(candles)
    for k in range(1, len(candles) + 1):
        prefix = spec.generate_signals(candles[:k])
        cutoff = candles[k - 1].timestamp
        expected = [s for s in full if s.asof <= cutoff]
        assert [(s.asof, s.side, s.strength) for s in prefix] == [
            (s.asof, s.side, s.strength) for s in expected
        ]


def test_swing_level_resets_each_day_no_cross_gap() -> None:
    spec = ReversalSpec(swing_lookback=3, break_buffer=0.001)
    d2 = DAY + timedelta(days=1)
    # Day 1 high reaches 110; day 2 trades in the low 100s. A day-2 failed break must reference
    # day-2's swing level (~102), NOT day-1's 110 carried across the gap.
    candles = _day([(100, 99, 100), (101, 100, 101), (110, 109, 110)], DAY) + _day(
        [(102, 100, 102), (103, 101, 101.5), (104, 102, 102.5)], d2
    )
    signals = spec.generate_signals(candles)
    # Day 2's first bar (idx 3) has no intraday reference yet -> no signal despite day-1 history.
    assert all(s.asof != candles[3].timestamp for s in signals)
    # Day 2's second bar (idx 4) fails a break of DAY-2's swing high (102) -> SHORT, though 103
    # is far below day-1's 110. The reset is what makes this a genuine (not cross-gap) signal.
    assert any(s.asof == candles[4].timestamp and s.side is Side.SHORT for s in signals)


def test_factory_reads_params_and_validates() -> None:
    spec = reversal_spec({"swing_lookback": 30, "break_buffer": 0.002})
    assert spec.swing_lookback == 30
    assert spec.break_buffer == 0.002
    assert spec.name == "reversal"
    assert spec.interval is BarInterval.MIN_5

    with pytest.raises(ValueError, match="swing_lookback"):
        ReversalSpec(swing_lookback=1)
    with pytest.raises(ValueError, match="break_buffer"):
        ReversalSpec(break_buffer=-0.1)


def test_factory_defaults_when_params_absent() -> None:
    spec = reversal_spec({})
    assert spec.swing_lookback == 20
    assert spec.break_buffer == 0.001
