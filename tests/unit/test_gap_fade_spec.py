"""Tests for the gap-fade spec (P3.10b): fade a gap that rejects VWAP -- gap-and-go's owed twin.

Verifies the rejection trigger (gap up + close BELOW VWAP -> SHORT; gap down + close ABOVE VWAP
-> LONG), the empirical DIVERGENCE from gap-and-go (the both-owed precondition: on the same
data the twins produce different trades), the point-in-time prefix invariance (the hard
no-lookahead precondition), and the factory.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from lab.core.types import BarInterval, Candle, Side
from lab.research.strategies.gap_and_go import GapAndGoSpec
from lab.research.strategies.gap_fade import GapFadeSpec, gap_fade_spec

IST = ZoneInfo("Asia/Kolkata")


def _mk(days: list[list[tuple[float, float, float, float, float]]]) -> list[Candle]:
    """Lay per-day (open, high, low, close, volume) rows onto 5-min IST sessions."""
    day0 = datetime(2024, 7, 1, 9, 15, tzinfo=IST)
    out: list[Candle] = []
    for d, rows in enumerate(days):
        for b, (o, hi, lo, c, v) in enumerate(rows):
            ts = day0 + timedelta(days=d, minutes=5 * b)
            out.append(Candle("X", BarInterval.MIN_5, ts, o, hi, lo, c, int(v)))
    return out


_BASE = [(100.0, 100.0, 100.0, 100.0, 1000.0)] * 20  # day 0: baseline (gap NaN)
# day 1: GAP UP, HOLDS above VWAP -> gap-and-go LONG, gap_fade nothing
_HOLD_UP = [(102.0, 103.0, 101.8, 102.8, 2500.0)] + [(103.0, 103.2, 102.8, 103.0, 1000.0)] * 19
# day 2: GAP UP (open 105 vs prior close 103), REJECTS below VWAP, declines -> gap_fade SHORT
_REJECT_UP = [(105.0, 105.0, 103.5, 103.8, 2500.0)] + [
    (103.8 - 0.1 * k, 104.0 - 0.1 * k, 103.6 - 0.1 * k, 103.8 - 0.1 * k, 1000.0)
    for k in range(1, 20)
]
# day 3: GAP DOWN (open 99 vs prior close 102), REJECTS above VWAP, rises -> gap_fade LONG
_REJECT_DN = [(99.0, 100.5, 99.0, 100.2, 2500.0)] + [
    (100.2 + 0.1 * k, 100.4 + 0.1 * k, 100.0 + 0.1 * k, 100.2 + 0.1 * k, 1000.0)
    for k in range(1, 20)
]
_CANDLES = _mk([_BASE, _HOLD_UP, _REJECT_UP, _REJECT_DN])


def test_fade_needs_gap_volume_and_vwap_rejection() -> None:
    signals = GapFadeSpec().generate_signals(_CANDLES)
    by_day = {s.asof.astimezone(IST).date(): s.side for s in signals}
    assert by_day == {date(2024, 7, 3): Side.SHORT, date(2024, 7, 4): Side.LONG}
    for s in signals:
        assert s.asof.hour == 9 and s.asof.minute == 15  # entered on the rejecting opening bar


def test_gap_fade_diverges_from_gap_and_go() -> None:
    # Both-owed precondition: on the SAME data the twins produce DIFFERENT trades (hold vs reject).
    go = {(s.asof, s.side) for s in GapAndGoSpec().generate_signals(_CANDLES)}
    fade = {(s.asof, s.side) for s in GapFadeSpec().generate_signals(_CANDLES)}
    assert go != fade  # genuinely divergent, not a degenerate identical pair
    assert go and fade  # both fire on this fixture
    assert go.isdisjoint(fade)  # gap-and-go on the hold day, gap_fade on the reject days
    assert {s.asof.astimezone(IST).date() for s in GapAndGoSpec().generate_signals(_CANDLES)} == {
        date(2024, 7, 2)
    }


def test_gap_fade_prefix_invariance() -> None:
    # HARD no-lookahead precondition (sparse form): signals up to each cutoff are stable.
    spec = GapFadeSpec()
    full = spec.generate_signals(_CANDLES)
    for k in range(1, len(_CANDLES) + 1):
        cutoff = _CANDLES[k - 1].timestamp
        prefix = spec.generate_signals(_CANDLES[:k])
        expected = [s for s in full if s.asof <= cutoff]
        assert [(s.asof, s.side) for s in prefix] == [(s.asof, s.side) for s in expected]


def test_factory_reads_params_and_validates() -> None:
    spec = gap_fade_spec({"gap_threshold": 0.02, "vol_mult": 2.0})
    assert spec.gap_threshold == 0.02 and spec.vol_mult == 2.0
    assert spec.name == "gap_fade" and spec.interval is BarInterval.MIN_5
    assert gap_fade_spec({}).gap_threshold == 0.010
    assert gap_fade_spec({}).vol_mult == 1.2
    with pytest.raises(ValueError, match="vol_mult"):
        GapFadeSpec(vol_mult=0.0)
