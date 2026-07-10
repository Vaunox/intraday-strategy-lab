"""Tests for the gap-and-go spec (P3.10): confirmed opening-gap continuation.

Verifies an entry fires only when the gap qualifies, volume confirms, and price is on the gap
side of the intraday VWAP (gap up + above VWAP -> LONG; gap down + below VWAP -> SHORT), the
point-in-time prefix invariance (the hard no-lookahead precondition), and the factory.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from lab.core.types import BarInterval, Candle, Side
from lab.research.strategies.gap_and_go import GapAndGoSpec, gap_and_go_spec

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


_FLAT = [(100.0, 100.0, 100.0, 100.0, 1000.0)] * 20  # day 0: baseline (gap NaN -> no signal)
# day 1: GAP UP (+2%), high opening volume, holds above VWAP -> LONG
_GAP_UP = [(102.0, 103.0, 101.8, 102.8, 2500.0)] + [(103.0, 103.2, 102.8, 103.0, 1000.0)] * 19
# day 2: GAP DOWN (from ~103 to 98), high opening volume, holds below VWAP -> SHORT
_GAP_DN = [(98.0, 98.2, 97.0, 97.2, 2500.0)] + [(97.0, 97.2, 96.8, 97.0, 1000.0)] * 19
# day 3: NO gap (opens at the prior close) -> no signal
_NOGAP = [(97.0, 97.1, 96.9, 97.0, 1000.0)] * 20
_CANDLES = _mk([_FLAT, _GAP_UP, _GAP_DN, _NOGAP])


def test_entry_needs_gap_volume_and_vwap_side() -> None:
    signals = GapAndGoSpec().generate_signals(_CANDLES)
    assert [s.side for s in signals] == [Side.LONG, Side.SHORT]  # gap-up day, gap-down day
    for s in signals:
        assert s.asof.hour == 9 and s.asof.minute == 15  # entered on the confirmed opening bar


def test_gap_and_go_prefix_invariance() -> None:
    # HARD no-lookahead precondition (sparse form): signals up to each cutoff are stable.
    spec = GapAndGoSpec()
    full = spec.generate_signals(_CANDLES)
    for k in range(1, len(_CANDLES) + 1):
        cutoff = _CANDLES[k - 1].timestamp
        prefix = spec.generate_signals(_CANDLES[:k])
        expected = [s for s in full if s.asof <= cutoff]
        assert [(s.asof, s.side) for s in prefix] == [(s.asof, s.side) for s in expected]


def test_factory_reads_params_and_validates() -> None:
    spec = gap_and_go_spec({"gap_threshold": 0.02, "vol_mult": 2.0})
    assert spec.gap_threshold == 0.02 and spec.vol_mult == 2.0
    assert spec.name == "gap_and_go" and spec.interval is BarInterval.MIN_5
    assert gap_and_go_spec({}).gap_threshold == 0.010
    assert gap_and_go_spec({}).vol_mult == 1.2
    with pytest.raises(ValueError, match="gap_threshold"):
        GapAndGoSpec(gap_threshold=0.0)
