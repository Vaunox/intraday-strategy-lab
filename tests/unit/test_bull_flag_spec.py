"""Tests for the bull-flag spec (P3.12): impulse + tight consolidation + breakout.

Verifies a bull flag fires LONG (up-impulse, tight pause, break up) and a bear flag fires SHORT,
with nothing on a flat day, the point-in-time prefix invariance (the hard no-lookahead
precondition), and the factory.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from lab.core.types import BarInterval, Candle, Side
from lab.research.strategies.bull_flag import BullFlagSpec, bull_flag_spec

IST = ZoneInfo("Asia/Kolkata")


def _flag_day(
    impulse: list[float], consol: list[float], breakout: float
) -> list[tuple[float, float, float]]:
    """6 flat + a 7-bar impulse leg + a 5-bar tight consolidation + a breakout bar + tail."""
    rows: list[tuple[float, float, float]] = [(100.0, 100.2, 99.8)] * 6
    rows += [(c, c + 0.2, c - 0.2) for c in impulse]  # bars 6-12 (impulse leg)
    rows += [(c, c + 0.15, c - 0.15) for c in consol]  # bars 13-17 (tight consolidation)
    rows.append((breakout, breakout + 0.2, breakout - 0.2))  # bar 18 (breakout)
    rows += [(breakout, breakout + 0.2, breakout - 0.2)] * 11
    return rows


_BULL = _flag_day(
    [100.0, 100.25, 100.5, 100.7, 100.9, 101.2, 101.4],
    [101.35, 101.45, 101.4, 101.3, 101.45],
    102.2,
)
_BEAR = _flag_day(
    [100.0, 99.75, 99.5, 99.3, 99.1, 98.8, 98.6], [98.65, 98.55, 98.6, 98.7, 98.55], 97.8
)
_FLAT = [(100.0, 100.2, 99.8)] * 30


def _mk(days: list[list[tuple[float, float, float]]]) -> list[Candle]:
    day0 = datetime(2024, 7, 1, 9, 15, tzinfo=IST)
    out: list[Candle] = []
    for d, rows in enumerate(days):
        for b, (c, hi, lo) in enumerate(rows):
            ts = day0 + timedelta(days=d, minutes=5 * b)
            out.append(Candle("X", BarInterval.MIN_5, ts, c, hi, lo, c, 1000))
    return out


_CANDLES = _mk([_BULL, _BEAR, _FLAT])


def test_bull_flag_longs_bear_flag_shorts_flat_nothing() -> None:
    signals = BullFlagSpec().generate_signals(_CANDLES)
    long_days = {s.asof.astimezone(IST).date() for s in signals if s.side is Side.LONG}
    short_days = {s.asof.astimezone(IST).date() for s in signals if s.side is Side.SHORT}
    assert long_days == {date(2024, 7, 1)}  # only the bull-flag day
    assert short_days == {date(2024, 7, 2)}  # only the bear-flag day
    assert all(s.asof.astimezone(IST).date() != date(2024, 7, 3) for s in signals)  # flat day: none


def test_bull_flag_prefix_invariance() -> None:
    # HARD no-lookahead precondition (sparse form): signals up to each cutoff are stable.
    spec = BullFlagSpec()
    full = spec.generate_signals(_CANDLES)
    for k in range(1, len(_CANDLES) + 1):
        cutoff = _CANDLES[k - 1].timestamp
        prefix = spec.generate_signals(_CANDLES[:k])
        expected = [s for s in full if s.asof <= cutoff]
        assert [(s.asof, s.side) for s in prefix] == [(s.asof, s.side) for s in expected]


def test_factory_reads_params_and_validates() -> None:
    spec = bull_flag_spec({"impulse_threshold": 0.015, "tight_frac": 0.35})
    assert spec.impulse_threshold == 0.015 and spec.tight_frac == 0.35
    assert spec.name == "bull_flag" and spec.impulse_lookback == 6 and spec.flag_lookback == 6
    assert bull_flag_spec({}).impulse_threshold == 0.010
    assert bull_flag_spec({}).tight_frac == 0.5
    with pytest.raises(ValueError, match="tight_frac"):
        BullFlagSpec(tight_frac=0.0)
