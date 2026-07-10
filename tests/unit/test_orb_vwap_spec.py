"""Tests for P4.2 orb_vwap: opening-range break confirmed by the VWAP side.

Covers the no-lookahead prefix-invariance precondition (hard gate), the VWAP-confirmed break
gating, and the factory.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from lab.core.types import BarInterval, Candle, Side
from lab.research.strategies.orb_vwap import OrbVwapSpec, orb_vwap_spec

IST = ZoneInfo("Asia/Kolkata")


def _mk(days: list[list[tuple[float, float, float]]]) -> list[Candle]:
    day0 = datetime(2024, 7, 1, 9, 15, tzinfo=IST)
    out: list[Candle] = []
    for d, rows in enumerate(days):
        for b, (c, hi, lo) in enumerate(rows):
            ts = day0 + timedelta(days=d, minutes=5 * b)
            out.append(Candle("X", BarInterval.MIN_5, ts, c, hi, lo, c, 1000))
    return out


# 6 opening-range bars (30 min, range [99,101]) then a break; VWAP sits ~100 so a break UP is
# above VWAP (LONG) and a break DOWN is below VWAP (SHORT).
_RANGE = [(100.0, 101.0, 99.0)] * 6
_CANDLES = _mk(
    [
        _RANGE
        + [(102.5 + 0.3 * k, 103.0 + 0.3 * k, 102.0 + 0.3 * k) for k in range(8)],  # break up
        _RANGE + [(97.5 - 0.3 * k, 98.0 - 0.3 * k, 97.0 - 0.3 * k) for k in range(8)],  # break down
    ]
)


def test_vwap_confirmed_break_both_directions() -> None:
    signals = OrbVwapSpec().generate_signals(_CANDLES)
    by_day = {s.asof.astimezone(IST).date(): s.side for s in signals}
    assert by_day == {date(2024, 7, 1): Side.LONG, date(2024, 7, 2): Side.SHORT}
    for s in signals:  # entered only after the 30-min opening-range window closed
        assert s.asof >= s.asof.replace(hour=9, minute=15) + timedelta(minutes=30)


def test_orb_vwap_prefix_invariance() -> None:
    spec = OrbVwapSpec()
    full = spec.generate_signals(_CANDLES)
    for k in range(1, len(_CANDLES) + 1):
        cutoff = _CANDLES[k - 1].timestamp
        prefix = spec.generate_signals(_CANDLES[:k])
        expected = [s for s in full if s.asof <= cutoff]
        assert [(s.asof, s.side) for s in prefix] == [(s.asof, s.side) for s in expected]


def test_factory_reads_params_and_validates() -> None:
    spec = orb_vwap_spec({"opening_range_minutes": 15, "break_buffer": 0.002})
    assert spec.opening_range_minutes == 15 and spec.break_buffer == 0.002
    assert spec.name == "orb_vwap"
    assert orb_vwap_spec({}).opening_range_minutes == 30 and orb_vwap_spec({}).break_buffer == 0.001
    with pytest.raises(ValueError, match="break_buffer"):
        OrbVwapSpec(break_buffer=-0.001)
