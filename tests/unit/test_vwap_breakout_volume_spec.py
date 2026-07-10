"""Tests for P4.1 vwap_breakout_volume: range break on the VWAP trend side, on a volume surge.

Covers the no-lookahead prefix-invariance precondition (hard gate), the three-way confluence
gating, and the factory.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from lab.core.types import BarInterval, Candle, Side
from lab.research.strategies.vwap_breakout_volume import (
    VwapBreakoutVolumeSpec,
    vwap_breakout_volume_spec,
)

IST = ZoneInfo("Asia/Kolkata")


def _mk(days: list[list[tuple[float, float, float, float, float]]]) -> list[Candle]:
    day0 = datetime(2024, 7, 1, 9, 15, tzinfo=IST)
    out: list[Candle] = []
    for d, rows in enumerate(days):
        for b, (o, hi, lo, c, v) in enumerate(rows):
            ts = day0 + timedelta(days=d, minutes=5 * b)
            out.append(Candle("X", BarInterval.MIN_5, ts, o, hi, lo, c, int(v)))
    return out


_RANGE = [(100.0, 101.0, 99.0, 100.0, 1000.0)] * 20  # 20 bars: intraday range [99,101], vol base
# day 1: break UP above 101, above VWAP, on a 2.2x volume surge -> LONG
_UP = _RANGE + [
    (102.0 + 0.4 * k, 103.0 + 0.4 * k, 101.0 + 0.4 * k, 102.5 + 0.4 * k, 2200.0) for k in range(10)
]
# day 2: break DOWN below 99, below VWAP, on a surge -> SHORT
_DN = _RANGE + [
    (98.0 - 0.4 * k, 99.0 - 0.4 * k, 97.0 - 0.4 * k, 97.5 - 0.4 * k, 2200.0) for k in range(10)
]
_CANDLES = _mk([_UP, _DN])


def test_confluence_break_up_and_down() -> None:
    signals = VwapBreakoutVolumeSpec().generate_signals(_CANDLES)
    sides = {s.asof.astimezone(IST).date(): s.side for s in signals}
    from datetime import date

    assert sides.get(date(2024, 7, 1)) is Side.LONG  # up break, above VWAP, on a surge
    assert sides.get(date(2024, 7, 2)) is Side.SHORT
    assert all(s.strength == 1.0 for s in signals)


def test_no_entry_without_volume_surge() -> None:
    # same breakout geometry but flat 1000 volume -> no surge -> no confluence, no signal
    quiet = [
        _RANGE
        + [
            (102.0 + 0.4 * k, 103.0 + 0.4 * k, 101.0 + 0.4 * k, 102.5 + 0.4 * k, 1000.0)
            for k in range(10)
        ]
    ]
    assert VwapBreakoutVolumeSpec().generate_signals(_mk(quiet)) == []


def test_vwap_breakout_volume_prefix_invariance() -> None:
    spec = VwapBreakoutVolumeSpec()
    full = spec.generate_signals(_CANDLES)
    for k in range(1, len(_CANDLES) + 1):
        cutoff = _CANDLES[k - 1].timestamp
        prefix = spec.generate_signals(_CANDLES[:k])
        expected = [s for s in full if s.asof <= cutoff]
        assert [(s.asof, s.side) for s in prefix] == [(s.asof, s.side) for s in expected]


def test_factory_reads_params_and_validates() -> None:
    spec = vwap_breakout_volume_spec({"breakout_lookback": 15, "vol_mult": 2.0})
    assert spec.breakout_lookback == 15 and spec.vol_mult == 2.0
    assert spec.name == "vwap_breakout_volume" and spec.volume_period == 20
    assert vwap_breakout_volume_spec({}).breakout_lookback == 20
    assert vwap_breakout_volume_spec({}).vol_mult == 1.5
    with pytest.raises(ValueError, match="vol_mult"):
        VwapBreakoutVolumeSpec(vol_mult=0.0)
