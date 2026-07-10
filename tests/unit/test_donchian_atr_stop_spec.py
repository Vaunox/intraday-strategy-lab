"""Tests for P4.6 donchian_atr_stop: Donchian breakout entry, ATR-trailing-stop exit.

Covers the no-lookahead prefix-invariance precondition (hard gate), the breakout entry + stop
exit behaviour, and the factory.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from lab.core.types import BarInterval, Candle, Side
from lab.research.strategies.donchian_atr_stop import DonchianAtrStopSpec, donchian_atr_stop_spec

IST = ZoneInfo("Asia/Kolkata")
_BARS_PER_DAY = 400  # keep the whole up-trend + drop within one IST day (no square-off reset)


def _sessions(closes: list[float]) -> list[Candle]:
    day0 = datetime(2024, 7, 1, 9, 15, tzinfo=IST)
    out: list[Candle] = []
    for i, c in enumerate(closes):
        d, b = divmod(i, _BARS_PER_DAY)
        ts = day0 + timedelta(days=d, minutes=5 * b)
        out.append(Candle("X", BarInterval.MIN_5, ts, c, c + 0.5, c - 0.5, c, 1000))
    return out


def _breakout_then_drop() -> list[Candle]:
    up = [100.0 + 1.0 * i for i in range(120)]  # steady new highs (drift > bar range) -> breakouts
    drop = [up[-1] - 3.0 * i for i in range(1, 25)]  # sharp drop -> ATR trailing stop hit
    return _sessions(up + drop + [drop[-1]] * 20)


_CANDLES = _breakout_then_drop()


def test_breakout_entry_then_atr_stop_exit() -> None:
    signals = DonchianAtrStopSpec().generate_signals(_CANDLES)
    longs = [i for i, s in enumerate(signals) if s.strength == 1.0 and s.side is Side.LONG]
    assert longs  # entered long on the up-trend breakouts
    # after the drop begins (bar 120+), the trailing stop is hit -> the position goes flat
    flat_after_drop = any(s.strength == 0.0 for s in signals[125:])
    assert flat_after_drop


def test_donchian_atr_stop_prefix_invariance() -> None:
    spec = DonchianAtrStopSpec()
    full = spec.generate_signals(_CANDLES)
    for k in range(1, len(_CANDLES) + 1):
        prefix = spec.generate_signals(_CANDLES[:k])
        assert [(s.side, s.strength) for s in prefix] == [(s.side, s.strength) for s in full[:k]]


def test_factory_reads_params_and_validates() -> None:
    spec = donchian_atr_stop_spec({"channel_lookback": 40, "atr_mult": 3.0})
    assert spec.channel_lookback == 40 and spec.atr_mult == 3.0
    assert spec.name == "donchian_atr_stop" and spec.atr_period == 14
    assert (
        donchian_atr_stop_spec({}).channel_lookback == 55
        and donchian_atr_stop_spec({}).atr_mult == 2.0
    )
    with pytest.raises(ValueError, match="atr_mult"):
        DonchianAtrStopSpec(atr_mult=0.0)
