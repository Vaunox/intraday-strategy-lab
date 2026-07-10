"""Tests for the momentum-pullback spec (P3.9): in-trend RSI pullback-resumption.

Verifies the core GATING (LONG entries only when close is above the trend SMA and RSI crosses
back up through the pullback level; SHORT the mirror), that the trend filter is not a no-op
(it suppresses same-shaped RSI crosses that occur against the trend), the same-day-cross guard
(no entry can fire on a day's first bar, so an overnight RSI jump is never read as a
resumption), the point-in-time prefix invariance (the hard no-lookahead precondition), and the
factory.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from lab.core.types import BarInterval, Candle, Side, StrategySignal
from lab.data.features.indicators import rsi, sma
from lab.data.features.ohlcv import OHLCV
from lab.research.strategies.momentum_pullback import (
    MomentumPullbackSpec,
    momentum_pullback_spec,
)

IST = ZoneInfo("Asia/Kolkata")
_BARS_PER_DAY = 75


def _sessions(closes: list[float]) -> list[Candle]:
    """Lay closes onto clean 5-min IST sessions (75 bars/day) so day boundaries are exercised."""
    day0 = datetime(2024, 7, 1, 9, 15, tzinfo=IST)
    out: list[Candle] = []
    for i, c in enumerate(closes):
        d, b = divmod(i, _BARS_PER_DAY)
        ts = day0 + timedelta(days=d, minutes=5 * b)
        out.append(Candle("X", BarInterval.MIN_5, ts, c, c + 0.5, c - 0.5, c, 1000))
    return out


def _trend_with_pullbacks() -> list[Candle]:
    # An up-drift then a down-drift (+/-0.7 per bar), each with a fast oscillation (18*sin,
    # period ~25 bars). The drift is tuned to hold close on one side of SMA(50) at the
    # oscillation TROUGHS (drift*24.5 ~= 17 > residual after smoothing) -- a persistent up/down
    # regime -- while the oscillation still swings RSI(14) below the pullback level and back, so
    # the resumption cross fires WHILE in-trend: LONG in the up half, SHORT in the down half. The
    # same upward-cross shape also occurs against the trend in the down half (the no-op test).
    closes: list[float] = [100.0 + 0.7 * i + 18.0 * math.sin(i / 4.0) for i in range(225)]
    peak = closes[-1]
    closes += [peak - 0.7 * i + 18.0 * math.sin((224 + i) / 4.0) for i in range(1, 226)]
    return _sessions(closes)


_CANDLES = _trend_with_pullbacks()
_IDX = {c.timestamp: i for i, c in enumerate(_CANDLES)}


def _entries(signals: list[StrategySignal]) -> list[StrategySignal]:
    """The signals that ENTER from flat (strength 1.0 following a flat/opposite bar)."""
    out: list[StrategySignal] = []
    prev = 0.0
    for s in signals:
        if s.strength == 1.0 and prev == 0.0:
            out.append(s)
        prev = s.strength
    return out


def test_entries_are_in_trend_and_the_trend_gate_is_not_a_noop() -> None:
    spec = MomentumPullbackSpec(rsi_pullback=45.0)  # a shallow level so the mild synthetic fires
    ohlcv = OHLCV.from_candles(_CANDLES)
    trend = sma(ohlcv, spec.trend_period)
    strength_index = rsi(ohlcv, spec.rsi_period)
    upper = 100.0 - spec.rsi_pullback
    signals = spec.generate_signals(_CANDLES)

    longs = shorts = 0
    for s in _entries(signals):
        i = _IDX[s.asof]
        prev_r, now_r, close, t = (
            float(strength_index[i - 1]),
            float(strength_index[i]),
            float(_CANDLES[i].close),
            float(trend[i]),
        )
        if s.side is Side.LONG:
            assert close > t  # entered long only in an uptrend
            assert prev_r < spec.rsi_pullback <= now_r  # on an upward cross of the pullback level
            longs += 1
        else:
            assert close < t  # entered short only in a downtrend
            assert prev_r > upper >= now_r
            shorts += 1
    assert longs > 0 and shorts > 0  # both sides exercised by the up/down halves

    # not a no-op: the SAME upward-cross shape also happens in downtrends, and is suppressed.
    all_up_crosses = sum(
        1
        for i in range(1, len(_CANDLES))
        if _CANDLES[i].timestamp.astimezone(IST).date()
        == _CANDLES[i - 1].timestamp.astimezone(IST).date()
        and math.isfinite(float(strength_index[i - 1]))
        and float(strength_index[i - 1]) < spec.rsi_pullback <= float(strength_index[i])
    )
    assert 0 < longs < all_up_crosses  # the trend filter blocked the against-trend crosses


def test_no_entry_on_the_first_bar_of_a_day() -> None:
    # The same-day-cross guard: rsi_prev is masked across a day boundary, so no resumption cross
    # can be detected on a day's first bar -> an overnight RSI jump never triggers an entry.
    spec = MomentumPullbackSpec(rsi_pullback=45.0)
    signals = spec.generate_signals(_CANDLES)
    for s in _entries(signals):
        i = _IDX[s.asof]
        assert i % _BARS_PER_DAY != 0  # never the first bar of a session


def test_momentum_pullback_prefix_invariance() -> None:
    # The HARD no-lookahead precondition: the per-bar target for bars 0..k-1 is identical whether
    # computed on the full series or only its length-k prefix (strictly point-in-time).
    spec = MomentumPullbackSpec()
    full = spec.generate_signals(_CANDLES)
    for k in range(1, len(_CANDLES) + 1):
        prefix = spec.generate_signals(_CANDLES[:k])
        assert [(s.side, s.strength) for s in prefix] == [(s.side, s.strength) for s in full[:k]]


def test_factory_reads_params_and_validates() -> None:
    spec = momentum_pullback_spec({"trend_period": 40, "rsi_pullback": 35})
    assert spec.trend_period == 40 and spec.rsi_pullback == 35 and spec.rsi_period == 14
    assert spec.name == "momentum_pullback" and spec.interval is BarInterval.MIN_5
    assert momentum_pullback_spec({}).trend_period == 50
    assert momentum_pullback_spec({}).rsi_pullback == 30
    with pytest.raises(ValueError, match="trend_period"):
        MomentumPullbackSpec(trend_period=1)
    with pytest.raises(ValueError, match="rsi_pullback"):
        MomentumPullbackSpec(rsi_pullback=60.0)  # must be below the 50 midline
