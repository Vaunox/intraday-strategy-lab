"""Tests for the domain value types and their construction-time validation."""

from __future__ import annotations

import dataclasses
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from lab.core.types import (
    BarInterval,
    Candle,
    Side,
    StrategySignal,
    Verdict,
)

IST = ZoneInfo("Asia/Kolkata")


def _ts(hour: int = 10, minute: int = 0) -> datetime:
    return datetime(2024, 7, 15, hour, minute, tzinfo=IST)


def test_candle_valid_construction() -> None:
    candle = Candle(
        symbol="RELIANCE",
        interval=BarInterval.MIN_5,
        timestamp=_ts(),
        open=100.0,
        high=102.5,
        low=99.5,
        close=101.0,
        volume=12345,
    )
    assert candle.high >= candle.close >= candle.low
    assert candle.open_interest is None


def test_candle_is_frozen() -> None:
    candle = Candle("X", BarInterval.DAY, _ts(), 1.0, 1.0, 1.0, 1.0, 0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        candle.close = 2.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"high": 98.0},  # high below low/open/close
        {"low": 103.0},  # low above open/close
        {"open": -1.0},  # non-positive price
        {"close": float("nan")},  # non-finite price
        {"volume": -5},  # negative volume
    ],
)
def test_candle_rejects_bad_values(kwargs: dict[str, float]) -> None:
    base = {
        "symbol": "X",
        "interval": BarInterval.MIN_5,
        "timestamp": _ts(),
        "open": 100.0,
        "high": 102.0,
        "low": 99.0,
        "close": 101.0,
        "volume": 10,
    }
    base.update(kwargs)
    with pytest.raises(ValueError):
        Candle(**base)  # type: ignore[arg-type]


def test_candle_rejects_naive_timestamp() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        Candle("X", BarInterval.MIN_5, datetime(2024, 7, 15, 10, 0), 1.0, 1.0, 1.0, 1.0, 0)


def test_strategy_signal_strength_bounds() -> None:
    ok = StrategySignal(asof=_ts(), symbol="X", side=Side.LONG, strength=0.5)
    assert ok.side is Side.LONG
    with pytest.raises(ValueError, match="strength"):
        StrategySignal(asof=_ts(), symbol="X", side=Side.SHORT, strength=1.5)


def test_enum_values_match_kite_intervals() -> None:
    # BarInterval values mirror Kite's interval strings (Phase 1 mapping).
    assert BarInterval.MIN_5.value == "5minute"
    assert BarInterval.DAY.value == "day"
    assert {v.value for v in Verdict} == {"pass", "kill"}
