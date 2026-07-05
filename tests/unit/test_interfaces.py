"""Tests that trivial fakes satisfy the core Protocols — structurally and at runtime.

The typed ``_use_*`` helpers exercise structural conformance under mypy (the
fakes are passed where the Protocol is expected); the ``isinstance`` checks
exercise the ``@runtime_checkable`` behavior. Together these are the P0.5
"trivial fakes type-check" acceptance criterion.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from zoneinfo import ZoneInfo

from lab.core.interfaces import BrokerAdapter, Repository, StrategySpec
from lab.core.types import BarInterval, Candle, Side, StrategySignal

IST = ZoneInfo("Asia/Kolkata")


class FakeBroker:
    """Minimal in-memory BrokerAdapter fake."""

    def fetch_historical_candles(
        self, symbol: str, interval: BarInterval, start: datetime, end: datetime
    ) -> Sequence[Candle]:
        return []


class FakeRepository:
    """Minimal in-memory Repository fake."""

    def __init__(self) -> None:
        self.written: list[Candle] = []

    def write_candles(self, symbol: str, interval: BarInterval, candles: Sequence[Candle]) -> None:
        self.written.extend(candles)

    def read_candles(
        self, symbol: str, interval: BarInterval, start: datetime, end: datetime
    ) -> Sequence[Candle]:
        return list(self.written)


class FakeStrategy:
    """Minimal StrategySpec fake that goes long on the first bar."""

    @property
    def name(self) -> str:
        return "fake"

    @property
    def interval(self) -> BarInterval:
        return BarInterval.MIN_5

    def generate_signals(self, candles: Sequence[Candle]) -> Sequence[StrategySignal]:
        if not candles:
            return []
        first = candles[0]
        return [StrategySignal(asof=first.timestamp, symbol=first.symbol, side=Side.LONG)]


def _use_broker(adapter: BrokerAdapter) -> Sequence[Candle]:
    return adapter.fetch_historical_candles(
        "X",
        BarInterval.MIN_5,
        datetime(2024, 7, 15, 9, 15, tzinfo=IST),
        datetime(2024, 7, 15, 15, 30, tzinfo=IST),
    )


def _use_repository(repo: Repository) -> Sequence[Candle]:
    repo.write_candles("X", BarInterval.MIN_5, [])
    return repo.read_candles(
        "X",
        BarInterval.MIN_5,
        datetime(2024, 7, 15, 9, 15, tzinfo=IST),
        datetime(2024, 7, 15, 15, 30, tzinfo=IST),
    )


def _use_strategy(spec: StrategySpec) -> Sequence[StrategySignal]:
    return spec.generate_signals([])


def test_fakes_conform_structurally() -> None:
    # These calls only type-check if the fakes satisfy the Protocols (mypy).
    assert _use_broker(FakeBroker()) == []
    assert _use_repository(FakeRepository()) == []
    assert _use_strategy(FakeStrategy()) == []


def test_fakes_are_runtime_checkable() -> None:
    assert isinstance(FakeBroker(), BrokerAdapter)
    assert isinstance(FakeRepository(), Repository)
    assert isinstance(FakeStrategy(), StrategySpec)


def test_strategy_emits_signal_on_first_candle() -> None:
    candle = Candle(
        "X",
        BarInterval.MIN_5,
        datetime(2024, 7, 15, 9, 15, tzinfo=IST),
        100.0,
        101.0,
        99.0,
        100.5,
        1000,
    )
    signals = FakeStrategy().generate_signals([candle])
    assert len(signals) == 1
    assert signals[0].side is Side.LONG
