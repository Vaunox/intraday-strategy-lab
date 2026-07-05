"""A trivial reference StrategySpec (Phase 2, P2.4).

Deterministic and self-contained (no feature library): it simply trades in the
direction of the last bar (up close -> long, down -> short). Its only job is to
run the validation engine end to end — it is NOT meant to be a real edge. The
real strategy studies (Phase 3) each provide their own thin ``StrategySpec``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from lab.core.types import BarInterval, Candle, Side, StrategySignal


@dataclass(frozen=True, slots=True)
class ReferenceMomentumSpec:
    """Reference spec: take the side of each bar's close-vs-open move."""

    name: str = "reference_momentum"
    interval: BarInterval = BarInterval.MIN_5

    def generate_signals(self, candles: Sequence[Candle]) -> list[StrategySignal]:
        """Emit one signal per bar: long if the bar closed up, else short."""
        return [
            StrategySignal(
                asof=candle.timestamp,
                symbol=candle.symbol,
                side=Side.LONG if candle.close >= candle.open else Side.SHORT,
                strength=1.0,
            )
            for candle in candles
        ]
