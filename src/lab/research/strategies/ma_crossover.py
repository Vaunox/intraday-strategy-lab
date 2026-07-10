"""Moving-Average-Crossover StrategySpec (Phase 3, P3.14): the canonical fast/slow SMA cross.

A-priori mechanism: a fast moving average crossing a slow one marks a shift in the prevailing
trend; hold the trend side (long while fast > slow, short while fast < slow) to capture the
sustained move. The canonical "great in a trend, whipsawed in chop" rule.

Construction (symmetric, intraday): per bar, target the sign of ``SMA(fast) - SMA(slow)`` --
LONG while fast is above slow, SHORT while below, flat during warmup -- so the position flips on
each crossover and is squared off at the MIS cutoff (the adapter resets flat each day). SMAs are
trailing (``talib.SMA``, prefix-invariant), so a signal at bar ``i`` uses only bars ``<= i`` and
fills at ``i+1``'s open. Distinct from P3.7 (adaptive KAMA): plain, non-adaptive SMAs. SMA is
the primary; EMA is a non-owed method variant (ledger-charged, run only if the primary shows
life). See the P3.14 pre-registration.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from lab.core.types import BarInterval, Candle, Side, StrategySignal
from lab.data.features.indicators import sma
from lab.data.features.ohlcv import OHLCV


@dataclass(frozen=True, slots=True)
class MaCrossoverSpec:
    """P3.14 -- hold the sign of (fast SMA - slow SMA); flip on the crossover."""

    fast_period: int = 20  # fast SMA window
    slow_period: int = 50  # slow SMA window (> fast)
    name: str = "ma_crossover"
    interval: BarInterval = BarInterval.MIN_5

    def __post_init__(self) -> None:
        """Fail loudly on mis-ordered / degenerate windows."""
        if not 2 <= self.fast_period < self.slow_period:
            raise ValueError(
                "require 2 <= fast_period < slow_period; got "
                f"fast={self.fast_period!r}, slow={self.slow_period!r}"
            )

    def generate_signals(self, candles: Sequence[Candle]) -> list[StrategySignal]:
        """Emit the trend-side target (sign of fast - slow) each bar."""
        ohlcv = OHLCV.from_candles(candles)
        fast = sma(ohlcv, self.fast_period)
        slow = sma(ohlcv, self.slow_period)
        signals: list[StrategySignal] = []
        for i, candle in enumerate(candles):
            f, s = float(fast[i]), float(slow[i])
            side: Side | None = None
            if math.isfinite(f) and math.isfinite(s):
                if f > s:
                    side = Side.LONG
                elif f < s:
                    side = Side.SHORT
            signals.append(
                StrategySignal(
                    asof=candle.timestamp,
                    symbol=candle.symbol,
                    side=side or Side.LONG,  # inert when strength == 0 (flat / warmup)
                    strength=1.0 if side is not None else 0.0,
                )
            )
        return signals


def ma_crossover_spec(params: Mapping[str, float]) -> MaCrossoverSpec:
    """Build the P3.14 spec from a parameter mapping (the run_study / CLI ``SpecFactory``)."""
    defaults = MaCrossoverSpec()
    return MaCrossoverSpec(
        fast_period=round(params.get("fast_period", defaults.fast_period)),
        slow_period=round(params.get("slow_period", defaults.slow_period)),
    )
