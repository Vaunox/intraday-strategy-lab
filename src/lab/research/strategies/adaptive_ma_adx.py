"""P4.4 · Adaptive MA + ADX — trend signal gated by trend strength StrategySpec (Phase 4).

Blueprint combination (RESEARCH_FINDINGS §4.2): *take the KAMA/AMA trend signal only when ADX
confirms sufficient trend strength (gate out chop).* A minimal **AND-confluence** of the KAMA
trend (P3.7) and an ADX strength gate — with the **fewest knobs** (`kama_period`,
`adx_threshold`; the ADX window is fixed textbook). ADX is direction-agnostic (it measures only
*strength*), so gating the trend by it is a reinforcing filter, not a mutually-exclusive one.

Construction (symmetric, per-bar, intraday): hold the KAMA **slope** direction — LONG while
`KAMA` is rising, SHORT while falling — **only when `ADX > adx_threshold`** (a real trend);
**flat when ADX is weak** (chop gated out). `talib.KAMA` and `talib.ADX` are trailing (confirmed
prefix-invariant), so the signal at bar `i` uses only bars `≤ i` and fills at `i+1`'s open. The
ADX gate is the intended cure for the P3.7-V2 slope's turnover death (it only trades in
confirmed trends); §6 will show whether it survives cost (informational in Phase 4). See the
P4.4 pre-registration.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from lab.core.types import BarInterval, Candle, Side, StrategySignal
from lab.data.features.indicators import adx, kama
from lab.data.features.ohlcv import OHLCV


@dataclass(frozen=True, slots=True)
class AdaptiveMaAdxSpec:
    """P4.4 -- KAMA slope trend, taken only when ADX confirms trend strength."""

    kama_period: int = 10  # KAMA window (the adaptive trend)
    adx_threshold: float = 25.0  # minimum ADX to call it a trend (gate out chop)
    adx_period: int = 14  # ADX window (fixed textbook)
    name: str = "adaptive_ma_adx"
    interval: BarInterval = BarInterval.MIN_5

    def __post_init__(self) -> None:
        """Fail loudly on degenerate parameters."""
        if self.kama_period < 2:
            raise ValueError(f"kama_period must be >= 2; got {self.kama_period!r}")
        if not 0.0 < self.adx_threshold < 100.0:
            raise ValueError(f"require 0 < adx_threshold < 100; got {self.adx_threshold!r}")
        if self.adx_period < 2:
            raise ValueError(f"adx_period must be >= 2; got {self.adx_period!r}")

    def generate_signals(self, candles: Sequence[Candle]) -> list[StrategySignal]:
        """Emit the KAMA-slope trend side each bar, but only while ADX confirms a trend."""
        ohlcv = OHLCV.from_candles(candles)
        trend = kama(ohlcv, self.kama_period)
        strength = adx(ohlcv, self.adx_period)
        signals: list[StrategySignal] = []
        for i, candle in enumerate(candles):
            side: Side | None = None
            if i > 0:
                now, prev, adx_now = float(trend[i]), float(trend[i - 1]), float(strength[i])
                if (
                    math.isfinite(now)
                    and math.isfinite(prev)
                    and math.isfinite(adx_now)
                    and adx_now > self.adx_threshold
                ):
                    if now > prev:
                        side = Side.LONG  # rising KAMA in a confirmed trend
                    elif now < prev:
                        side = Side.SHORT
            signals.append(
                StrategySignal(
                    asof=candle.timestamp,
                    symbol=candle.symbol,
                    side=side or Side.LONG,  # inert when strength == 0 (flat / chop / warmup)
                    strength=1.0 if side is not None else 0.0,
                )
            )
        return signals


def adaptive_ma_adx_spec(params: Mapping[str, float]) -> AdaptiveMaAdxSpec:
    """Build the P4.4 spec from a parameter mapping (the run_study / CLI ``SpecFactory``)."""
    defaults = AdaptiveMaAdxSpec()
    return AdaptiveMaAdxSpec(
        kama_period=round(params.get("kama_period", defaults.kama_period)),
        adx_threshold=float(params.get("adx_threshold", defaults.adx_threshold)),
    )
