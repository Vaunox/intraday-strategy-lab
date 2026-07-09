"""Adaptive-MA StrategySpecs (Phase 3, P3.7): KAMA fast/slow cross (V1) and slope (V2).

Two OWED directional variants, a genuine cross/slope dichotomy (both pre-registered, both
run):

- **Cross (:class:`AdaptiveMaCrossSpec`, V1)** -- the crossing signal: position follows the
  cross of a FAST and a SLOW KAMA (fast above slow -> LONG, below -> SHORT).
- **Slope (:class:`AdaptiveMaSlopeSpec`, V2)** -- the trend-state signal: position follows a
  single KAMA's own direction (rising -> LONG, falling -> SHORT).

**Why V1 uses TWO KAMAs (a corrected definition):** for a single KAMA, a *price-vs-KAMA*
cross is mathematically IDENTICAL to its slope -- KAMA sits between its prior value and the
close, so ``sign(close - KAMA) == sign(KAMA - KAMA_prev)`` every bar, and the two would be
the SAME strategy (found + proven pre-run, 2026-07-09). A **fast/slow cross of two KAMAs**
breaks that identity: the fast KAMA can lead above the slow while the slow KAMA still falls
(a reversal-lead divergence a single MA cannot have). The divergence is **proven
empirically** -- ~34.5% of real RELIANCE bars disagree (see ``test_adaptive_ma_spec.py``) --
not asserted. Per the pre-registration, if BOTH pass that is a contradiction to investigate.

Distinct from P3.14 (Moving-Average Crossover): P3.14 crosses *fixed* SMA/EMA; here BOTH
MAs are the *efficiency-ratio adaptive* KAMA -- adaptive vs fixed smoothing, genuinely
different, not a near-duplicate.

TRAILING KAMA (``indicators.kama`` = ``talib.KAMA``), NOT intraday-reset (operator ruling,
P3.7 §3): trailing KAMA is causal (reflecting yesterday's level at today's open is legitimate
*past* data, not lookahead), and resetting it daily would amputate the cross-day adaptivity
that is the defining property of an adaptive MA. KAMA's prefix-invariance (no forward leak)
is proven by ``tests/unit/test_kama_causality.py`` -- a P3.7 precondition. Point-in-time: a
signal at bar ``t``'s close fills at ``t+1``'s open (Inviolable Rule 2); each day opens flat;
the backtester squares off at the MIS cutoff.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from lab.core.types import BarInterval, Candle, Side, StrategySignal
from lab.data.features.indicators import kama
from lab.data.features.ohlcv import OHLCV


@dataclass(frozen=True, slots=True)
class AdaptiveMaCrossSpec:
    """V1 -- trade the cross of a FAST and SLOW KAMA (the crossing signal)."""

    fast_period: int = 10  # fast KAMA efficiency-ratio window
    slow_period: int = 30  # slow KAMA efficiency-ratio window (> fast)
    name: str = "adaptive_ma_cross"
    interval: BarInterval = BarInterval.MIN_5

    def __post_init__(self) -> None:
        """Fail loudly on degenerate or mis-ordered windows."""
        if self.fast_period < 2:
            raise ValueError(f"fast_period must be >= 2; got {self.fast_period!r}")
        if self.slow_period <= self.fast_period:
            raise ValueError(
                "slow_period must be > fast_period; got "
                f"fast={self.fast_period!r}, slow={self.slow_period!r}"
            )

    def generate_signals(self, candles: Sequence[Candle]) -> list[StrategySignal]:
        """Per bar: LONG while fast KAMA > slow KAMA, SHORT while below, flat while warming up."""
        ohlcv = OHLCV.from_candles(candles)
        fast = kama(ohlcv, self.fast_period)
        slow = kama(ohlcv, self.slow_period)
        signals: list[StrategySignal] = []
        for i, candle in enumerate(candles):
            fast_i, slow_i = float(fast[i]), float(slow[i])
            side: Side | None = None
            strength = 0.0
            if math.isfinite(fast_i) and math.isfinite(slow_i):
                if fast_i > slow_i:
                    side, strength = Side.LONG, 1.0
                elif fast_i < slow_i:
                    side, strength = Side.SHORT, 1.0
            signals.append(
                StrategySignal(
                    asof=candle.timestamp,
                    symbol=candle.symbol,
                    side=side or Side.LONG,  # inert when strength == 0 (flat)
                    strength=strength,
                )
            )
        return signals


def adaptive_ma_cross_spec(params: Mapping[str, float]) -> AdaptiveMaCrossSpec:
    """Build the cross spec from a parameter mapping (the run_study / CLI ``SpecFactory``)."""
    defaults = AdaptiveMaCrossSpec()
    return AdaptiveMaCrossSpec(
        fast_period=round(params.get("fast_period", defaults.fast_period)),
        slow_period=round(params.get("slow_period", defaults.slow_period)),
    )


@dataclass(frozen=True, slots=True)
class AdaptiveMaSlopeSpec:
    """V2 -- trade the direction a single KAMA points (the trend-state signal)."""

    kama_period: int = 10  # KAMA efficiency-ratio window (Kaufman-classic)
    name: str = "adaptive_ma_slope"
    interval: BarInterval = BarInterval.MIN_5

    def __post_init__(self) -> None:
        """Fail loudly on a degenerate window."""
        if self.kama_period < 2:
            raise ValueError(f"kama_period must be >= 2; got {self.kama_period!r}")

    def generate_signals(self, candles: Sequence[Candle]) -> list[StrategySignal]:
        """Per bar: LONG while the KAMA is rising, SHORT while falling, flat if flat/warming up."""
        ma = kama(OHLCV.from_candles(candles), self.kama_period)
        signals: list[StrategySignal] = []
        for i, candle in enumerate(candles):
            side: Side | None = None
            strength = 0.0
            if i > 0:
                level, prev = float(ma[i]), float(ma[i - 1])
                if math.isfinite(level) and math.isfinite(prev):
                    if level > prev:
                        side, strength = Side.LONG, 1.0  # KAMA rising -> trend up
                    elif level < prev:
                        side, strength = Side.SHORT, 1.0  # KAMA falling -> trend down
            signals.append(
                StrategySignal(
                    asof=candle.timestamp,
                    symbol=candle.symbol,
                    side=side or Side.LONG,  # inert when strength == 0 (flat)
                    strength=strength,
                )
            )
        return signals


def adaptive_ma_slope_spec(params: Mapping[str, float]) -> AdaptiveMaSlopeSpec:
    """Build the slope spec from a parameter mapping (the run_study / CLI ``SpecFactory``)."""
    defaults = AdaptiveMaSlopeSpec()
    return AdaptiveMaSlopeSpec(kama_period=round(params.get("kama_period", defaults.kama_period)))
