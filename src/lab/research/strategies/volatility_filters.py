"""Volatility-based-filter StrategySpecs (Phase 3, P3.8): two INDEPENDENT regime studies.

A volatility filter has no standalone edge -- it only means something applied on top of a
directional signal. So P3.8 tests the filter thesis as two self-contained, blind,
regime-conditional strategies (regime + signal defined together), run as INDEPENDENT
standalone studies (each its own verdict; NOT a both-owed dichotomy -- they are compatible,
disjoint-regime strategies):

- **C1 (:class:`VolExpansionBreakoutSpec`)** -- in an EXPANDING-vol regime, trade the intraday
  range breakout. A-priori: a breakout mechanically requires volatility expansion to have a
  move to break into.
- **C2 (:class:`VolContractionReversionSpec`)** -- in a CONTRACTING-vol regime, fade the
  intraday z-score back toward the mean. A-priori: mean-reversion requires a quiet range to
  oscillate within.

The shared regime is the causal, self-normalizing ATR ratio
(:func:`~lab.data.features.indicators.atr_ratio` = ATR(short)/ATR(long); ``> 1`` expanding,
``< 1`` contracting). Point-in-time: a signal at bar ``t``'s close fills at ``t+1``'s open;
each day opens flat; the backtester squares off at the MIS cutoff. See the P3.8
pre-registration.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from zoneinfo import ZoneInfo

from lab.core.constants import INDIA_TZ
from lab.core.types import BarInterval, Candle, Side, StrategySignal
from lab.data.features.indicators import atr_ratio, intraday_donchian, intraday_zscore
from lab.data.features.ohlcv import OHLCV


@dataclass(frozen=True, slots=True)
class VolExpansionBreakoutSpec:
    """C1 -- intraday breakout, gated to an EXPANDING-volatility regime (ATR ratio > 1)."""

    breakout_lookback: int = 20  # prior-N-bar intraday range (the directional trigger)
    atr_short: int = 20  # recent ATR window (intraday)
    atr_long: int = 100  # baseline ATR window (> atr_short)
    name: str = "vol_expansion_breakout"
    interval: BarInterval = BarInterval.MIN_5

    def __post_init__(self) -> None:
        """Fail loudly on degenerate/mis-ordered windows."""
        if self.breakout_lookback < 2:
            raise ValueError(f"breakout_lookback must be >= 2; got {self.breakout_lookback!r}")
        if not 2 <= self.atr_short < self.atr_long:
            raise ValueError(
                "require 2 <= atr_short < atr_long; got "
                f"short={self.atr_short!r}, long={self.atr_long!r}"
            )

    def generate_signals(self, candles: Sequence[Candle]) -> list[StrategySignal]:
        """Emit a breakout LONG/SHORT only when volatility is expanding; hold forward-filled."""
        ohlcv = OHLCV.from_candles(candles)
        ratio = atr_ratio(ohlcv, self.atr_short, self.atr_long)
        upper, lower = intraday_donchian(ohlcv, self.breakout_lookback)
        signals: list[StrategySignal] = []
        for i, candle in enumerate(candles):
            r = float(ratio[i])
            if not math.isfinite(r) or r <= 1.0:
                continue  # out of the expanding-vol regime -> no new entry
            prior_high, prior_low = float(upper[i]), float(lower[i])
            side: Side | None = None
            if math.isfinite(prior_high) and candle.close > prior_high:
                side = Side.LONG  # breakout of today's range in expanding vol
            elif math.isfinite(prior_low) and candle.close < prior_low:
                side = Side.SHORT
            if side is not None:
                signals.append(
                    StrategySignal(
                        asof=candle.timestamp, symbol=candle.symbol, side=side, strength=1.0
                    )
                )
        return signals


def vol_expansion_breakout_spec(params: Mapping[str, float]) -> VolExpansionBreakoutSpec:
    """Build the C1 spec from a parameter mapping (the run_study / CLI ``SpecFactory``)."""
    defaults = VolExpansionBreakoutSpec()
    return VolExpansionBreakoutSpec(
        breakout_lookback=round(params.get("breakout_lookback", defaults.breakout_lookback)),
        atr_long=round(params.get("atr_long", defaults.atr_long)),
    )


@dataclass(frozen=True, slots=True)
class VolContractionReversionSpec:
    """C2 -- intraday z-score fade, gated to a CONTRACTING-volatility regime (ATR ratio < 1)."""

    entry_z: float = 2.0  # |z| beyond this -> enter the fade
    exit_z: float = 0.5  # |z| back within this of the mean -> exit
    lookback: int = 20  # intraday z-score window
    atr_short: int = 20  # recent ATR window (intraday)
    atr_long: int = 100  # baseline ATR window (> atr_short)
    name: str = "vol_contraction_reversion"
    interval: BarInterval = BarInterval.MIN_5

    def __post_init__(self) -> None:
        """Fail loudly on degenerate bands/windows."""
        if self.lookback < 2:
            raise ValueError(f"lookback must be >= 2; got {self.lookback!r}")
        if not 0.0 < self.exit_z < self.entry_z:
            raise ValueError(
                f"require 0 < exit_z < entry_z; got exit_z={self.exit_z!r}, entry_z={self.entry_z!r}"
            )
        if not 2 <= self.atr_short < self.atr_long:
            raise ValueError(
                "require 2 <= atr_short < atr_long; got "
                f"short={self.atr_short!r}, long={self.atr_long!r}"
            )

    def generate_signals(self, candles: Sequence[Candle]) -> list[StrategySignal]:
        """Emit a z-score fade / hold / flat per bar; ENTER only while volatility is contracting."""
        tz = ZoneInfo(INDIA_TZ)
        ohlcv = OHLCV.from_candles(candles)
        ratio = atr_ratio(ohlcv, self.atr_short, self.atr_long)
        z = intraday_zscore(ohlcv, self.lookback)
        signals: list[StrategySignal] = []
        held: Side | None = None
        current_day = None
        for i, candle in enumerate(candles):
            day = candle.timestamp.astimezone(tz).date()
            if day != current_day:
                held = None  # each day opens flat
                current_day = day
            side, strength = self._target(float(z[i]), float(ratio[i]), held)
            held = side if strength > 0.0 else None
            signals.append(
                StrategySignal(
                    asof=candle.timestamp,
                    symbol=candle.symbol,
                    side=side or Side.LONG,  # inert when strength == 0 (flat)
                    strength=strength,
                )
            )
        return signals

    def _target(self, z: float, ratio: float, held: Side | None) -> tuple[Side | None, float]:
        """Fade in contracting vol; once faded, exit on reversion toward the mean."""
        if not math.isfinite(z):
            return None, 0.0
        if held is None:
            # ENTER a fade only while volatility is contracting (ATR ratio < 1)
            if math.isfinite(ratio) and ratio < 1.0:
                if z > self.entry_z:
                    return Side.SHORT, 1.0
                if z < -self.entry_z:
                    return Side.LONG, 1.0
            return None, 0.0
        # already in a fade: exit on reversion to within exit_z of the mean (regime-independent)
        if held is Side.SHORT:
            return (None, 0.0) if z <= self.exit_z else (Side.SHORT, 1.0)
        return (None, 0.0) if z >= -self.exit_z else (Side.LONG, 1.0)


def vol_contraction_reversion_spec(params: Mapping[str, float]) -> VolContractionReversionSpec:
    """Build the C2 spec from a parameter mapping (the run_study / CLI ``SpecFactory``)."""
    defaults = VolContractionReversionSpec()
    return VolContractionReversionSpec(
        entry_z=float(params.get("entry_z", defaults.entry_z)),
        atr_long=round(params.get("atr_long", defaults.atr_long)),
    )
