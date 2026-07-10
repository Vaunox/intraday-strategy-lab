"""Scalping StrategySpecs (Phase 3, P3.13): fast micro in/out -- a BOTH-OWED dichotomy.

Scalping takes many small, fast trades on the finest bars, capturing either micro
**mean-reversion** (fade the last bar's move) or micro **momentum** (chase it). The two
directions are a genuine both-owed dichotomy (per the handoff) -- so BOTH are drafted, as exact
opposites: on every triggered bar they take opposite sides (divergence is definitional, not the
subtle P3.7 case). Extremely cost-sensitive by nature.

- **MR (:class:`ScalpMeanReversionSpec`)** -- fade: last-bar return ``> +threshold`` ⇒ SHORT,
  ``< -threshold`` ⇒ LONG.
- **Momentum (:class:`ScalpMomentumSpec`)** -- chase: ``> +threshold`` ⇒ LONG, ``< -threshold``
  ⇒ SHORT.

Per-bar target from the SAME-DAY last-bar return (``close[i]/close[i-1]-1``; the day's first bar
has no valid prior-bar return ⇒ flat, so an overnight gap is never scalped). Held one bar (the
next bar's target overrides), so turnover is very high. **Frequency caveat: the archive holds
5-minute bars only** -- scalping's natural 1-3 min bars are unavailable, so this is a
coarser-than-natural proxy and cost-death is the expected honest §6 landing (the catalog
includes P3.13 to demonstrate the cost wall). See the P3.13 pre-registration.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from zoneinfo import ZoneInfo

from lab.core.constants import INDIA_TZ
from lab.core.types import BarInterval, Candle, Side, StrategySignal


def _same_day_returns(candles: Sequence[Candle], tz: ZoneInfo) -> list[float]:
    """Last-bar fractional return per bar; NaN at each day's first bar (no cross-gap scalp)."""
    out = [math.nan] * len(candles)
    prev_close = math.nan
    current_day = None
    for i, candle in enumerate(candles):
        day = candle.timestamp.astimezone(tz).date()
        if day != current_day:
            current_day = day
            prev_close = math.nan  # day's first bar: no valid prior-bar return
        if math.isfinite(prev_close) and prev_close > 0.0:
            out[i] = float(candle.close) / prev_close - 1.0
        prev_close = float(candle.close)
    return out


def _scalp_signals(
    candles: Sequence[Candle], threshold: float, *, fade: bool
) -> list[StrategySignal]:
    """Per-bar micro fade/chase target from the same-day last-bar return."""
    tz = ZoneInfo(INDIA_TZ)
    returns = _same_day_returns(candles, tz)
    signals: list[StrategySignal] = []
    for candle, ret in zip(candles, returns, strict=True):
        side: Side | None = None
        if math.isfinite(ret):
            if ret > threshold:
                side = Side.SHORT if fade else Side.LONG  # fade an up-move / chase it
            elif ret < -threshold:
                side = Side.LONG if fade else Side.SHORT
        signals.append(
            StrategySignal(
                asof=candle.timestamp,
                symbol=candle.symbol,
                side=side or Side.LONG,  # inert when strength == 0 (flat)
                strength=1.0 if side is not None else 0.0,
            )
        )
    return signals


@dataclass(frozen=True, slots=True)
class ScalpMeanReversionSpec:
    """P3.13 (MR) -- fade the last bar's micro move; exit into the next bar."""

    entry_threshold: float = 0.002  # |last-bar return| to fade (0.2%)
    name: str = "scalp_mean_reversion"
    interval: BarInterval = BarInterval.MIN_5

    def __post_init__(self) -> None:
        if self.entry_threshold <= 0.0:
            raise ValueError(f"entry_threshold must be > 0; got {self.entry_threshold!r}")

    def generate_signals(self, candles: Sequence[Candle]) -> list[StrategySignal]:
        """Per-bar fade of the same-day last-bar micro move."""
        return _scalp_signals(candles, self.entry_threshold, fade=True)


@dataclass(frozen=True, slots=True)
class ScalpMomentumSpec:
    """P3.13 (momentum) -- chase the last bar's micro move; exit into the next bar."""

    entry_threshold: float = 0.002  # |last-bar return| to chase (0.2%)
    name: str = "scalp_momentum"
    interval: BarInterval = BarInterval.MIN_5

    def __post_init__(self) -> None:
        if self.entry_threshold <= 0.0:
            raise ValueError(f"entry_threshold must be > 0; got {self.entry_threshold!r}")

    def generate_signals(self, candles: Sequence[Candle]) -> list[StrategySignal]:
        """Per-bar chase of the same-day last-bar micro move."""
        return _scalp_signals(candles, self.entry_threshold, fade=False)


def scalp_mean_reversion_spec(params: Mapping[str, float]) -> ScalpMeanReversionSpec:
    """Build the P3.13-MR spec from a parameter mapping (the CLI ``SpecFactory``)."""
    defaults = ScalpMeanReversionSpec()
    return ScalpMeanReversionSpec(
        entry_threshold=float(params.get("entry_threshold", defaults.entry_threshold))
    )


def scalp_momentum_spec(params: Mapping[str, float]) -> ScalpMomentumSpec:
    """Build the P3.13-momentum spec from a parameter mapping (the CLI ``SpecFactory``)."""
    defaults = ScalpMomentumSpec()
    return ScalpMomentumSpec(
        entry_threshold=float(params.get("entry_threshold", defaults.entry_threshold))
    )
