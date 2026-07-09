"""Volume-filtered breakout StrategySpec (Phase 3, P3.2).

Enters in the direction of a decisive break of the prior N-bar range, filtered by
a relative-volume surge: LONG when the close breaks above the prior-N-bar high on
volume greater than ``volume_mult`` x its recent average, SHORT on the symmetric
downside break. It rides the continuation intraday — the position is held
(forward-filled) until an opposite filtered break flips it or the intraday
square-off closes it; each day opens flat, so a low-volume drift through a level
(the classic false break) never triggers, and no position carries overnight.

Deterministic and point-in-time: the prior-N-bar high/low is the rolling Donchian
channel AS OF THE PREVIOUS bar (``donchian(...)[i-1]``, excludes the current bar),
and relative volume uses only prior volumes, so bar ``i`` uses only bars ``0..i``;
a signal at bar ``t``'s close fills at bar ``t+1``'s open (Inviolable Rule 2).

Economic rationale (tested, not assumed): a trading range is a temporary
equilibrium; a decisive break on expanding participation — the volume filter is
the crux — signals the equilibrium has resolved (resting liquidity absorbed, stops
and momentum traders drawn in), so price CONTINUES in the break direction. This is
the opposite directional bet to VWAP-fade (P3.1); see the pre-registration.
Whether the continuation survives the false-break rate and the full round-trip
cost is what the kill-gate decides.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from lab.core.types import BarInterval, Candle, Side, StrategySignal
from lab.data.features.indicators import donchian, relative_volume
from lab.data.features.ohlcv import OHLCV


@dataclass(frozen=True, slots=True)
class BreakoutSpec:
    """Enter on a volume-filtered break of the prior N-bar range; ride the continuation."""

    breakout_lookback: int = 20  # prior N-bar high/low that defines the range
    volume_mult: float = 1.5  # break must occur on volume > this x its recent average
    volume_period: int = 20  # lookback for the relative-volume average (fixed, not swept)
    name: str = "breakout"
    interval: BarInterval = BarInterval.MIN_5

    def __post_init__(self) -> None:
        """Fail loudly on nonsensical parameters."""
        if self.breakout_lookback < 2:
            raise ValueError(f"breakout_lookback must be >= 2; got {self.breakout_lookback!r}")
        if self.volume_mult <= 0.0:
            raise ValueError(f"volume_mult must be > 0; got {self.volume_mult!r}")
        if self.volume_period < 2:
            raise ValueError(f"volume_period must be >= 2; got {self.volume_period!r}")

    def generate_signals(self, candles: Sequence[Candle]) -> list[StrategySignal]:
        """Emit a LONG/SHORT signal on each filtered break; the hold is forward-filled."""
        ohlcv = OHLCV.from_candles(candles)
        upper, lower = donchian(ohlcv, self.breakout_lookback)
        rel_vol = relative_volume(ohlcv, self.volume_period)
        signals: list[StrategySignal] = []
        for i, candle in enumerate(candles):
            if i < 1:
                continue
            rv = float(rel_vol[i])
            if not math.isfinite(rv) or rv <= self.volume_mult:
                continue  # no participation surge -> not a filtered break
            prior_high = float(upper[i - 1])  # rolling N-bar high AS OF the previous bar
            prior_low = float(lower[i - 1])
            side: Side | None = None
            if math.isfinite(prior_high) and candle.close > prior_high:
                side = Side.LONG  # decisive upside break on volume -> continuation
            elif math.isfinite(prior_low) and candle.close < prior_low:
                side = Side.SHORT  # decisive downside break on volume -> continuation
            if side is not None:
                signals.append(
                    StrategySignal(
                        asof=candle.timestamp, symbol=candle.symbol, side=side, strength=1.0
                    )
                )
        return signals


def breakout_spec(params: Mapping[str, float]) -> BreakoutSpec:
    """Build the spec from a parameter mapping (the run_study / CLI ``SpecFactory``).

    Reads ``breakout_lookback`` (rounded to int) and ``volume_mult``; absent keys
    fall back to the pre-committed defaults. Used so run_study can sweep the +/-
    one-step neighbours for criterion-6a parameter sensitivity and the PBO
    configuration matrix, charging each variant to the trial ledger.
    """
    defaults = BreakoutSpec()
    return BreakoutSpec(
        breakout_lookback=round(params.get("breakout_lookback", defaults.breakout_lookback)),
        volume_mult=float(params.get("volume_mult", defaults.volume_mult)),
    )
