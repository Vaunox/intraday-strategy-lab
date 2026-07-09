"""Reversal StrategySpec (Phase 3, P3.4): swing-failure (failed-breakout) fade.

Fades a FAILED intraday breakout: when a bar pokes above the prior intraday swing high
by at least ``break_buffer`` (a genuine, trap-inducing penetration) but then closes back
BELOW it, the break has failed -- the traders who chased it are trapped and their exits
fuel a reversal -- so the spec enters SHORT (and symmetrically LONG on a failed downside
break). It rides the reversal (forward-filled) until an opposite failed break flips it or
the intraday square-off closes it.

The swing level is the CAUSAL, intraday-reset prior-N-bar high/low
(:func:`~lab.data.features.indicators.intraday_donchian`) -- the same already-proven
primitive breakout uses (excludes the current bar, resets each IST day, NaN at the day's
first bar). This deliberately SIDESTEPS the confirmed-swing-pivot lookahead (a centred
pivot needs *future* bars to confirm -- the exact leak that manufactures a phantom edge):
the failure is detected SAME-BAR from the prior level and bar ``i``'s own high/low/close,
all known at bar ``i``'s close. A signal at bar ``t``'s close fills at bar ``t+1``'s open
(Inviolable Rule 2); each day opens flat; the backtester squares off at the MIS cutoff.

This is the OPPOSITE bet to breakout (P3.2), which enters when ``close > prior_high`` (the
break holds -> continuation LONG); here the break must FAIL (``high > prior_high`` but
``close < prior_high``) -> reversal SHORT. Whether failed breaks reverse net-of-cost is
what the kill-gate decides. See the P3.4 pre-registration.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from lab.core.types import BarInterval, Candle, Side, StrategySignal
from lab.data.features.indicators import intraday_donchian
from lab.data.features.ohlcv import OHLCV


@dataclass(frozen=True, slots=True)
class ReversalSpec:
    """Fade a failed breakout of the prior intraday swing level (swing-failure reversal)."""

    swing_lookback: int = 20  # prior N-bar intraday high/low that defines the swing level
    break_buffer: float = 0.001  # poke must exceed the swing by >= this to be a genuine break
    name: str = "reversal"
    interval: BarInterval = BarInterval.MIN_5

    def __post_init__(self) -> None:
        """Fail loudly on nonsensical parameters."""
        if self.swing_lookback < 2:
            raise ValueError(f"swing_lookback must be >= 2; got {self.swing_lookback!r}")
        if self.break_buffer < 0.0:
            raise ValueError(f"break_buffer must be >= 0; got {self.break_buffer!r}")

    def generate_signals(self, candles: Sequence[Candle]) -> list[StrategySignal]:
        """Emit a LONG/SHORT signal on each swing-failure; the hold is forward-filled."""
        ohlcv = OHLCV.from_candles(candles)
        upper, lower = intraday_donchian(ohlcv, self.swing_lookback)
        signals: list[StrategySignal] = []
        for i, candle in enumerate(candles):
            # Prior swing high/low WITHIN today, as of bar i (excludes i); NaN at the day's
            # first bars -> no reference level -> no signal. All inputs known at bar i close.
            prior_high = float(upper[i])
            prior_low = float(lower[i])
            side: Side | None = None
            if (
                math.isfinite(prior_high)
                and candle.high >= prior_high * (1.0 + self.break_buffer)
                and candle.close < prior_high
            ):
                side = Side.SHORT  # failed upside break (poked above, closed back below) -> fade
            elif (
                math.isfinite(prior_low)
                and candle.low <= prior_low * (1.0 - self.break_buffer)
                and candle.close > prior_low
            ):
                side = Side.LONG  # failed downside break (poked below, closed back above) -> fade
            if side is not None:
                signals.append(
                    StrategySignal(
                        asof=candle.timestamp, symbol=candle.symbol, side=side, strength=1.0
                    )
                )
        return signals


def reversal_spec(params: Mapping[str, float]) -> ReversalSpec:
    """Build the spec from a parameter mapping (the run_study / CLI ``SpecFactory``).

    Reads ``swing_lookback`` (rounded to int) and ``break_buffer``; absent keys fall back
    to the pre-committed defaults. Used so run_study can sweep the +/- one-step neighbours
    for criterion-6a parameter sensitivity and the PBO configuration matrix, charging each
    variant to the trial ledger.
    """
    defaults = ReversalSpec()
    return ReversalSpec(
        swing_lookback=round(params.get("swing_lookback", defaults.swing_lookback)),
        break_buffer=float(params.get("break_buffer", defaults.break_buffer)),
    )
