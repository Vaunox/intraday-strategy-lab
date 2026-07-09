"""Donchian-channel StrategySpec (Phase 3, P3.6): global multi-session breakout.

Enters in the break direction of the prior N-bar GLOBAL Donchian channel: LONG when the
close exceeds the prior-N-bar high, SHORT when it breaks the prior-N-bar low. It rides the
continuation intraday -- held (forward-filled) until an opposite-channel breakout flips it
or the intraday square-off closes it; each day opens flat, so the *position* is intraday
even though the *channel* is multi-session.

Deterministic and point-in-time: the channel is
:func:`~lab.data.features.indicators.prior_donchian` -- the max-high / min-low over the
prior ``channel_lookback`` bars EXCLUDING the current bar, GLOBAL (no day reset), so an
early-session break references the PRIOR session's extreme (a multi-session level). This is
the deliberate distinction from P3.2 breakout (which used an intraday-RESET range plus a
volume filter): here the level persists across the overnight gap and there is NO volume
filter, so the tested question is "does breaking a longer, multi-session high/low continue
intraday?" -- a distinct hypothesis. A signal at bar ``t``'s close fills at bar ``t+1``'s
open (Inviolable Rule 2).

Economic rationale (tested, not assumed): classic Donchian/Turtle trend-following bets a
break of an N-bar extreme continues. Its pedigree is on daily/higher timeframes; whether a
multi-session break continues *intraday* net-of-cost is what the kill-gate decides. All
prior continuation bets (P3.1 V2, P3.2) KILLed, so the prior is low. See the P3.6
pre-registration.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from lab.core.types import BarInterval, Candle, Side, StrategySignal
from lab.data.features.indicators import prior_donchian
from lab.data.features.ohlcv import OHLCV


@dataclass(frozen=True, slots=True)
class DonchianBreakoutSpec:
    """Break of the prior N-bar GLOBAL Donchian channel; ride the continuation (unfiltered)."""

    channel_lookback: int = 55  # prior N-bar global high/low channel (Turtle-classic entry)
    name: str = "donchian_breakout"
    interval: BarInterval = BarInterval.MIN_5

    def __post_init__(self) -> None:
        """Fail loudly on a degenerate channel window."""
        if self.channel_lookback < 2:
            raise ValueError(f"channel_lookback must be >= 2; got {self.channel_lookback!r}")

    def generate_signals(self, candles: Sequence[Candle]) -> list[StrategySignal]:
        """Emit a LONG/SHORT signal on each channel breakout; the hold is forward-filled."""
        upper, lower = prior_donchian(OHLCV.from_candles(candles), self.channel_lookback)
        signals: list[StrategySignal] = []
        for i, candle in enumerate(candles):
            # Prior N-bar high/low EXCLUDING bar i (global); NaN until N prior bars exist.
            prior_high = float(upper[i])
            prior_low = float(lower[i])
            side: Side | None = None
            if math.isfinite(prior_high) and candle.close > prior_high:
                side = Side.LONG  # broke the prior N-bar high -> continuation
            elif math.isfinite(prior_low) and candle.close < prior_low:
                side = Side.SHORT  # broke the prior N-bar low -> continuation
            if side is not None:
                signals.append(
                    StrategySignal(
                        asof=candle.timestamp, symbol=candle.symbol, side=side, strength=1.0
                    )
                )
        return signals


def donchian_breakout_spec(params: Mapping[str, float]) -> DonchianBreakoutSpec:
    """Build the spec from a parameter mapping (the run_study / CLI ``SpecFactory``).

    Reads ``channel_lookback`` (rounded to int); an absent key falls back to the
    pre-committed default. Used so run_study can sweep the +/- one-step neighbour for
    criterion-6a parameter sensitivity and the PBO configuration matrix, charging each
    variant to the trial ledger.
    """
    defaults = DonchianBreakoutSpec()
    return DonchianBreakoutSpec(
        channel_lookback=round(params.get("channel_lookback", defaults.channel_lookback)),
    )
