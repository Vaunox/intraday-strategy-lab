"""Pivot-points StrategySpec (Phase 3, P3.5): classic pivot S/R reversion.

Fades price at the classic daily pivot support/resistance levels: when price reaches the
resistance ``R1`` it is faded SHORT (betting the widely-watched level holds and price
reverts toward the central pivot ``P``), and symmetrically LONG at the support ``S1``; the
position is held until price reverts to ``P`` (target) or the intraday square-off.

The levels are the CLASSIC daily pivots derived from the PRIOR completed session's
high/low/close (:func:`~lab.data.features.indicators.pivot` for ``P`` and
:func:`~lab.data.features.indicators.classic_pivot_levels` for ``R1``/``S1``) -- constant
within the day, known at the open, first day NaN. They read ONLY the prior day, never the
current day's HLC, so there is no same-day or future leak (the P3.5 no-lookahead
precondition). A signal at bar ``t``'s close fills at bar ``t+1``'s open (Inviolable Rule
2); each day opens flat; the backtester squares off at the configured MIS cutoff.

The only tunable knobs are the two proximity bands -- the levels themselves are formulaic
(no lookback/window), the smallest overfitting surface in the slate. Economic rationale
(tested, not assumed): classic pivots are widely watched, so the only plausible edge is a
self-fulfilling support/resistance effect; whether it survives the round-trip cost on
large-caps is what the kill-gate decides. See the P3.5 pre-registration.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from zoneinfo import ZoneInfo

from lab.core.constants import INDIA_TZ
from lab.core.types import BarInterval, Candle, Side, StrategySignal
from lab.data.features.indicators import classic_pivot_levels, pivot
from lab.data.features.ohlcv import OHLCV


@dataclass(frozen=True, slots=True)
class PivotReversionSpec:
    """Fade price at the classic pivot R1/S1 levels, targeting the central pivot P."""

    entry_band: float = 0.001  # enter the fade within this fraction of R1/S1
    exit_band: float = 0.001  # exit within this fraction of the central pivot P
    name: str = "pivot_reversion"
    interval: BarInterval = BarInterval.MIN_5

    def __post_init__(self) -> None:
        """Fail loudly on negative proximity bands."""
        if self.entry_band < 0.0:
            raise ValueError(f"entry_band must be >= 0; got {self.entry_band!r}")
        if self.exit_band < 0.0:
            raise ValueError(f"exit_band must be >= 0; got {self.exit_band!r}")

    def generate_signals(self, candles: Sequence[Candle]) -> list[StrategySignal]:
        """Emit one point-in-time target per bar (fade / hold / flat), reset each day."""
        tz = ZoneInfo(INDIA_TZ)
        ohlcv = OHLCV.from_candles(candles)
        p_levels = pivot(ohlcv)
        r1, s1 = classic_pivot_levels(ohlcv)
        signals: list[StrategySignal] = []
        held: Side | None = None
        current_day = None
        for i, candle in enumerate(candles):
            day = candle.timestamp.astimezone(tz).date()
            if day != current_day:
                held = None  # each day opens flat (intraday square-off, no carry)
                current_day = day
            side, strength = self._target(
                candle, float(p_levels[i]), float(r1[i]), float(s1[i]), held
            )
            held = side if strength > 0.0 else None
            signals.append(
                StrategySignal(
                    asof=candle.timestamp,
                    symbol=candle.symbol,
                    side=side or Side.LONG,  # side is inert when strength == 0 (flat)
                    strength=strength,
                )
            )
        return signals

    def _target(
        self, candle: Candle, p: float, r1: float, s1: float, held: Side | None
    ) -> tuple[Side | None, float]:
        """Desired ``(side, strength)`` at a bar given the pivot levels and state."""
        if not (math.isfinite(p) and math.isfinite(r1) and math.isfinite(s1)):
            return None, 0.0  # no prior-day pivot yet (first day) -> stay flat
        if held is None:
            if candle.high >= r1 * (1.0 - self.entry_band):
                return Side.SHORT, 1.0  # reached resistance R1 -> fade (expect rejection)
            if candle.low <= s1 * (1.0 + self.entry_band):
                return Side.LONG, 1.0  # reached support S1 -> fade (expect bounce)
            return None, 0.0
        if held is Side.SHORT:
            # exit once price reverts DOWN to within exit_band of the central pivot P
            return (None, 0.0) if candle.low <= p * (1.0 + self.exit_band) else (Side.SHORT, 1.0)
        # held LONG: exit once price reverts UP to within exit_band of P
        return (None, 0.0) if candle.high >= p * (1.0 - self.exit_band) else (Side.LONG, 1.0)


def pivot_reversion_spec(params: Mapping[str, float]) -> PivotReversionSpec:
    """Build the spec from a parameter mapping (the run_study / CLI ``SpecFactory``).

    Reads ``entry_band`` and ``exit_band``; absent keys fall back to the pre-committed
    defaults. Used so run_study can sweep the +/- one-step neighbours for criterion-6a
    parameter sensitivity and the PBO configuration matrix, charging each variant to the
    trial ledger.
    """
    defaults = PivotReversionSpec()
    return PivotReversionSpec(
        entry_band=float(params.get("entry_band", defaults.entry_band)),
        exit_band=float(params.get("exit_band", defaults.exit_band)),
    )
