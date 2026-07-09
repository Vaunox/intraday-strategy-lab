"""VWAP StrategySpecs (Phase 3, P3.1): mean-reversion fade (V1) and decisive cross (V2).

Fades intraday deviations from the volume-weighted average price back toward it:
when price stretches beyond ``entry_threshold`` above VWAP it is faded SHORT (and
symmetrically LONG below), the position held until price reverts to within
``exit_threshold`` of VWAP. The wide-entry / narrow-exit **hysteresis** is
deliberate — it captures the reversion *toward* VWAP rather than only the small
excursion beyond the entry band (which would be cost-dead).

Deterministic and point-in-time: the VWAP deviation at bar ``i`` uses only that
trading day's bars ``0..i`` (intraday-cumulative VWAP, daily reset —
:func:`~lab.data.features.indicators.vwap_deviation`); a signal at bar ``t``'s
close fills at bar ``t+1``'s open in the backtester (Inviolable Rule 2). Each day
opens flat; the backtester squares off at the configured MIS cutoff.

Economic rationale (tested, not assumed): VWAP is the intraday execution
benchmark institutions size and grade fills against. VWAP-target algorithms supply
liquidity that pulls price back toward VWAP (buying below it, selling above it), so
extreme intraday deviations should partially revert. Whether that reversion
survives the full Indian round-trip cost is exactly what the kill-gate decides.

The cross variant (:class:`VwapCrossSpec`, V2) is the OPPOSITE directional bet on
the same phenomenon: price that crosses VWAP *decisively* -- beyond a
``cross_threshold`` band that suppresses whipsaw at VWAP -- is bet to CONTINUE past
it, not revert, riding the continuation until the opposite decisive cross flips it.
Both variants are pre-registered in P3.1 and judged once by the same unchanged panel
gate; per the pre-registration, if BOTH pass that is a contradiction to investigate.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from zoneinfo import ZoneInfo

from lab.core.constants import INDIA_TZ
from lab.core.types import BarInterval, Candle, Side, StrategySignal
from lab.data.features.indicators import vwap_deviation
from lab.data.features.ohlcv import OHLCV


@dataclass(frozen=True, slots=True)
class VwapMeanReversionSpec:
    """Fade intraday VWAP deviations back toward VWAP (mean-reversion, hysteresis)."""

    entry_threshold: float = 0.004  # |close/VWAP - 1| beyond this -> enter the fade
    exit_threshold: float = 0.001  # deviation back within this of VWAP -> exit
    name: str = "vwap_mean_reversion"
    interval: BarInterval = BarInterval.MIN_5

    def __post_init__(self) -> None:
        """Fail loudly on a nonsensical band: exit must sit inside entry, both > 0."""
        if not 0.0 < self.exit_threshold < self.entry_threshold:
            raise ValueError(
                "require 0 < exit_threshold < entry_threshold; got "
                f"exit={self.exit_threshold!r}, entry={self.entry_threshold!r}"
            )

    def generate_signals(self, candles: Sequence[Candle]) -> list[StrategySignal]:
        """Emit one point-in-time target per bar (fade / hold / flat), reset each day."""
        tz = ZoneInfo(INDIA_TZ)
        deviation = vwap_deviation(OHLCV.from_candles(candles))
        signals: list[StrategySignal] = []
        held: Side | None = None
        current_day = None
        for i, candle in enumerate(candles):
            day = candle.timestamp.astimezone(tz).date()
            if day != current_day:
                held = None  # each day opens flat (intraday square-off, no carry)
                current_day = day
            side, strength = self._target(float(deviation[i]), held)
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

    def _target(self, deviation: float, held: Side | None) -> tuple[Side | None, float]:
        """Desired ``(side, strength)`` at a bar given its VWAP deviation and state."""
        if not math.isfinite(deviation):
            return None, 0.0  # no VWAP yet (zero cumulative volume) -> stay flat
        if held is None:
            if deviation > self.entry_threshold:
                return Side.SHORT, 1.0  # stretched above VWAP -> fade short
            if deviation < -self.entry_threshold:
                return Side.LONG, 1.0  # stretched below VWAP -> fade long
            return None, 0.0
        if held is Side.SHORT:
            # exit once price has reverted down to within exit_threshold of VWAP
            return (None, 0.0) if deviation <= self.exit_threshold else (Side.SHORT, 1.0)
        # held LONG: exit once price has reverted up to within exit_threshold of VWAP
        return (None, 0.0) if deviation >= -self.exit_threshold else (Side.LONG, 1.0)


def vwap_mean_reversion_spec(params: Mapping[str, float]) -> VwapMeanReversionSpec:
    """Build the spec from a parameter mapping (the run_study / CLI ``SpecFactory``).

    Reads ``entry_threshold`` and ``exit_threshold``; absent keys fall back to the
    pre-committed defaults. Used so run_study can sweep the +/- one-step parameter
    neighbours for criterion-6a parameter sensitivity and the PBO configuration
    matrix, and charge each variant to the trial ledger.
    """
    defaults = VwapMeanReversionSpec()
    return VwapMeanReversionSpec(
        entry_threshold=float(params.get("entry_threshold", defaults.entry_threshold)),
        exit_threshold=float(params.get("exit_threshold", defaults.exit_threshold)),
    )


@dataclass(frozen=True, slots=True)
class VwapCrossSpec:
    """Trade the direction of a decisive VWAP cross; ride the continuation (trend).

    The opposite directional bet to :class:`VwapMeanReversionSpec` on the SAME
    phenomenon (behaviour around VWAP): where the fade bets that stretched price
    reverts, the cross bets that price which crosses VWAP *decisively* -- beyond a
    ``cross_threshold`` band that suppresses whipsaw *at* VWAP -- CONTINUES past it.
    From flat, a deviation above ``+cross_threshold`` goes LONG and below
    ``-cross_threshold`` goes SHORT; the position then rides the continuation and
    flips only on the opposite decisive cross (a deviation beyond the far band, which
    price can only reach by crossing back through VWAP), until the intraday
    square-off. A small deviation inside the ``±cross_threshold`` band never flips an
    open position -- that band is the "real cross, not noise" guard.

    Deterministic and point-in-time: identical VWAP-deviation substrate as the fade
    (intraday-cumulative VWAP, daily reset --
    :func:`~lab.data.features.indicators.vwap_deviation`); a signal at bar ``t``'s
    close fills at bar ``t+1``'s open (Inviolable Rule 2). Each day opens flat; the
    backtester squares off at the configured MIS cutoff.

    Economic rationale (tested, not assumed): a decisive cross of VWAP on real
    participation marks a shift in the session's fair-value consensus -- resting
    benchmark liquidity on the far side is absorbed -- so price continues past VWAP.
    Whether that continuation survives the false-cross rate and the full Indian
    round-trip cost is exactly what the kill-gate decides. See the P3.1
    pre-registration (V2, the owed cross variant).
    """

    cross_threshold: float = 0.002  # |close/VWAP - 1| past VWAP marking a cross decisive
    name: str = "vwap_cross"
    interval: BarInterval = BarInterval.MIN_5

    def __post_init__(self) -> None:
        """Fail loudly on a non-positive band: the whipsaw guard must be > 0."""
        if not self.cross_threshold > 0.0:
            raise ValueError(f"cross_threshold must be > 0; got {self.cross_threshold!r}")

    def generate_signals(self, candles: Sequence[Candle]) -> list[StrategySignal]:
        """Emit one point-in-time target per bar (cross / hold / flat), reset each day."""
        tz = ZoneInfo(INDIA_TZ)
        deviation = vwap_deviation(OHLCV.from_candles(candles))
        signals: list[StrategySignal] = []
        held: Side | None = None
        current_day = None
        for i, candle in enumerate(candles):
            day = candle.timestamp.astimezone(tz).date()
            if day != current_day:
                held = None  # each day opens flat (intraday square-off, no carry)
                current_day = day
            side, strength = self._target(float(deviation[i]), held)
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

    def _target(self, deviation: float, held: Side | None) -> tuple[Side | None, float]:
        """Desired ``(side, strength)`` at a bar given its VWAP deviation and state.

        Beyond the far band the target is (re)set to the cross direction -- this one
        rule enters from flat, holds a continuation, AND flips on the opposite
        decisive cross (price reaches the far band only by crossing back through
        VWAP). Inside the ``±cross_threshold`` band an OPEN position rides on (no
        whipsaw exit at VWAP); flat stays flat until the first decisive cross.
        """
        if not math.isfinite(deviation):
            return None, 0.0  # no VWAP yet (zero cumulative volume) -> stay flat
        if deviation > self.cross_threshold:
            return Side.LONG, 1.0  # decisive upside cross -> long the continuation
        if deviation < -self.cross_threshold:
            return Side.SHORT, 1.0  # decisive downside cross -> short the continuation
        return (held, 1.0) if held is not None else (None, 0.0)  # inside band -> hold current


def vwap_cross_spec(params: Mapping[str, float]) -> VwapCrossSpec:
    """Build the spec from a parameter mapping (the run_study / CLI ``SpecFactory``).

    Reads ``cross_threshold``; an absent key falls back to the pre-committed default.
    Used so run_study can sweep the +/- one-step neighbour for criterion-6a parameter
    sensitivity and the PBO configuration matrix, charging each variant to the ledger.
    """
    defaults = VwapCrossSpec()
    return VwapCrossSpec(
        cross_threshold=float(params.get("cross_threshold", defaults.cross_threshold)),
    )
