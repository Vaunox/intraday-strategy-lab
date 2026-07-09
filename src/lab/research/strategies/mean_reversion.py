"""Mean-reversion StrategySpec (Phase 3, P3.3): intraday-reset z-score fade.

Fades statistically stretched intraday moves back toward a rolling mean: when the
close is more than ``entry_z`` standard deviations above an **intraday-reset** rolling
mean it is faded SHORT (and symmetrically LONG below), the position held until price
reverts to within ``exit_z`` of the mean. The wide-entry / narrow-exit **hysteresis**
mirrors the P3.1 VWAP fade and captures the reversion *toward* the mean, not only the
excursion beyond the entry band (which would be cost-dead).

Same fade *direction* as the P3.1 VWAP mean-reversion (which KILLed), but a DIFFERENT
anchor: this fades deviation from a rolling SMA normalized by rolling sigma (the
Bollinger / z-score anchor), not from intraday VWAP. Distinct, low-prior hypothesis;
see the P3.3 pre-registration.

Deterministic and point-in-time: the z-score at bar ``i`` uses only the current IST
day's bars ``0..i`` (:func:`~lab.data.features.indicators.intraday_zscore`, a full
``lookback``-bar intraday window; INTRADAY-RESET so an overnight gap cannot read as a
huge z and turn the fade into a gap-fade). A signal at bar ``t``'s close fills at bar
``t+1``'s open (Inviolable Rule 2); each day opens flat; the backtester squares off at
the configured MIS cutoff.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from zoneinfo import ZoneInfo

from lab.core.constants import INDIA_TZ
from lab.core.types import BarInterval, Candle, Side, StrategySignal
from lab.data.features.indicators import intraday_zscore
from lab.data.features.ohlcv import OHLCV


@dataclass(frozen=True, slots=True)
class MeanReversionSpec:
    """Fade intraday z-score extremes back toward the rolling mean (mean-reversion)."""

    entry_z: float = 2.0  # |z| beyond this -> enter the fade (2-sigma textbook stretch)
    exit_z: float = 0.5  # |z| back within this of the mean -> exit
    lookback: int = 20  # intraday rolling window for the mean + sigma (fixed, not swept)
    name: str = "mean_reversion"
    interval: BarInterval = BarInterval.MIN_5

    def __post_init__(self) -> None:
        """Fail loudly on a nonsensical band or window."""
        if self.lookback < 2:
            raise ValueError(f"lookback must be >= 2; got {self.lookback!r}")
        if not 0.0 < self.exit_z < self.entry_z:
            raise ValueError(
                f"require 0 < exit_z < entry_z; got exit_z={self.exit_z!r}, entry_z={self.entry_z!r}"
            )

    def generate_signals(self, candles: Sequence[Candle]) -> list[StrategySignal]:
        """Emit one point-in-time target per bar (fade / hold / flat), reset each day."""
        tz = ZoneInfo(INDIA_TZ)
        z = intraday_zscore(OHLCV.from_candles(candles), self.lookback)
        signals: list[StrategySignal] = []
        held: Side | None = None
        current_day = None
        for i, candle in enumerate(candles):
            day = candle.timestamp.astimezone(tz).date()
            if day != current_day:
                held = None  # each day opens flat (intraday square-off, no carry)
                current_day = day
            side, strength = self._target(float(z[i]), held)
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

    def _target(self, z: float, held: Side | None) -> tuple[Side | None, float]:
        """Desired ``(side, strength)`` at a bar given its z-score and state."""
        if not math.isfinite(z):
            return None, 0.0  # window not yet formed (intraday warmup) -> stay flat
        if held is None:
            if z > self.entry_z:
                return Side.SHORT, 1.0  # stretched above the mean -> fade short
            if z < -self.entry_z:
                return Side.LONG, 1.0  # stretched below the mean -> fade long
            return None, 0.0
        if held is Side.SHORT:
            # exit once price has reverted down to within exit_z of the mean
            return (None, 0.0) if z <= self.exit_z else (Side.SHORT, 1.0)
        # held LONG: exit once price has reverted up to within exit_z of the mean
        return (None, 0.0) if z >= -self.exit_z else (Side.LONG, 1.0)


def mean_reversion_spec(params: Mapping[str, float]) -> MeanReversionSpec:
    """Build the spec from a parameter mapping (the run_study / CLI ``SpecFactory``).

    Reads ``entry_z``, ``exit_z``, and ``lookback`` (rounded to int); absent keys fall
    back to the pre-committed defaults. Used so run_study can sweep the +/- one-step
    neighbours (``entry_z``, ``exit_z``) for criterion-6a sensitivity and the PBO
    configuration matrix, charging each variant to the trial ledger.
    """
    defaults = MeanReversionSpec()
    return MeanReversionSpec(
        entry_z=float(params.get("entry_z", defaults.entry_z)),
        exit_z=float(params.get("exit_z", defaults.exit_z)),
        lookback=round(params.get("lookback", defaults.lookback)),
    )
