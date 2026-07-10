"""Bull-Flag StrategySpec (Phase 3, P3.12): impulse, tight consolidation, breakout continuation.

A-priori mechanism: a sharp directional impulse (committed buyers) followed by a shallow, tight
consolidation (profit-taking without real selling) is a pause, not a reversal; the move resumes
when price breaks out of the consolidation. Symmetric (bull flag long / bear flag short).

Construction (causal, intraday). At bar ``i``, over three contiguous, SAME-DAY windows:
- **impulse** — the ``impulse_lookback``-bar move ending ``flag_lookback`` bars ago:
  ``impulse_return = close[i-M] / close[i-M-K] - 1`` (``K`` = impulse, ``M`` = flag length);
- **consolidation** — the prior ``M`` bars ``[i-M .. i-1]`` (``prior_donchian``, excludes the
  current bar): TIGHT iff ``consol_range <= tight_frac * |impulse price move|``;
- **breakout** — ``close[i]`` beyond the consolidation extreme (``> consol_high`` bull /
  ``< consol_low`` bear).

All three windows must lie within one IST day (a **same-day guard**), so an overnight gap can
never masquerade as the impulse (which would make this a gap-and-go, P3.10). On a completed
pattern, emit an entry in the breakout direction, ridden to the MIS square-off (sparse — the
adapter forward-fills, resets flat next open). ``prior_donchian`` and indexed closes are
point-in-time, so a signal at bar ``i`` uses only bars ``<= i`` and fills at ``i+1``'s open.
**Largest parameter surface in the slate (K, X, M, tight_frac)** — flagged in the
pre-registration. See the P3.12 pre-registration.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from zoneinfo import ZoneInfo

from lab.core.constants import INDIA_TZ
from lab.core.types import BarInterval, Candle, Side, StrategySignal
from lab.data.features.indicators import prior_donchian
from lab.data.features.ohlcv import OHLCV


@dataclass(frozen=True, slots=True)
class BullFlagSpec:
    """P3.12 -- impulse + tight consolidation + breakout, ridden to square-off (symmetric)."""

    impulse_lookback: int = 6  # K: bars in the impulse leg (~30 min)
    impulse_threshold: float = 0.010  # X: minimum |impulse return| to qualify (1.0%)
    flag_lookback: int = 6  # M: bars in the consolidation (~30 min)
    tight_frac: float = 0.5  # consolidation range <= tight_frac * |impulse price move|
    name: str = "bull_flag"
    interval: BarInterval = BarInterval.MIN_5

    def __post_init__(self) -> None:
        """Fail loudly on degenerate windows / thresholds."""
        if self.impulse_lookback < 1:
            raise ValueError(f"impulse_lookback must be >= 1; got {self.impulse_lookback!r}")
        if self.flag_lookback < 2:
            raise ValueError(f"flag_lookback must be >= 2; got {self.flag_lookback!r}")
        if self.impulse_threshold <= 0.0:
            raise ValueError(f"impulse_threshold must be > 0; got {self.impulse_threshold!r}")
        if self.tight_frac <= 0.0:
            raise ValueError(f"tight_frac must be > 0; got {self.tight_frac!r}")

    def generate_signals(self, candles: Sequence[Candle]) -> list[StrategySignal]:
        """Emit a breakout-direction entry on each completed flag; hold (forward-filled)."""
        tz = ZoneInfo(INDIA_TZ)
        ohlcv = OHLCV.from_candles(candles)
        consol_high, consol_low = prior_donchian(ohlcv, self.flag_lookback)
        close = ohlcv.close
        k, m = self.impulse_lookback, self.flag_lookback
        signals: list[StrategySignal] = []
        for i, candle in enumerate(candles):
            if i < m + k:
                continue
            # whole pattern (impulse + consolidation) must lie within one IST day
            if (
                candles[i - m - k].timestamp.astimezone(tz).date()
                != candle.timestamp.astimezone(tz).date()
            ):
                continue
            side = self._pattern_side(
                float(close[i - m - k]),
                float(close[i - m]),
                float(consol_high[i]),
                float(consol_low[i]),
                float(candle.close),
            )
            if side is not None:
                signals.append(
                    StrategySignal(
                        asof=candle.timestamp, symbol=candle.symbol, side=side, strength=1.0
                    )
                )
        return signals

    def _pattern_side(
        self, c_start: float, c_peak: float, consol_high: float, consol_low: float, close: float
    ) -> Side | None:
        """A completed bull/bear flag: qualifying impulse, tight consolidation, and a breakout."""
        if not (math.isfinite(consol_high) and math.isfinite(consol_low)) or c_start <= 0.0:
            return None
        impulse_return = c_peak / c_start - 1.0
        impulse_move = abs(c_peak - c_start)
        tight = (consol_high - consol_low) <= self.tight_frac * impulse_move
        if not tight:
            return None
        if impulse_return >= self.impulse_threshold and close > consol_high:
            return Side.LONG  # bull flag: up-impulse, tight pause, break up
        if impulse_return <= -self.impulse_threshold and close < consol_low:
            return Side.SHORT  # bear flag: down-impulse, tight pause, break down
        return None


def bull_flag_spec(params: Mapping[str, float]) -> BullFlagSpec:
    """Build the P3.12 spec from a parameter mapping (the run_study / CLI ``SpecFactory``)."""
    defaults = BullFlagSpec()
    return BullFlagSpec(
        impulse_threshold=float(params.get("impulse_threshold", defaults.impulse_threshold)),
        tight_frac=float(params.get("tight_frac", defaults.tight_frac)),
    )
