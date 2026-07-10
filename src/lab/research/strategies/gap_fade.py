"""Gap-Fade StrategySpec (Phase 3, P3.10b): fade an opening gap that fails to hold.

The **directional twin** of gap-and-go (P3.10) — a both-owed pair (operator ruling 2026-07-10:
a directional opposite must not be direction-cherry-picked). Same qualifying event (a large,
participated opening gap), but the **trigger is rejection, not continuation**: a gap that
**rejects back through the intraday VWAP** (fails to hold the gap side) is faded **toward the
gap fill** (the prior close).

Construction (symmetric, one entry per day, intraday), mirroring `GapAndGoSpec` where sensible:
at each of the first `open_window` bars, if the overnight gap qualifies (`|gap| ≥ gap_threshold`),
participation confirms (`relative_volume ≥ vol_mult`), and price has **rejected to the opposite
side of the intraday VWAP** — gap up but `close < VWAP` ⇒ **SHORT** (fade toward fill), gap down
but `close > VWAP` ⇒ **LONG** — ENTER the fade and ride to the MIS square-off (sparse — the
adapter forward-fills, resets flat next open). The only flip from gap-and-go is the VWAP-side
condition (hold → reject) and the resulting side. `gap` is known at the open, `vwap` is the
intraday cumulative average, `relative_volume` is trailing — all point-in-time, so a signal at
bar `i` uses only bars `≤ i` and fills at `i+1`'s open. See the P3.10b pre-registration.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from zoneinfo import ZoneInfo

from lab.core.constants import INDIA_TZ
from lab.core.types import BarInterval, Candle, Side, StrategySignal
from lab.data.features.indicators import gap, relative_volume, vwap
from lab.data.features.ohlcv import OHLCV


@dataclass(frozen=True, slots=True)
class GapFadeSpec:
    """P3.10b -- fade a qualifying opening gap that REJECTS back through VWAP, toward the fill."""

    gap_threshold: float = 0.010  # |overnight gap| to qualify (1.0%)
    vol_mult: float = 1.2  # relative-volume participation confirmation
    relvol_period: int = 20  # trailing window for relative_volume
    open_window: int = 12  # only enter within the first N bars of the day (~60 min opening play)
    name: str = "gap_fade"
    interval: BarInterval = BarInterval.MIN_5

    def __post_init__(self) -> None:
        """Fail loudly on degenerate thresholds / windows."""
        if self.gap_threshold <= 0.0:
            raise ValueError(f"gap_threshold must be > 0; got {self.gap_threshold!r}")
        if self.vol_mult <= 0.0:
            raise ValueError(f"vol_mult must be > 0; got {self.vol_mult!r}")
        if self.relvol_period < 2:
            raise ValueError(f"relvol_period must be >= 2; got {self.relvol_period!r}")
        if self.open_window < 1:
            raise ValueError(f"open_window must be >= 1; got {self.open_window!r}")

    def generate_signals(self, candles: Sequence[Candle]) -> list[StrategySignal]:
        """Emit at most one gap-fade entry per day; hold (forward-filled) to square-off."""
        tz = ZoneInfo(INDIA_TZ)
        ohlcv = OHLCV.from_candles(candles)
        gap_pct = gap(ohlcv)
        relvol = relative_volume(ohlcv, self.relvol_period)
        vwap_line = vwap(ohlcv)
        signals: list[StrategySignal] = []
        current_day = None
        day_bar = 0
        entered = False
        for i, candle in enumerate(candles):
            day = candle.timestamp.astimezone(tz).date()
            if day != current_day:
                current_day = day
                day_bar = 0
                entered = False
            if not entered and day_bar < self.open_window:
                side = self._entry_side(
                    float(gap_pct[i]), float(relvol[i]), float(vwap_line[i]), float(candle.close)
                )
                if side is not None:
                    signals.append(
                        StrategySignal(
                            asof=candle.timestamp, symbol=candle.symbol, side=side, strength=1.0
                        )
                    )
                    entered = True
            day_bar += 1
        return signals

    def _entry_side(
        self, gap_pct: float, relvol: float, vwap_ref: float, close: float
    ) -> Side | None:
        """A fade entry iff the gap qualifies, volume confirms, and price REJECTED across VWAP."""
        if not (math.isfinite(gap_pct) and math.isfinite(relvol) and math.isfinite(vwap_ref)):
            return None
        if abs(gap_pct) < self.gap_threshold or relvol < self.vol_mult:
            return None
        if gap_pct > 0.0 and close < vwap_ref:
            return Side.SHORT  # gap up but rejected below VWAP -> fade down toward the fill
        if gap_pct < 0.0 and close > vwap_ref:
            return Side.LONG  # gap down but rejected above VWAP -> fade up toward the fill
        return None


def gap_fade_spec(params: Mapping[str, float]) -> GapFadeSpec:
    """Build the P3.10b spec from a parameter mapping (the run_study / CLI ``SpecFactory``)."""
    defaults = GapFadeSpec()
    return GapFadeSpec(
        gap_threshold=float(params.get("gap_threshold", defaults.gap_threshold)),
        vol_mult=float(params.get("vol_mult", defaults.vol_mult)),
    )
