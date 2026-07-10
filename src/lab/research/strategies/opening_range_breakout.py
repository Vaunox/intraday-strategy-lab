"""Opening-Range-Breakout StrategySpec (Phase 3, P3.11): trade the break of the opening range.

A-priori mechanism: the first ``opening_range_minutes`` of the session form the day's initial
balance (the opening auction's price discovery). A decisive break of that range signals the
day's directional resolution as the auction trends away from balance. Trade the break.

Construction (symmetric, one entry per day, intraday): mark the opening range (running high/low
over the first ``opening_range_minutes``, then fixed). Once the window has CLOSED, the first bar
whose close breaks the range by ``break_buffer`` triggers an entry in the break direction
(``close > OR_high*(1+buffer)`` ⇒ LONG, ``close < OR_low*(1-buffer)`` ⇒ SHORT), ridden to the
MIS square-off (sparse signal — the adapter forward-fills to the day's end, resets flat next
open). ``opening_range`` is point-in-time (running then fixed), so a signal at bar ``i`` uses
only bars ``<= i`` and fills at ``i+1``'s open. Distinct from P3.2 (prior-N-bar intraday range)
and P3.6 (global multi-session Donchian): the level here is the OPENING range specifically. See
the P3.11 pre-registration.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from lab.core.constants import INDIA_TZ
from lab.core.types import BarInterval, Candle, Side, StrategySignal
from lab.data.features.indicators import opening_range
from lab.data.features.ohlcv import OHLCV


@dataclass(frozen=True, slots=True)
class OpeningRangeBreakoutSpec:
    """P3.11 -- break of the first-N-minute opening range, ridden to square-off."""

    opening_range_minutes: int = 30  # window that forms the opening range
    break_buffer: float = 0.001  # fractional buffer beyond the range to confirm a break
    name: str = "opening_range_breakout"
    interval: BarInterval = BarInterval.MIN_5

    def __post_init__(self) -> None:
        """Fail loudly on degenerate window / buffer."""
        if self.opening_range_minutes < 1:
            raise ValueError(
                f"opening_range_minutes must be >= 1; got {self.opening_range_minutes!r}"
            )
        if self.break_buffer < 0.0:
            raise ValueError(f"break_buffer must be >= 0; got {self.break_buffer!r}")

    def generate_signals(self, candles: Sequence[Candle]) -> list[StrategySignal]:
        """Emit at most one opening-range-break entry per day; hold to square-off."""
        tz = ZoneInfo(INDIA_TZ)
        ohlcv = OHLCV.from_candles(candles)
        or_high, or_low = opening_range(ohlcv, self.opening_range_minutes)
        window = timedelta(minutes=self.opening_range_minutes)
        signals: list[StrategySignal] = []
        current_day = None
        session_open: datetime | None = None
        entered = False
        for i, candle in enumerate(candles):
            day = candle.timestamp.astimezone(tz).date()
            if day != current_day:
                current_day = day
                session_open = candle.timestamp
                entered = False
            # only trade once the opening-range window has CLOSED (the range is now fixed)
            if entered or session_open is None or candle.timestamp < session_open + window:
                continue
            side = self._break_side(float(or_high[i]), float(or_low[i]), float(candle.close))
            if side is not None:
                signals.append(
                    StrategySignal(
                        asof=candle.timestamp, symbol=candle.symbol, side=side, strength=1.0
                    )
                )
                entered = True
        return signals

    def _break_side(self, or_high: float, or_low: float, close: float) -> Side | None:
        """A break of the fixed opening range by at least ``break_buffer``."""
        if not (math.isfinite(or_high) and math.isfinite(or_low)):
            return None
        if close > or_high * (1.0 + self.break_buffer):
            return Side.LONG
        if close < or_low * (1.0 - self.break_buffer):
            return Side.SHORT
        return None


def opening_range_breakout_spec(params: Mapping[str, float]) -> OpeningRangeBreakoutSpec:
    """Build the P3.11 spec from a parameter mapping (the run_study / CLI ``SpecFactory``)."""
    defaults = OpeningRangeBreakoutSpec()
    return OpeningRangeBreakoutSpec(
        opening_range_minutes=round(
            params.get("opening_range_minutes", defaults.opening_range_minutes)
        ),
        break_buffer=float(params.get("break_buffer", defaults.break_buffer)),
    )
