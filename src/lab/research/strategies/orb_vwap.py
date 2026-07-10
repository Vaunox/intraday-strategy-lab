"""P4.2 · ORB + VWAP Confirmation — two-factor confluence StrategySpec (Phase 4).

Blueprint combination (RESEARCH_FINDINGS §4.2): *take the opening-range break only if price is on
the confirming side of VWAP.* A minimal **AND-confluence** of two Phase-3 primitives — the
opening-range breakout (P3.11) and the VWAP side (P3.1) — with the **fewest knobs**
(`opening_range_minutes`, `break_buffer`).

Construction (symmetric, sparse, intraday): after the opening-range window has CLOSED, the first
bar whose close **breaks the range** by `break_buffer` **AND** is on the **confirming side of
VWAP** enters in the break direction (`close > OR_high·(1+buffer)` and `close > VWAP` ⇒ LONG;
`close < OR_low·(1-buffer)` and `close < VWAP` ⇒ SHORT), ridden to the MIS square-off (sparse —
the adapter forward-fills, resets flat next open). `opening_range` (running then fixed) and VWAP
(intraday-cumulative) are point-in-time, so a signal at bar `i` uses only bars `≤ i` and fills at
`i+1`'s open. See the P4.2 pre-registration.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from lab.core.constants import INDIA_TZ
from lab.core.types import BarInterval, Candle, Side, StrategySignal
from lab.data.features.indicators import opening_range, vwap
from lab.data.features.ohlcv import OHLCV


@dataclass(frozen=True, slots=True)
class OrbVwapSpec:
    """P4.2 -- opening-range break confirmed by the VWAP side; ride to square-off."""

    opening_range_minutes: int = 30  # window that forms the opening range
    break_buffer: float = 0.001  # fractional buffer beyond the range to confirm a break
    name: str = "orb_vwap"
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
        """Emit at most one VWAP-confirmed opening-range break per day; hold to square-off."""
        tz = ZoneInfo(INDIA_TZ)
        ohlcv = OHLCV.from_candles(candles)
        or_high, or_low = opening_range(ohlcv, self.opening_range_minutes)
        vwap_line = vwap(ohlcv)
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
            if entered or session_open is None or candle.timestamp < session_open + window:
                continue  # only after the opening-range window has closed
            side = self._entry_side(
                float(or_high[i]), float(or_low[i]), float(vwap_line[i]), float(candle.close)
            )
            if side is not None:
                signals.append(
                    StrategySignal(
                        asof=candle.timestamp, symbol=candle.symbol, side=side, strength=1.0
                    )
                )
                entered = True
        return signals

    def _entry_side(
        self, or_high: float, or_low: float, vwap_ref: float, close: float
    ) -> Side | None:
        """A range break confirmed by the VWAP side."""
        if not (math.isfinite(or_high) and math.isfinite(or_low) and math.isfinite(vwap_ref)):
            return None
        if close > or_high * (1.0 + self.break_buffer) and close > vwap_ref:
            return Side.LONG
        if close < or_low * (1.0 - self.break_buffer) and close < vwap_ref:
            return Side.SHORT
        return None


def orb_vwap_spec(params: Mapping[str, float]) -> OrbVwapSpec:
    """Build the P4.2 spec from a parameter mapping (the run_study / CLI ``SpecFactory``)."""
    defaults = OrbVwapSpec()
    return OrbVwapSpec(
        opening_range_minutes=round(
            params.get("opening_range_minutes", defaults.opening_range_minutes)
        ),
        break_buffer=float(params.get("break_buffer", defaults.break_buffer)),
    )
