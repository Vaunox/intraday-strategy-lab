"""P4.5 · Pivot Confluence + MACD — level + momentum-trigger StrategySpec (Phase 4).

Blueprint combination (RESEARCH_FINDINGS §4.2): *act at a pivot level only when a MACD crossover
gives a momentum trigger.* A minimal **AND-confluence** of the classic pivot S/R levels (P3.5)
and a MACD-crossover event — with the **fewest knobs** (`entry_band`; MACD is fixed textbook
12/26/9). Direction: a **bullish MACD crossover near support S1** ⇒ LONG (a confirmed bounce);
a **bearish crossover near resistance R1** ⇒ SHORT.

**Degeneracy watch (flagged):** "price near a level" and "a MACD crossover on the same bar" are
two events that may rarely coincide — a possible P3.9-style near-mutual-exclusivity. The §6
landscape reports the trade count; if the base config is degenerate (a hollow trade count), it is
flagged, not recorded as a hollow number.

Construction (symmetric, sparse, intraday): on a bar where `close` is within `entry_band` of S1
and the MACD line crosses **above** its signal ⇒ LONG; within `entry_band` of R1 and the MACD
line crosses **below** ⇒ SHORT. Ride to the MIS square-off (sparse — the adapter forward-fills,
resets flat next open). Classic pivots are from the completed prior day and MACD is trailing
(both point-in-time), so a signal at bar `i` uses only bars `≤ i` and fills at `i+1`'s open. See
the P4.5 pre-registration.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from lab.core.types import BarInterval, Candle, Side, StrategySignal
from lab.data.features.indicators import classic_pivot_levels, macd
from lab.data.features.ohlcv import OHLCV

_MACD_FAST, _MACD_SLOW, _MACD_SIGNAL = 12, 26, 9  # fixed textbook MACD


@dataclass(frozen=True, slots=True)
class PivotMacdSpec:
    """P4.5 -- a MACD crossover near a classic pivot level; ride to square-off."""

    entry_band: float = 0.002  # fractional proximity to the pivot level to act (0.2%)
    name: str = "pivot_macd"
    interval: BarInterval = BarInterval.MIN_5

    def __post_init__(self) -> None:
        """Fail loudly on a degenerate band."""
        if self.entry_band <= 0.0:
            raise ValueError(f"entry_band must be > 0; got {self.entry_band!r}")

    def generate_signals(self, candles: Sequence[Candle]) -> list[StrategySignal]:
        """Emit a LONG at support / SHORT at resistance on a confirming MACD crossover."""
        ohlcv = OHLCV.from_candles(candles)
        r1, s1 = classic_pivot_levels(ohlcv)
        macd_line, signal_line, _hist = macd(ohlcv, _MACD_FAST, _MACD_SLOW, _MACD_SIGNAL)
        signals: list[StrategySignal] = []
        for i, candle in enumerate(candles):
            if i == 0:
                continue
            m_now, m_prev = float(macd_line[i]), float(macd_line[i - 1])
            s_now, s_prev = float(signal_line[i]), float(signal_line[i - 1])
            if not all(math.isfinite(x) for x in (m_now, m_prev, s_now, s_prev)):
                continue
            cross_up = m_prev <= s_prev and m_now > s_now
            cross_down = m_prev >= s_prev and m_now < s_now
            close, s1_i, r1_i = float(candle.close), float(s1[i]), float(r1[i])
            side: Side | None = None
            if cross_up and math.isfinite(s1_i) and abs(close - s1_i) <= self.entry_band * s1_i:
                side = Side.LONG  # bullish momentum trigger at support
            elif cross_down and math.isfinite(r1_i) and abs(close - r1_i) <= self.entry_band * r1_i:
                side = Side.SHORT  # bearish momentum trigger at resistance
            if side is not None:
                signals.append(
                    StrategySignal(
                        asof=candle.timestamp, symbol=candle.symbol, side=side, strength=1.0
                    )
                )
        return signals


def pivot_macd_spec(params: Mapping[str, float]) -> PivotMacdSpec:
    """Build the P4.5 spec from a parameter mapping (the run_study / CLI ``SpecFactory``)."""
    defaults = PivotMacdSpec()
    return PivotMacdSpec(entry_band=float(params.get("entry_band", defaults.entry_band)))
