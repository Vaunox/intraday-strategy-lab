"""P4.1 · VWAP + Breakout + Volume Surge — three-way confluence StrategySpec (Phase 4).

Blueprint combination (RESEARCH_FINDINGS §4.2): *take a range break only when it happens on the
trend side of VWAP AND on a relative-volume surge.* A minimal **AND-confluence** of three
Phase-3 primitives — the intraday-reset range break (P3.2), the VWAP side (P3.1), and the
volume surge (P3.2's filter) — with the **fewest knobs** (`breakout_lookback`, `vol_mult`; the
volume window is fixed). No knob is added to rescue; confluence is high-surface and the gate
deflates by the honest trial count.

Construction (symmetric, sparse, intraday): LONG when the close **breaks above** the prior-N-bar
intraday range (`intraday_donchian`) **AND** is **above VWAP** (trend side) **AND** the bar's
`relative_volume ≥ vol_mult` (surge); SHORT is the mirror. Ride the continuation to the MIS
square-off (the adapter forward-fills the entry, resets flat next open). All three legs are
point-in-time (VWAP intraday-cumulative, `intraday_donchian` excludes the current bar, relative
volume is trailing), so a signal at bar `i` uses only bars `≤ i` and fills at `i+1`'s open. See
the P4.1 pre-registration.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from lab.core.types import BarInterval, Candle, Side, StrategySignal
from lab.data.features.indicators import intraday_donchian, relative_volume, vwap
from lab.data.features.ohlcv import OHLCV


@dataclass(frozen=True, slots=True)
class VwapBreakoutVolumeSpec:
    """P4.1 -- range break on the trend side of VWAP AND on a volume surge; ride to square-off."""

    breakout_lookback: int = 20  # prior N-bar intraday range (the break trigger)
    vol_mult: float = 1.5  # break must occur on volume >= this x its recent average
    volume_period: int = 20  # trailing window for relative_volume (fixed, not swept)
    name: str = "vwap_breakout_volume"
    interval: BarInterval = BarInterval.MIN_5

    def __post_init__(self) -> None:
        """Fail loudly on degenerate parameters."""
        if self.breakout_lookback < 2:
            raise ValueError(f"breakout_lookback must be >= 2; got {self.breakout_lookback!r}")
        if self.vol_mult <= 0.0:
            raise ValueError(f"vol_mult must be > 0; got {self.vol_mult!r}")
        if self.volume_period < 2:
            raise ValueError(f"volume_period must be >= 2; got {self.volume_period!r}")

    def generate_signals(self, candles: Sequence[Candle]) -> list[StrategySignal]:
        """Emit a confluence breakout LONG/SHORT; hold (forward-filled) to square-off."""
        ohlcv = OHLCV.from_candles(candles)
        vwap_line = vwap(ohlcv)
        upper, lower = intraday_donchian(ohlcv, self.breakout_lookback)
        relvol = relative_volume(ohlcv, self.volume_period)
        signals: list[StrategySignal] = []
        for i, candle in enumerate(candles):
            rv = float(relvol[i])
            if not math.isfinite(rv) or rv < self.vol_mult:
                continue  # no volume surge -> no confluence
            vw, close = float(vwap_line[i]), float(candle.close)
            hi, lo = float(upper[i]), float(lower[i])
            side: Side | None = None
            if math.isfinite(hi) and math.isfinite(vw) and close > hi and close > vw:
                side = Side.LONG  # break up, on the trend side of VWAP, on a surge
            elif math.isfinite(lo) and math.isfinite(vw) and close < lo and close < vw:
                side = Side.SHORT
            if side is not None:
                signals.append(
                    StrategySignal(
                        asof=candle.timestamp, symbol=candle.symbol, side=side, strength=1.0
                    )
                )
        return signals


def vwap_breakout_volume_spec(params: Mapping[str, float]) -> VwapBreakoutVolumeSpec:
    """Build the P4.1 spec from a parameter mapping (the run_study / CLI ``SpecFactory``)."""
    defaults = VwapBreakoutVolumeSpec()
    return VwapBreakoutVolumeSpec(
        breakout_lookback=round(params.get("breakout_lookback", defaults.breakout_lookback)),
        vol_mult=float(params.get("vol_mult", defaults.vol_mult)),
    )
