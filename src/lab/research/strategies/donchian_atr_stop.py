"""P4.6 · Donchian Breakout + ATR Stop — breakout entry with a volatility-scaled stop (Phase 4).

Blueprint combination (RESEARCH_FINDINGS §4.2): *Donchian channel breakout with a
volatility-scaled (ATR-multiple) stop-loss for risk control.* A minimal composition of the
global multi-session Donchian breakout (P3.6) with an **ATR-trailing-stop exit** — with the
**fewest knobs** (`channel_lookback`, `atr_mult`; the ATR window is fixed textbook). The ATR
stop is *risk control on the exit*, not a second entry condition — it is expected to improve
drawdowns, not the entry's weak intraday base (an honest a-priori read).

Construction (symmetric, per-bar stateful, intraday): ENTER on a break of the prior-N-bar GLOBAL
Donchian channel (`prior_donchian`; `close > upper` ⇒ LONG, `close < lower` ⇒ SHORT). Once in,
trail a stop at `atr_mult x ATR` from the best close (a long's stop only ratchets up); EXIT when
the close crosses the trailing stop. Each day opens flat; MIS square-off. `prior_donchian`
(excludes the current bar) and `talib.ATR` are trailing (prefix-invariant), so the per-bar target
uses only bars `≤ i` and fills at `i+1`'s open. See the P4.6 pre-registration.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from zoneinfo import ZoneInfo

from lab.core.constants import INDIA_TZ
from lab.core.types import BarInterval, Candle, Side, StrategySignal
from lab.data.features.indicators import atr, prior_donchian
from lab.data.features.ohlcv import OHLCV


@dataclass(frozen=True, slots=True)
class DonchianAtrStopSpec:
    """P4.6 -- global Donchian breakout entry, ATR-trailing-stop exit; flat each day."""

    channel_lookback: int = 55  # prior N-bar global Donchian channel (the break trigger)
    atr_mult: float = 2.0  # trailing stop distance in ATR multiples
    atr_period: int = 14  # ATR window (fixed textbook)
    name: str = "donchian_atr_stop"
    interval: BarInterval = BarInterval.MIN_5

    def __post_init__(self) -> None:
        """Fail loudly on degenerate parameters."""
        if self.channel_lookback < 2:
            raise ValueError(f"channel_lookback must be >= 2; got {self.channel_lookback!r}")
        if self.atr_mult <= 0.0:
            raise ValueError(f"atr_mult must be > 0; got {self.atr_mult!r}")
        if self.atr_period < 2:
            raise ValueError(f"atr_period must be >= 2; got {self.atr_period!r}")

    def generate_signals(self, candles: Sequence[Candle]) -> list[StrategySignal]:
        """Emit a Donchian-breakout position per bar; exit on the ATR trailing stop."""
        tz = ZoneInfo(INDIA_TZ)
        ohlcv = OHLCV.from_candles(candles)
        upper, lower = prior_donchian(ohlcv, self.channel_lookback)
        atr_line = atr(ohlcv, self.atr_period)
        signals: list[StrategySignal] = []
        held: Side | None = None
        stop = math.nan
        current_day = None
        for i, candle in enumerate(candles):
            day = candle.timestamp.astimezone(tz).date()
            if day != current_day:
                held, stop, current_day = None, math.nan, day  # each day opens flat
            close, a = float(candle.close), float(atr_line[i])
            # (1) manage an open position: trail the stop, exit if the close crosses it
            if held is Side.LONG:
                if math.isfinite(a):
                    stop = max(stop, close - self.atr_mult * a)
                if close < stop:
                    held = None
            elif held is Side.SHORT:
                if math.isfinite(a):
                    stop = min(stop, close + self.atr_mult * a)
                if close > stop:
                    held = None
            # (2) if flat, enter on a fresh Donchian break
            if held is None and math.isfinite(a):
                hi, lo = float(upper[i]), float(lower[i])
                if math.isfinite(hi) and close > hi:
                    held, stop = Side.LONG, close - self.atr_mult * a
                elif math.isfinite(lo) and close < lo:
                    held, stop = Side.SHORT, close + self.atr_mult * a
            signals.append(
                StrategySignal(
                    asof=candle.timestamp,
                    symbol=candle.symbol,
                    side=held or Side.LONG,  # inert when strength == 0 (flat)
                    strength=1.0 if held is not None else 0.0,
                )
            )
        return signals


def donchian_atr_stop_spec(params: Mapping[str, float]) -> DonchianAtrStopSpec:
    """Build the P4.6 spec from a parameter mapping (the run_study / CLI ``SpecFactory``)."""
    defaults = DonchianAtrStopSpec()
    return DonchianAtrStopSpec(
        channel_lookback=round(params.get("channel_lookback", defaults.channel_lookback)),
        atr_mult=float(params.get("atr_mult", defaults.atr_mult)),
    )
