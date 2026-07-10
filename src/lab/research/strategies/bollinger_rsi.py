"""P4.3 · Bollinger MR + RSI Extreme — two-factor fade confluence StrategySpec (Phase 4).

Blueprint combination (RESEARCH_FINDINGS §4.2): *fade a Bollinger-band touch only when RSI is
simultaneously at an extreme (momentum exhaustion).* A minimal **AND-confluence** of two
Phase-3-family exhaustion signals — the Bollinger-band touch and an RSI extreme — with the
**fewest knobs** (`bb_num_std`, `rsi_oversold`; the two periods are fixed textbook). The two legs
are **correlated** (both fire on the same stretch move), so this is a *reinforcing* confluence,
not a near-mutually-exclusive one — it is NOT expected to be degenerate.

Construction (symmetric, per-bar with hysteresis, intraday): ENTER a fade only when **both**
agree — `close < lower band` AND `RSI < rsi_oversold` ⇒ LONG (oversold); `close > upper band` AND
`RSI > 100 - rsi_oversold` ⇒ SHORT (overbought). EXIT on reversion to the Bollinger **middle
band**. Each day opens flat; MIS square-off. `talib` Bollinger and RSI are trailing (confirmed
prefix-invariant), so the signal at bar `i` uses only bars `≤ i` and fills at `i+1`'s open; the
RSI leg limits gap-driven entries (a lone gap bar rarely drives RSI to an extreme). See the P4.3
pre-registration.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from zoneinfo import ZoneInfo

from lab.core.constants import INDIA_TZ
from lab.core.types import BarInterval, Candle, Side, StrategySignal
from lab.data.features.indicators import bollinger, rsi
from lab.data.features.ohlcv import OHLCV


@dataclass(frozen=True, slots=True)
class BollingerRsiSpec:
    """P4.3 -- fade a Bollinger-band touch confirmed by an RSI extreme; exit to the middle band."""

    bb_num_std: float = 2.0  # Bollinger band width in std devs
    rsi_oversold: float = 30.0  # RSI extreme; the overbought mirror is 100 - rsi_oversold
    bb_period: int = 20  # Bollinger window (fixed textbook)
    rsi_period: int = 14  # RSI window (fixed textbook)
    name: str = "bollinger_rsi"
    interval: BarInterval = BarInterval.MIN_5

    def __post_init__(self) -> None:
        """Fail loudly on degenerate parameters."""
        if self.bb_num_std <= 0.0:
            raise ValueError(f"bb_num_std must be > 0; got {self.bb_num_std!r}")
        if not 0.0 < self.rsi_oversold < 50.0:
            raise ValueError(f"require 0 < rsi_oversold < 50; got {self.rsi_oversold!r}")
        if self.bb_period < 2 or self.rsi_period < 2:
            raise ValueError("bb_period and rsi_period must be >= 2")

    def generate_signals(self, candles: Sequence[Candle]) -> list[StrategySignal]:
        """Emit a fade / hold / flat target per bar; enter only on the two-signal confluence."""
        tz = ZoneInfo(INDIA_TZ)
        ohlcv = OHLCV.from_candles(candles)
        upper, middle, lower = bollinger(ohlcv, self.bb_period, self.bb_num_std)
        strength_index = rsi(ohlcv, self.rsi_period)
        rsi_overbought = 100.0 - self.rsi_oversold
        signals: list[StrategySignal] = []
        held: Side | None = None
        current_day = None
        for i, candle in enumerate(candles):
            day = candle.timestamp.astimezone(tz).date()
            if day != current_day:
                held = None  # each day opens flat
                current_day = day
            side, strength = self._target(
                float(candle.close),
                float(upper[i]),
                float(middle[i]),
                float(lower[i]),
                float(strength_index[i]),
                held,
                rsi_overbought,
            )
            held = side if strength > 0.0 else None
            signals.append(
                StrategySignal(
                    asof=candle.timestamp,
                    symbol=candle.symbol,
                    side=side or Side.LONG,  # inert when strength == 0 (flat)
                    strength=strength,
                )
            )
        return signals

    def _target(
        self,
        close: float,
        upper: float,
        middle: float,
        lower: float,
        rsi_now: float,
        held: Side | None,
        rsi_overbought: float,
    ) -> tuple[Side | None, float]:
        """Enter a fade on the band+RSI confluence; exit on reversion to the middle band."""
        if not all(math.isfinite(x) for x in (close, upper, middle, lower, rsi_now)):
            return None, 0.0
        if held is None:
            if close < lower and rsi_now < self.rsi_oversold:
                return Side.LONG, 1.0  # oversold band touch + oversold RSI -> fade up
            if close > upper and rsi_now > rsi_overbought:
                return Side.SHORT, 1.0
            return None, 0.0
        if held is Side.LONG:
            return (None, 0.0) if close >= middle else (Side.LONG, 1.0)  # exit at the mean
        return (None, 0.0) if close <= middle else (Side.SHORT, 1.0)


def bollinger_rsi_spec(params: Mapping[str, float]) -> BollingerRsiSpec:
    """Build the P4.3 spec from a parameter mapping (the run_study / CLI ``SpecFactory``)."""
    defaults = BollingerRsiSpec()
    return BollingerRsiSpec(
        bb_num_std=float(params.get("bb_num_std", defaults.bb_num_std)),
        rsi_oversold=float(params.get("rsi_oversold", defaults.rsi_oversold)),
    )
