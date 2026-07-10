"""Momentum-pullback StrategySpec (Phase 3, P3.9): buy the resumption after a shallow dip.

A single, self-contained CONTINUATION mechanism (not a dichotomy): in an established trend,
price does not travel in a straight line -- it retraces (profit-taking, weak-hand shakeouts)
and then resumes. Rather than *chasing* a breakout (P3.2/P3.6) this enters on the **resumption
of momentum after a pullback**, for a lower-risk entry within the same trend.

Construction (symmetric long/short, intraday positions -- flat at each day open, squared off at
the MIS cutoff):

- **Trend regime** -- ``close`` vs a trailing ``SMA(trend_period)``. ``close > SMA`` = uptrend
  (long side active); ``close < SMA`` = downtrend (short side active).
- **Pullback + resumption** -- an ``RSI(rsi_period)`` dip and recovery. LONG: in an uptrend,
  once RSI has dipped below ``rsi_pullback`` (the shallow oversold-within-uptrend), ENTER when
  RSI **crosses back up** through ``rsi_pullback`` on the SAME day (momentum resuming). SHORT is
  the mirror through ``100 - rsi_pullback``.
- **Exit** -- the trend flips (``close`` crosses the SMA against the position) OR RSI reaches the
  opposite extreme (a long banks at ``>= 100 - rsi_pullback``); plus the backtester's EOD
  square-off.

Point-in-time: ``talib.SMA`` / ``talib.RSI`` are trailing (confirmed prefix-invariant), so the
signal at bar ``i`` uses only bars ``<= i``; a signal at bar ``i``'s close fills at ``i+1``'s
open. The resumption cross is gated to bars **within the same IST day** so an overnight RSI jump
across the level is never read as an intraday pullback-resumption (the gap-as-signal trap the
``intraday_zscore`` docstring warns about). See the P3.9 pre-registration.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from zoneinfo import ZoneInfo

from lab.core.constants import INDIA_TZ
from lab.core.types import BarInterval, Candle, Side, StrategySignal
from lab.data.features.indicators import rsi, sma
from lab.data.features.ohlcv import OHLCV


@dataclass(frozen=True, slots=True)
class MomentumPullbackSpec:
    """P3.9 -- with-trend pullback: enter on momentum RESUMPTION after a shallow RSI dip."""

    trend_period: int = 50  # SMA window defining the trend regime (close vs SMA)
    rsi_period: int = 14  # RSI window (textbook default)
    rsi_pullback: float = 30.0  # RSI dip level for the entry cross; mirror is 100 - rsi_pullback
    name: str = "momentum_pullback"
    interval: BarInterval = BarInterval.MIN_5

    def __post_init__(self) -> None:
        """Fail loudly on degenerate windows / an out-of-range pullback level."""
        if self.trend_period < 2:
            raise ValueError(f"trend_period must be >= 2; got {self.trend_period!r}")
        if self.rsi_period < 2:
            raise ValueError(f"rsi_period must be >= 2; got {self.rsi_period!r}")
        if not 0.0 < self.rsi_pullback < 50.0:
            # must sit below the RSI midline so the pullback level and its 100-x mirror are distinct
            raise ValueError(f"require 0 < rsi_pullback < 50; got {self.rsi_pullback!r}")

    def generate_signals(self, candles: Sequence[Candle]) -> list[StrategySignal]:
        """Emit an entry-on-resumption / hold / flat target per bar; enter only in-trend."""
        tz = ZoneInfo(INDIA_TZ)
        ohlcv = OHLCV.from_candles(candles)
        trend = sma(ohlcv, self.trend_period)
        strength_index = rsi(ohlcv, self.rsi_period)
        upper = 100.0 - self.rsi_pullback
        signals: list[StrategySignal] = []
        held: Side | None = None
        current_day = None
        for i, candle in enumerate(candles):
            day = candle.timestamp.astimezone(tz).date()
            new_day = day != current_day
            if new_day:
                held = None  # each day opens flat
                current_day = day
            # prior RSI only counts for a cross when it is the SAME day's prior bar, so an
            # overnight gap across the level is never read as an intraday resumption.
            rsi_prev = float(strength_index[i - 1]) if (i > 0 and not new_day) else math.nan
            side, strength = self._target(
                float(trend[i]),
                float(strength_index[i]),
                rsi_prev,
                float(candle.close),
                held,
                upper,
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
        trend: float,
        rsi_now: float,
        rsi_prev: float,
        close: float,
        held: Side | None,
        upper: float,
    ) -> tuple[Side | None, float]:
        """Enter on the in-trend RSI resumption cross; exit on trend-flip or opposite extreme."""
        if not (math.isfinite(trend) and math.isfinite(rsi_now)):
            return None, 0.0
        uptrend = close > trend
        downtrend = close < trend
        if held is None:
            if math.isfinite(rsi_prev):
                # LONG: uptrend AND RSI crossed UP through the pullback level (dip resuming)
                if uptrend and rsi_prev < self.rsi_pullback <= rsi_now:
                    return Side.LONG, 1.0
                # SHORT: downtrend AND RSI crossed DOWN through the mirror level
                if downtrend and rsi_prev > upper >= rsi_now:
                    return Side.SHORT, 1.0
            return None, 0.0
        if held is Side.LONG:
            # exit the long on a trend flip or once RSI reaches the overbought mirror
            if (not uptrend) or rsi_now >= upper:
                return None, 0.0
            return Side.LONG, 1.0
        # held SHORT: exit on a trend flip or once RSI reaches the oversold level
        if (not downtrend) or rsi_now <= self.rsi_pullback:
            return None, 0.0
        return Side.SHORT, 1.0


def momentum_pullback_spec(params: Mapping[str, float]) -> MomentumPullbackSpec:
    """Build the P3.9 spec from a parameter mapping (the run_study / CLI ``SpecFactory``)."""
    defaults = MomentumPullbackSpec()
    return MomentumPullbackSpec(
        trend_period=round(params.get("trend_period", defaults.trend_period)),
        rsi_pullback=float(params.get("rsi_pullback", defaults.rsi_pullback)),
    )
