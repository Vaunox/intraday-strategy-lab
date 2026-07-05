"""Event-driven backtester (Phase 2, P2.1).

Non-negotiable realism (Part III Layer 2): decisions made on bar *t*'s close are
filled at bar *t+1*'s **open** (never same-bar); positions **square off intraday**
at the session's last bar (no overnight carry); every round trip pays the full
Indian cost model + slippage. A signal on a day's last bar does not carry
overnight — each day starts flat.

The output is a series of round-trip :class:`Trade` observations (each with its
entry/exit times, so the purged CV can operate on them) whose net returns feed
the Sharpe/CPCV machinery.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import numpy.typing as npt

from lab.core.constants import INDIA_TZ
from lab.core.types import Candle, Side
from lab.research.validation.costs import CostModel

FloatArray = npt.NDArray[np.float64]

#: Representative per-trade order value (rupees) at which cost fractions are
#: evaluated, so the brokerage per-order cap applies realistically.
DEFAULT_NOTIONAL_PER_TRADE = 100_000.0


@dataclass(frozen=True, slots=True)
class Trade:
    """One realized round-trip trade, costs included."""

    side: Side
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    gross_return: float  # signed fractional return (side-adjusted, pre-cost)
    cost_fraction: float

    @property
    def net_return(self) -> float:
        """Fractional return after the modeled round-trip cost."""
        return self.gross_return - self.cost_fraction


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """The trades produced by a run, plus convenient array views."""

    trades: tuple[Trade, ...]

    @property
    def net_returns(self) -> FloatArray:
        """Per-trade net returns (the return series the Sharpe/CPCV math uses)."""
        return np.array([t.net_return for t in self.trades], dtype=np.float64)

    @property
    def entry_times(self) -> tuple[datetime, ...]:
        """Entry time of each trade (observation start, for purged CV)."""
        return tuple(t.entry_time for t in self.trades)

    @property
    def exit_times(self) -> tuple[datetime, ...]:
        """Exit time of each trade (observation end, for purged CV)."""
        return tuple(t.exit_time for t in self.trades)


def _target_side(target: float) -> Side | None:
    if target > 0:
        return Side.LONG
    if target < 0:
        return Side.SHORT
    return None


def run_backtest(
    candles: Sequence[Candle],
    target_positions: Sequence[float],
    cost_model: CostModel,
    *,
    notional_per_trade: float = DEFAULT_NOTIONAL_PER_TRADE,
    timezone: str = INDIA_TZ,
) -> BacktestResult:
    """Simulate ``target_positions`` over ``candles`` with next-bar-open fills.

    Args:
        candles: Ascending decision bars.
        target_positions: Target position per bar (sign = side, 0 = flat),
            decided at each bar's close.
        cost_model: The Indian cost model applied to every round trip.
        notional_per_trade: Order value at which the cost fraction is evaluated.
        timezone: IST timezone for session/day boundaries.

    Returns:
        The round-trip trades produced by the simulation.
    """
    if len(candles) != len(target_positions):
        raise ValueError("candles and target_positions must have equal length")
    tz = ZoneInfo(timezone)
    cost_fraction = cost_model.round_trip_cost_fraction(notional_per_trade)

    trades: list[Trade] = []
    side: Side | None = None
    entry_price = 0.0
    entry_time: datetime | None = None

    def close(exit_price: float, exit_time: datetime) -> None:
        nonlocal side, entry_time
        if side is None or entry_time is None:
            raise RuntimeError("close() called with no open position")
        direction = 1.0 if side is Side.LONG else -1.0
        gross = direction * (exit_price / entry_price - 1.0)
        trades.append(
            Trade(side, entry_time, exit_time, entry_price, exit_price, gross, cost_fraction)
        )
        side = None
        entry_time = None

    n = len(candles)
    for t in range(n):
        candle = candles[t]
        day = candle.timestamp.astimezone(tz).date()
        same_day_as_prev = t >= 1 and candles[t - 1].timestamp.astimezone(tz).date() == day
        is_last_of_day = t == n - 1 or candles[t + 1].timestamp.astimezone(tz).date() != day

        # Execute the decision from the previous (same-day) bar's close at this open.
        if same_day_as_prev:
            desired = _target_side(target_positions[t - 1])
            if desired is not side:
                if side is not None:
                    close(candle.open, candle.timestamp)
                if desired is not None:
                    side = desired
                    entry_price = candle.open
                    entry_time = candle.timestamp

        # Intraday square-off: force flat at the day's last bar (exit at its close).
        if is_last_of_day and side is not None:
            close(candle.close, candle.timestamp)

    return BacktestResult(tuple(trades))
