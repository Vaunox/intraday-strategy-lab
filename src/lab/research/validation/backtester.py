"""Event-driven backtester (Phase 2, P2.1).

Non-negotiable realism (Part III Layer 2): decisions made on bar *t*'s close are
filled at bar *t+1*'s **open** (never same-bar); positions **square off intraday**
at the configured MIS cutoff (``square_off``, e.g. 15:20) — or the day's last bar
when no cutoff is given (no overnight carry); every round trip pays the full
Indian cost model + slippage. A signal on a day's last bar does not carry
overnight — each day starts flat.

The output is a series of round-trip :class:`Trade` observations (each with its
entry/exit times, so the purged CV can operate on them) whose net returns feed
the Sharpe/CPCV machinery.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, time
from zoneinfo import ZoneInfo

import numpy as np
import numpy.typing as npt

from lab.core.constants import INDIA_TZ
from lab.core.types import Candle, Side
from lab.research.validation.costs import CostModel, trade_cost_fraction

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
    square_off: time | None = None,
) -> BacktestResult:
    """Simulate ``target_positions`` over ``candles`` with next-bar-open fills.

    Args:
        candles: Ascending decision bars.
        target_positions: Target position per bar (sign = side, 0 = flat),
            decided at each bar's close.
        cost_model: The Indian cost model applied to every round trip.
        notional_per_trade: Order value at which the cost fraction is evaluated.
        timezone: IST timezone for session/day boundaries.
        square_off: Intraday MIS square-off cutoff (IST wall-clock, e.g. 15:20, the
            configured ``calendar.session.square_off``). A position is force-closed
            at the last bar *before* the cutoff and no new position is opened at or
            after it — so a grid that runs past the cutoff (Kite 5-min bars reach
            15:25) never holds beyond the real MIS square-off. ``>=`` matches the
            calendar's :meth:`~lab.core.nse_calendar.NseCalendar.is_past_square_off`.
            When ``None`` the fallback is the day's last bar (legacy behaviour).

    Returns:
        The round-trip trades produced by the simulation.
    """
    if len(candles) != len(target_positions):
        raise ValueError("candles and target_positions must have equal length")
    tz = ZoneInfo(timezone)

    trades: list[Trade] = []
    side: Side | None = None
    entry_price = 0.0
    entry_volume = 0.0
    entry_time: datetime | None = None

    def close(exit_price: float, exit_time: datetime) -> None:
        nonlocal side, entry_time
        if side is None or entry_time is None:
            raise RuntimeError("close() called with no open position")
        direction = 1.0 if side is Side.LONG else -1.0
        gross = direction * (exit_price / entry_price - 1.0)
        cost_fraction = trade_cost_fraction(
            cost_model, notional_per_trade, entry_price, entry_volume
        )
        trades.append(
            Trade(side, entry_time, exit_time, entry_price, exit_price, gross, cost_fraction)
        )
        side = None
        entry_time = None

    def past_cutoff(index: int) -> bool:
        """Whether bar ``index`` is at/after the square-off cutoff (>= the cutoff)."""
        return (
            square_off is not None and candles[index].timestamp.astimezone(tz).time() >= square_off
        )

    n = len(candles)
    for t in range(n):
        candle = candles[t]
        day = candle.timestamp.astimezone(tz).date()
        same_day_as_prev = t >= 1 and candles[t - 1].timestamp.astimezone(tz).date() == day
        next_new_day = t == n - 1 or candles[t + 1].timestamp.astimezone(tz).date() != day
        # The square-off bar is the last tradeable bar of the day at/before the cutoff:
        # the day's last bar, or the last bar before the next one crosses the cutoff.
        is_square_off_bar = (not past_cutoff(t)) and (next_new_day or past_cutoff(t + 1))

        # Execute the previous (same-day) bar's decision at this open — but never at or
        # after the cutoff (no new MIS position is opened past square-off).
        if same_day_as_prev and not past_cutoff(t):
            desired = _target_side(target_positions[t - 1])
            if desired is not side:
                if side is not None:
                    close(candle.open, candle.timestamp)
                if desired is not None:
                    side = desired
                    entry_price = candle.open
                    entry_volume = float(candle.volume)
                    entry_time = candle.timestamp

        # Intraday square-off: force flat at the last bar before the cutoff (exit at its close).
        if is_square_off_bar and side is not None:
            close(candle.close, candle.timestamp)

    return BacktestResult(tuple(trades))
