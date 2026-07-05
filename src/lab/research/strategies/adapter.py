"""StrategySpec adapter (Phase 2, P2.4).

Bridges a rule-based ``StrategySpec`` (event → entry → exit/holding → position)
to the validation engine: it turns the spec's point-in-time signals into the
per-bar target-position series the event-driven backtester consumes, then runs
the backtest. The validation engine (backtester, CPCV, metrics) is generic — it
never imports strategy code; strategies flow in only as candles → signals →
positions → returns.

Positions are held within a day and reset flat at each day's open (no overnight
carry); a signal's signed strength (LONG +, SHORT -, strength 0 = flat) sets the
target, forward-filled until the next signal or the day boundary.
"""

from __future__ import annotations

from collections.abc import Sequence
from zoneinfo import ZoneInfo

from lab.core.constants import INDIA_TZ
from lab.core.interfaces import StrategySpec
from lab.core.types import Candle, Side, StrategySignal
from lab.research.validation.backtester import BacktestResult, run_backtest
from lab.research.validation.costs import CostModel


def signals_to_targets(
    candles: Sequence[Candle], signals: Sequence[StrategySignal], *, timezone: str = INDIA_TZ
) -> list[float]:
    """Convert point-in-time signals into a per-bar target-position series.

    Held within a day, reset flat at each day's first bar; a signal at a bar's
    timestamp sets the signed target (side x strength), forward-filled until the
    next signal.
    """
    tz = ZoneInfo(timezone)
    by_timestamp = {s.asof: (1.0 if s.side is Side.LONG else -1.0) * s.strength for s in signals}
    targets = [0.0] * len(candles)
    current = 0.0
    current_day = None
    for i, candle in enumerate(candles):
        day = candle.timestamp.astimezone(tz).date()
        if day != current_day:
            current = 0.0
            current_day = day
        if candle.timestamp in by_timestamp:
            current = by_timestamp[candle.timestamp]
        targets[i] = current
    return targets


def run_strategy(
    spec: StrategySpec,
    candles: Sequence[Candle],
    cost_model: CostModel,
    *,
    notional_per_trade: float = 100_000.0,
    timezone: str = INDIA_TZ,
) -> BacktestResult:
    """Run ``spec`` over ``candles`` through the cost-inclusive backtester."""
    signals = spec.generate_signals(candles)
    targets = signals_to_targets(candles, signals, timezone=timezone)
    return run_backtest(
        candles, targets, cost_model, notional_per_trade=notional_per_trade, timezone=timezone
    )
