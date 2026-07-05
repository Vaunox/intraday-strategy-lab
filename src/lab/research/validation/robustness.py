"""Robustness battery + two-engine reconciliation (Phase 2, P2.5).

Utilities the kill-gate's robustness criterion draws on (Part III Layer 2):

* **Monte-Carlo sign-flip** — the real net Sharpe must beat a null that randomly
  flips each trade's sign (no directional edge).
* **Noise injection** — perturb OHLC levels and re-run; the edge must survive.
* **Fraction-positive** — for cross-symbol (majority of held-out symbols net-
  positive) and parameter-sensitivity aggregation.
* **Two-engine reconciliation** — an independent *vectorized* engine (run-
  detection on the held-position series) must reproduce the event-driven engine's
  trades, catching implementation bugs.
"""

from __future__ import annotations

from collections.abc import Sequence
from zoneinfo import ZoneInfo

import numpy as np
import numpy.typing as npt

from lab.core.constants import INDIA_TZ
from lab.core.types import Candle, Side
from lab.research.validation.backtester import BacktestResult, Trade
from lab.research.validation.costs import CostModel, trade_cost_fraction
from lab.research.validation.sharpe import per_period_sharpe

FloatArray = npt.NDArray[np.float64]


def monte_carlo_sign_flip(
    returns: npt.ArrayLike, *, n_shuffles: int = 1000, seed: int = 0
) -> float:
    """Fraction of sign-flipped nulls the real per-period Sharpe beats.

    Each shuffle multiplies every return by a random +/-1 (a no-directional-edge
    null); a genuine edge beats most of them. Kill-gate bar: >= 0.95.
    """
    values = np.asarray(returns, dtype=np.float64)
    if values.size < 2:
        return float("nan")
    real = per_period_sharpe(values)
    if not np.isfinite(real):
        return float("nan")
    rng = np.random.default_rng(seed)
    beaten = 0
    for _ in range(n_shuffles):
        shuffled = per_period_sharpe(values * rng.choice([-1.0, 1.0], size=values.size))
        if np.isfinite(shuffled) and real > shuffled:
            beaten += 1
    return beaten / n_shuffles


def inject_ohlc_noise(
    candles: Sequence[Candle], *, relative_scale: float = 0.0005, seed: int = 0
) -> list[Candle]:
    """Return ``candles`` with each bar's price level multiplicatively perturbed.

    A per-bar factor scales O/H/L/C together, so the OHLC relations stay valid
    while the price level is jittered (realistic tick-level noise).
    """
    rng = np.random.default_rng(seed)
    perturbed: list[Candle] = []
    for candle in candles:
        factor = 1.0 + float(rng.normal(0.0, relative_scale))
        perturbed.append(
            Candle(
                symbol=candle.symbol,
                interval=candle.interval,
                timestamp=candle.timestamp,
                open=candle.open * factor,
                high=candle.high * factor,
                low=candle.low * factor,
                close=candle.close * factor,
                volume=candle.volume,
                open_interest=candle.open_interest,
            )
        )
    return perturbed


def fraction_positive(values: npt.ArrayLike) -> float:
    """Fraction of finite ``values`` that are strictly positive."""
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    return float(np.mean(finite > 0.0)) if finite.size else float("nan")


def _side(target: float) -> Side | None:
    if target > 0:
        return Side.LONG
    if target < 0:
        return Side.SHORT
    return None


def vectorized_backtest(
    candles: Sequence[Candle],
    target_positions: Sequence[float],
    cost_model: CostModel,
    *,
    notional_per_trade: float = 100_000.0,
    timezone: str = INDIA_TZ,
) -> BacktestResult:
    """A second, vectorized backtest engine via run-detection on held positions.

    The position held during bar ``k`` is ``side(target[k-1])`` (each bar executes
    the prior bar's decision at this bar's open). Maximal runs of a constant
    nonzero side are trades: entry at the run's first open, exit at the next
    change's open, or the day's last close on square-off. Must reconcile with the
    event-driven engine.
    """
    if len(candles) != len(target_positions):
        raise ValueError("candles and target_positions must have equal length")
    tz = ZoneInfo(timezone)
    days = [c.timestamp.astimezone(tz).date() for c in candles]

    trades: list[Trade] = []
    start = 0
    n = len(candles)
    while start < n:
        end = start
        while end < n and days[end] == days[start]:
            end += 1
        held: list[Side | None] = [None] * (end - start)
        for k in range(1, end - start):
            held[k] = _side(target_positions[start + k - 1])

        k = 0
        while k < end - start:
            side = held[k]
            if side is None:
                k += 1
                continue
            run_start = k
            while k + 1 < end - start and held[k + 1] is side:
                k += 1
            entry = candles[start + run_start]
            if k == end - start - 1:  # run reaches the day's last bar -> square-off
                exit_candle, exit_price = candles[start + k], candles[start + k].close
            else:
                exit_candle, exit_price = candles[start + k + 1], candles[start + k + 1].open
            direction = 1.0 if side is Side.LONG else -1.0
            gross = direction * (exit_price / entry.open - 1.0)
            cost_fraction = trade_cost_fraction(
                cost_model, notional_per_trade, entry.open, float(entry.volume)
            )
            trades.append(
                Trade(
                    side,
                    entry.timestamp,
                    exit_candle.timestamp,
                    entry.open,
                    exit_price,
                    gross,
                    cost_fraction,
                )
            )
            k += 1
        start = end
    return BacktestResult(tuple(trades))


def two_engines_agree(
    event_driven: BacktestResult, vectorized: BacktestResult, *, tolerance: float = 1e-9
) -> bool:
    """Return whether the two engines' per-trade net returns match within tolerance."""
    a = event_driven.net_returns
    b = vectorized.net_returns
    if a.shape != b.shape:
        return False
    return bool(np.allclose(a, b, atol=tolerance))
