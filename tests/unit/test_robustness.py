"""Tests for the robustness battery and two-engine reconciliation (P2.5)."""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from lab.core.types import BarInterval, Candle
from lab.research.strategies.adapter import signals_to_targets
from lab.research.strategies.reference import ReferenceMomentumSpec
from lab.research.validation.backtester import run_backtest
from lab.research.validation.costs import load_cost_model
from lab.research.validation.robustness import (
    fraction_positive,
    inject_ohlc_noise,
    monte_carlo_sign_flip,
    two_engines_agree,
    vectorized_backtest,
)

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config"
IST = ZoneInfo("Asia/Kolkata")


def _synthetic() -> list[Candle]:
    rng = np.random.default_rng(13)
    candles: list[Candle] = []
    price = 100.0
    for day in (date(2024, 7, 15), date(2024, 7, 16), date(2024, 7, 18)):
        open_ts = datetime(day.year, day.month, day.day, 9, 15, tzinfo=IST)
        for bar in range(30):
            prev = price
            price = prev * math.exp(float(rng.normal(0.0, 0.003)))
            candles.append(
                Candle(
                    "SYN",
                    BarInterval.MIN_5,
                    open_ts + timedelta(minutes=5 * bar),
                    prev,
                    max(prev, price) + 0.3,
                    min(prev, price) - 0.3,
                    price,
                    1000,
                )
            )
    return candles


def test_monte_carlo_sign_flip_discriminates_edge() -> None:
    rng = np.random.default_rng(0)
    edge = rng.normal(0.01, 0.005, 200)  # clear positive edge (per-period Sharpe ~2)
    noise = rng.normal(0.0, 0.01, 200)  # no directional edge
    # A real edge clears the kill-gate bar; a no-edge series does not.
    assert monte_carlo_sign_flip(edge, n_shuffles=500) > 0.95
    assert monte_carlo_sign_flip(noise, n_shuffles=500) < 0.95


def test_inject_noise_keeps_valid_and_close() -> None:
    candles = _synthetic()
    perturbed = inject_ohlc_noise(candles, relative_scale=0.001, seed=1)
    assert len(perturbed) == len(candles)
    # Levels are jittered but close, and OHLC validity held (construction didn't raise).
    assert perturbed[0].close != candles[0].close
    assert abs(perturbed[0].close / candles[0].close - 1) < 0.01


def test_fraction_positive() -> None:
    assert fraction_positive([1.0, -1.0, 2.0, -3.0]) == 0.5
    assert fraction_positive([0.1, 0.2, float("nan")]) == 1.0


def test_two_engines_reconcile_on_reference_spec() -> None:
    costs = load_cost_model(REPO_CONFIG)
    candles = _synthetic()
    targets = signals_to_targets(candles, ReferenceMomentumSpec().generate_signals(candles))

    event = run_backtest(candles, targets, costs)
    vector = vectorized_backtest(candles, targets, costs)

    assert len(event.trades) == len(vector.trades) > 0
    assert two_engines_agree(event, vector)
    assert np.allclose(event.net_returns, vector.net_returns)


def test_two_engines_disagree_on_different_targets() -> None:
    costs = load_cost_model(REPO_CONFIG)
    candles = _synthetic()
    targets = signals_to_targets(candles, ReferenceMomentumSpec().generate_signals(candles))
    event = run_backtest(candles, targets, costs)
    flat = vectorized_backtest(candles, [0.0] * len(candles), costs)  # no trades
    assert not two_engines_agree(event, flat)
