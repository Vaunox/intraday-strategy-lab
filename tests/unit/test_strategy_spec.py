"""Tests for the StrategySpec adapter and reference spec, end-to-end (P2.4)."""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from lab.core.interfaces import StrategySpec
from lab.core.types import BarInterval, Candle, Side, StrategySignal
from lab.research.strategies.adapter import run_strategy, signals_to_targets
from lab.research.strategies.reference import ReferenceMomentumSpec
from lab.research.validation.costs import load_cost_model
from lab.research.validation.cpcv import combinatorial_purged_cv

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config"
IST = ZoneInfo("Asia/Kolkata")


def _synthetic(bars_per_day: int = 40) -> list[Candle]:
    rng = np.random.default_rng(11)
    candles: list[Candle] = []
    price = 100.0
    for day in (date(2024, 7, 15), date(2024, 7, 16), date(2024, 7, 18), date(2024, 7, 19)):
        open_ts = datetime(day.year, day.month, day.day, 9, 15, tzinfo=IST)
        for bar in range(bars_per_day):
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


def test_reference_spec_satisfies_protocol() -> None:
    spec: StrategySpec = ReferenceMomentumSpec()
    assert isinstance(spec, StrategySpec)
    assert spec.name == "reference_momentum"
    assert spec.interval is BarInterval.MIN_5


def test_signals_to_targets_holds_within_day_and_resets() -> None:
    d1 = datetime(2024, 7, 15, 9, 15, tzinfo=IST)
    candles = [
        Candle("X", BarInterval.MIN_5, d1, 100, 101, 99, 100, 1000),
        Candle("X", BarInterval.MIN_5, d1 + timedelta(minutes=5), 100, 101, 99, 100, 1000),
        Candle(
            "X",
            BarInterval.MIN_5,
            datetime(2024, 7, 16, 9, 15, tzinfo=IST),
            100,
            101,
            99,
            100,
            1000,
        ),
    ]
    signals = [StrategySignal(asof=d1, symbol="X", side=Side.LONG, strength=1.0)]
    targets = signals_to_targets(candles, signals)
    assert targets == [1.0, 1.0, 0.0]  # held within day 1, reset flat on day 2


def test_run_strategy_produces_trades() -> None:
    costs = load_cost_model(REPO_CONFIG)
    result = run_strategy(ReferenceMomentumSpec(), _synthetic(), costs)
    assert len(result.trades) > 0
    assert all(t.side in (Side.LONG, Side.SHORT) for t in result.trades)


def test_reference_spec_runs_end_to_end_through_cpcv() -> None:
    # The P2.4 acceptance: a trivial spec runs through the UNCHANGED validation
    # engine (backtester -> returns -> CPCV) with no strategy code in the engine.
    costs = load_cost_model(REPO_CONFIG)
    result = run_strategy(ReferenceMomentumSpec(), _synthetic(), costs)
    cpcv = combinatorial_purged_cv(
        result.net_returns,
        result.entry_times,
        result.exit_times,
        n_groups=5,
        k_test_groups=2,
        periods_per_year=18750.0,
        embargo=timedelta(0),  # compressed synthetic span; embargo exercised in test_cpcv_pbo
    )
    assert len(cpcv.path_sharpes) == 10  # C(5,2)
    assert math.isfinite(cpcv.median_path_sharpe)
