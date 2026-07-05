"""Tests for the study orchestrator, robustness battery, and walk-forward equity (P2 gaps)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from lab.core.types import BarInterval, Candle, Side, StrategySignal, Verdict
from lab.research.reports.killgate import load_kill_gate_thresholds
from lab.research.reports.report import StudyReport, equity_curve, render_report
from lab.research.study import (
    enumerate_param_configs,
    regime_bucket_stats,
    run_param_configs,
    run_robustness_battery,
    run_study,
)
from lab.research.trials.ledger import TrialLedger
from lab.research.validation.backtester import Trade
from lab.research.validation.costs import load_cost_model

IST = ZoneInfo("Asia/Kolkata")
REPO_CONFIG = Path(__file__).resolve().parents[2] / "config"
PERIODS = 18750.0
NO_EMBARGO = timedelta(0)


@dataclass(frozen=True, slots=True)
class ThresholdMomentumSpec:
    """Parametrized reference: trade the bar's direction only when |move| > threshold."""

    threshold: float
    name: str = "threshold_momentum"
    interval: BarInterval = BarInterval.MIN_5

    def generate_signals(self, candles: Sequence[Candle]) -> list[StrategySignal]:
        signals: list[StrategySignal] = []
        for candle in candles:
            move = candle.close / candle.open - 1.0
            side = Side.LONG if move >= 0 else Side.SHORT
            strength = 1.0 if abs(move) > self.threshold else 0.0
            signals.append(
                StrategySignal(
                    asof=candle.timestamp, symbol=candle.symbol, side=side, strength=strength
                )
            )
        return signals


def _make_spec(params: Mapping[str, float]) -> ThresholdMomentumSpec:
    return ThresholdMomentumSpec(threshold=float(params["threshold"]))


def _series(symbol: str, seed: int, *, days: int = 3, bars_per_day: int = 75) -> list[Candle]:
    """A deterministic full-session multi-day 5-min series (spans morning->afternoon)."""
    rng = np.random.default_rng(seed)
    candles: list[Candle] = []
    price = 100.0
    for day in range(days):
        open_ts = datetime(2024, 7, 15 + day, 9, 15, tzinfo=IST)
        for bar in range(bars_per_day):
            prev = price
            price = prev * math.exp(float(rng.normal(0.0, 0.003)))
            high = max(prev, price) + 0.3
            low = min(prev, price) - 0.3
            candles.append(
                Candle(
                    symbol,
                    BarInterval.MIN_5,
                    open_ts + timedelta(minutes=5 * bar),
                    prev,
                    high,
                    low,
                    price,
                    1000 + int(rng.integers(0, 500)),
                )
            )
    return candles


def _trade(minute_hour: int, net: float) -> Trade:
    ts = datetime(2024, 7, 15, minute_hour, 30, tzinfo=IST)
    return Trade(Side.LONG, ts, ts + timedelta(minutes=5), 100.0, 100.0 * (1 + net), net, 0.0)


# --- walk-forward equity ---------------------------------------------------- #
def test_equity_curve_cumulates_and_measures_drawdown() -> None:
    trades = [_trade(10, 0.1), _trade(11, -0.2), _trade(12, 0.3)]
    eq = equity_curve(trades)
    assert eq.equity == pytest.approx((0.1, -0.1, 0.2))
    assert eq.total_return == pytest.approx(0.2)
    assert eq.max_drawdown == pytest.approx(0.2)  # peak 0.1 -> trough -0.1
    assert eq.n_trades == 3


def test_equity_curve_handles_no_trades() -> None:
    eq = equity_curve([])
    assert eq.equity == () and eq.total_return == 0.0 and eq.max_drawdown == 0.0


# --- parameter configs ------------------------------------------------------ #
def test_enumerate_param_configs_is_base_plus_two_per_param() -> None:
    configs = enumerate_param_configs({"threshold": 0.0, "lookback": 10}, {"threshold": 0.01})
    labels = [label for label, _ in configs]
    assert labels == ["base", "threshold+", "threshold-"]
    assert dict(configs)["threshold+"]["threshold"] == pytest.approx(0.01)
    assert dict(configs)["threshold-"]["threshold"] == pytest.approx(-0.01)


def test_run_param_configs_varies_with_threshold() -> None:
    candles = _series("SYN", 1)
    costs = load_cost_model(REPO_CONFIG)
    streams = run_param_configs(
        _make_spec, {"threshold": 0.0}, {"threshold": 0.002}, candles, costs
    )
    assert set(streams) == {"base", "threshold+", "threshold-"}
    # A higher threshold gates out weak bars -> a different (usually shorter) stream.
    assert streams["threshold+"].size != streams["base"].size


# --- robustness battery ----------------------------------------------------- #
def test_robustness_battery_populates_criterion6_inputs() -> None:
    candles = _series("SYN", 2)
    costs = load_cost_model(REPO_CONFIG)
    report = run_robustness_battery(
        _make_spec,
        {"threshold": 0.0},
        {"threshold": 0.002},
        candles,
        costs,
        periods_per_year=PERIODS,
        cross_symbol_candles={"S2": _series("S2", 3), "S3": _series("S3", 4)},
        noise_seeds=4,
        mc_shuffles=100,
    )
    assert report.two_engines_reconcile  # both engines must agree exactly
    assert len(report.param_net_sharpes) == 3  # base + 2 perturbations
    assert len(report.noise_net_sharpes) == 4
    assert 0.0 <= report.cross_symbol_positive_fraction <= 1.0
    assert 0.0 <= report.mc_shuffle_beat_fraction <= 1.0
    # Parameter sensitivity is the WORST finite net Sharpe across the configs.
    finite = [s for s in report.param_net_sharpes if math.isfinite(s)]
    assert report.param_sensitivity_min_net_sharpe == pytest.approx(min(finite))
    assert isinstance(report.noise_survives, bool)


def test_robustness_battery_reuses_supplied_config_streams() -> None:
    candles = _series("SYN", 5)
    costs = load_cost_model(REPO_CONFIG)
    streams = run_param_configs(_make_spec, {"threshold": 0.0}, {}, candles, costs)
    report = run_robustness_battery(
        _make_spec,
        {"threshold": 0.0},
        {},
        candles,
        costs,
        periods_per_year=PERIODS,
        config_streams=streams,
        noise_seeds=2,
        mc_shuffles=50,
    )
    assert len(report.param_net_sharpes) == 1  # only the base config


# --- regime buckets --------------------------------------------------------- #
def test_regime_bucket_stats_splits_by_session_third() -> None:
    trades = [_trade(9, 0.01), _trade(12, 0.02), _trade(14, -0.01), _trade(14, 0.03)]
    medians, positive_without_best = regime_bucket_stats(trades, PERIODS)
    assert len(medians) == 3  # morning, midday, afternoon all occupied
    assert isinstance(positive_without_best, bool)


# --- full orchestrator ------------------------------------------------------ #
def test_run_study_produces_full_report_and_kills_no_edge(tmp_path: Path) -> None:
    candles = _series("SYN", 10)
    costs = load_cost_model(REPO_CONFIG)
    thresholds = load_kill_gate_thresholds(REPO_CONFIG)
    ledger = TrialLedger(tmp_path / "trials")

    report = run_study(
        ThresholdMomentumSpec(0.0),
        candles,
        costs,
        thresholds,
        ledger,
        periods_per_year=PERIODS,
        spec_factory=_make_spec,
        base_params={"threshold": 0.0},
        param_steps={"threshold": 0.002},
        cross_symbol_candles={"S2": _series("S2", 11)},
        cpcv_embargo=NO_EMBARGO,  # compressed synthetic span
        pbo_splits=8,
    )

    assert isinstance(report, StudyReport)
    assert report.equity is not None and report.equity.n_trades > 0
    assert math.isfinite(report.dsr)
    assert 0.0 <= report.pbo <= 1.0  # base + 2 configs -> a real PBO matrix
    assert report.effective_trials >= 1.0
    assert ledger.count() == 3  # base + 2 param configs logged as trials
    # A directionless rule on random data is not a real edge -> honest verdict is KILL.
    assert report.kill_gate.verdict is Verdict.KILL
    # The report renders the walk-forward equity line.
    assert "Walk-forward equity" in render_report(report)


def test_run_study_without_params_has_nan_pbo(tmp_path: Path) -> None:
    # A parameter-free spec: no cross-config matrix -> PBO is NaN (cannot be measured).
    candles = _series("SYN", 12)
    costs = load_cost_model(REPO_CONFIG)
    thresholds = load_kill_gate_thresholds(REPO_CONFIG)
    ledger = TrialLedger(tmp_path / "trials")
    report = run_study(
        ThresholdMomentumSpec(0.0),
        candles,
        costs,
        thresholds,
        ledger,
        periods_per_year=PERIODS,
        cpcv_embargo=NO_EMBARGO,
    )
    assert math.isnan(report.pbo)
    assert ledger.count() == 1  # only the base config
    assert report.kill_gate.verdict is Verdict.KILL  # NaN PBO fails criterion 3
