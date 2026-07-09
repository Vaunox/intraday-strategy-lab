"""Tests for the study orchestrator, robustness battery, and walk-forward equity (P2 gaps).

Also covers the Phase-3 hardening: keyed criterion-6 evidence (B-1), the year x
vol/trend regime labeler (B-4), and that a NaN day-aligned PBO fails closed (B-2).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from lab.core.types import BarInterval, Candle, Side, StrategySignal, Verdict
from lab.research.reports.killgate import KillGateInputs, load_kill_gate_thresholds
from lab.research.reports.report import StudyReport, equity_curve, render_report
from lab.research.study import (
    build_regime_labeler,
    enumerate_param_configs,
    gate_pass_margin,
    regime_bucket_stats,
    run_param_configs,
    run_robustness_battery,
    run_study,
    survivorship_stamp,
)
from lab.research.trials.ledger import TrialLedger
from lab.research.validation.backtester import Trade
from lab.research.validation.costs import load_cost_model

IST = ZoneInfo("Asia/Kolkata")
REPO_CONFIG = Path(__file__).resolve().parents[2] / "config"
PERIODS = 18750.0
NO_EMBARGO = timedelta(0)
THRESH = load_kill_gate_thresholds(REPO_CONFIG)


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


def _by_day(trade: Trade) -> str:
    """A deterministic >=1-bucket-per-day regime labeler for tests (>= 4 days -> >= 4 buckets)."""
    return str(trade.entry_time.astimezone(IST).date())


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
    results = run_param_configs(
        _make_spec, {"threshold": 0.0}, {"threshold": 0.002}, candles, costs
    )
    assert set(results) == {"base", "threshold+", "threshold-"}
    # A higher threshold gates out weak bars -> a different (usually shorter) stream.
    assert results["threshold+"].net_returns.size != results["base"].net_returns.size


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
    assert set(report.param_config_sharpes) == {"base", "threshold+", "threshold-"}  # keyed
    assert set(report.cross_symbol_sharpes) == {"S2", "S3"}  # keyed by held-out symbol
    assert len(report.noise_net_sharpes) == 4
    assert 0.0 <= report.cross_symbol_positive_fraction <= 1.0
    assert 0.0 <= report.mc_shuffle_beat_fraction <= 1.0
    # Parameter sensitivity is the WORST finite net Sharpe across the configs.
    finite = [s for s in report.param_config_sharpes.values() if math.isfinite(s)]
    assert report.param_sensitivity_min_net_sharpe == pytest.approx(min(finite))
    assert isinstance(report.noise_survives, bool)


def test_robustness_battery_reuses_supplied_config_results() -> None:
    candles = _series("SYN", 5)
    costs = load_cost_model(REPO_CONFIG)
    results = run_param_configs(_make_spec, {"threshold": 0.0}, {}, candles, costs)
    report = run_robustness_battery(
        _make_spec,
        {"threshold": 0.0},
        {},
        candles,
        costs,
        periods_per_year=PERIODS,
        config_results=results,
        noise_seeds=2,
        mc_shuffles=50,
    )
    assert set(report.param_config_sharpes) == {"base"}  # only the base config


# --- regime buckets (B-4) --------------------------------------------------- #
def test_regime_bucket_stats_keys_by_bucket_label() -> None:
    trades = [_trade(9, 0.01), _trade(12, 0.02), _trade(14, -0.01), _trade(14, 0.03)]
    sharpes, positive_without_best = regime_bucket_stats(
        trades, PERIODS, labeler=lambda t: "am" if t.entry_time.hour < 12 else "pm"
    )
    assert set(sharpes) == {"am", "pm"}
    assert isinstance(positive_without_best, bool)


def test_regime_bucket_stats_default_labels_by_year() -> None:
    sharpes, _ = regime_bucket_stats([_trade(9, 0.01), _trade(12, 0.02)], PERIODS)
    assert set(sharpes) == {"2024"}


def test_build_regime_labeler_tags_year_vol_trend() -> None:
    candles = _series("SYN", 1)
    labeler = build_regime_labeler(candles)
    # A trade entering at a known bar inherits that bar's "year|vol|trend" label.
    trade = Trade(Side.LONG, candles[40].timestamp, candles[41].timestamp, 100.0, 101.0, 0.01, 0.0)
    year, vol, trend = labeler(trade).split("|")
    assert year == "2024"
    assert vol in {"hivol", "lovol"}
    assert trend in {"up", "down"}


# --- full orchestrator ------------------------------------------------------ #
def test_run_study_produces_full_report_and_kills_no_edge(tmp_path: Path) -> None:
    candles = _series("SYN", 10, days=8)
    costs = load_cost_model(REPO_CONFIG)
    ledger = TrialLedger(tmp_path / "trials")

    report = run_study(
        ThresholdMomentumSpec(0.0),
        candles,
        costs,
        THRESH,
        ledger,
        spec_factory=_make_spec,
        base_params={"threshold": 0.0},
        param_steps={"threshold": 0.002},
        cross_symbol_candles={
            "S2": _series("S2", 11, days=8),
            "S3": _series("S3", 12, days=8),
            "S4": _series("S4", 13, days=8),
        },
        cpcv_embargo=NO_EMBARGO,  # compressed synthetic span
        regime_labeler=_by_day,  # 6 day-buckets -> clears the regime evidence floor
        pbo_splits=4,
    )

    assert isinstance(report, StudyReport)
    assert report.equity is not None and report.equity.n_trades > 0
    assert math.isfinite(report.dsr)
    assert math.isnan(report.pbo) or 0.0 <= report.pbo <= 1.0
    assert report.effective_trials >= 1.0
    assert ledger.count() == 3  # base + 2 param configs logged as trials
    # A directionless rule on random data is not a real edge -> honest verdict is KILL
    # (evidence is well-shaped, so the gate GRADES rather than returning INSUFFICIENT).
    assert report.kill_gate.verdict is Verdict.KILL
    assert "Walk-forward equity" in render_report(report)


def test_nan_pbo_fails_closed_not_pass_by_absence(tmp_path: Path) -> None:
    # A parameter-free spec has no cross-config matrix -> PBO is NaN. With otherwise
    # well-shaped evidence the gate still GRADES, and a NaN PBO must FAIL criterion 3
    # (KILL), never read as a pass-by-absence (B-2).
    candles = _series("SYN", 12, days=6)
    costs = load_cost_model(REPO_CONFIG)
    ledger = TrialLedger(tmp_path / "trials")
    report = run_study(
        ThresholdMomentumSpec(0.0),
        candles,
        costs,
        THRESH,
        ledger,
        cross_symbol_candles={
            "S2": _series("S2", 13, days=6),
            "S3": _series("S3", 14, days=6),
            "S4": _series("S4", 15, days=6),
        },
        cpcv_embargo=NO_EMBARGO,
        regime_labeler=_by_day,
        pbo_splits=4,
    )
    assert math.isnan(report.pbo)
    assert ledger.count() == 1  # only the base config
    assert report.kill_gate.verdict is Verdict.KILL  # NaN PBO fails criterion 3, not skipped


def test_run_study_insufficient_when_no_cross_symbols(tmp_path: Path) -> None:
    # Missing the cross-symbol holdout is an un-computed input, not a KILL — the gate
    # refuses to certify (INSUFFICIENT) rather than grading a stub (B-1).
    candles = _series("SYN", 16, days=6)
    report = run_study(
        ThresholdMomentumSpec(0.0),
        candles,
        load_cost_model(REPO_CONFIG),
        THRESH,
        TrialLedger(tmp_path / "trials"),
        cpcv_embargo=NO_EMBARGO,
        regime_labeler=_by_day,
    )
    assert report.kill_gate.verdict is Verdict.INSUFFICIENT
    assert not report.kill_gate.passed


def test_run_study_insufficient_when_base_too_thin(tmp_path: Path) -> None:
    # The realized-frequency annualization is data-dependent (len(trades)/span), so a base
    # with too few trades yields an unstable factor and a CPCV distribution dominated by a
    # handful of (possibly lucky) trades. The gate must refuse to certify (INSUFFICIENT),
    # never grade a thin base or crash inside CPCV -- fail-closed, like the structural floors.
    candles = _series("SYN", 30, days=1, bars_per_day=12)  # ~5 trades, below the min floor
    ledger = TrialLedger(tmp_path / "trials")
    report = run_study(
        ThresholdMomentumSpec(0.0),
        candles,
        load_cost_model(REPO_CONFIG),
        THRESH,
        ledger,
        cpcv_embargo=NO_EMBARGO,
        regime_labeler=_by_day,
    )
    assert report.kill_gate.verdict is Verdict.INSUFFICIENT
    assert not report.kill_gate.passed
    assert report.trades.n_trades < THRESH.min_base_observations
    assert ledger.count() == 0  # a non-starter logs no trial to the program effective-N


def test_run_study_default_regime_labeler_runs_end_to_end(tmp_path: Path) -> None:
    # With NO explicit labeler, run_study uses the year x vol/trend partition (B-4);
    # the default path must run and produce a recorded verdict (KILL or INSUFFICIENT,
    # never PASS for a no-edge rule).
    candles = _series("SYN", 17, days=8)
    report = run_study(
        ThresholdMomentumSpec(0.0),
        candles,
        load_cost_model(REPO_CONFIG),
        THRESH,
        TrialLedger(tmp_path / "trials"),
        cross_symbol_candles={
            "S2": _series("S2", 18, days=8),
            "S3": _series("S3", 19, days=8),
            "S4": _series("S4", 20, days=8),
        },
        cpcv_embargo=NO_EMBARGO,
        pbo_splits=4,
    )
    assert report.kill_gate.verdict in {Verdict.KILL, Verdict.INSUFFICIENT}


# --- provisional / upper-bound stamp (survivorship handling) ---------------- #
def _kg_inputs(**over: object) -> KillGateInputs:
    """A fully-passing KillGateInputs with well-shaped evidence; override to make one thin."""
    base: dict[str, object] = {
        "cpcv_path_sharpes": (1.5, 1.7, 1.8, 1.9, 2.0, 2.0, 2.1, 2.2, 2.3, 2.5),
        "dsr": 0.99,
        "pbo": 0.05,
        "profit_factor": 2.0,
        "top5_winners_fraction": 0.2,
        "expectancy": 0.01,
        "round_trip_cost": 0.002,
        "param_config_sharpes": {"base": 1.5, "threshold+": 1.4, "threshold-": 1.6},
        "has_tunable_params": True,
        "cross_symbol_sharpes": {"AAA": 1.0, "BBB": 1.1, "CCC": 0.9},
        "primary_symbol": "XYZ",
        "mc_shuffle_beat_fraction": 0.99,
        "two_engines_reconcile": True,
        "noise_survives": True,
        "regime_bucket_sharpes": {"y1": 1.0, "y2": 1.0, "y3": 1.0, "y4": 1.0},
        "regime_positive_without_best": True,
    }
    base.update(over)
    return KillGateInputs(**base)  # type: ignore[arg-type]


def test_gate_pass_margin_picks_tightest_return_criterion() -> None:
    margin, criterion = gate_pass_margin(_kg_inputs(cpcv_path_sharpes=(1.05,) * 10), THRESH)
    assert criterion == "cpcv_median"
    assert margin == pytest.approx(0.05)  # (1.05 - 1.0) / 1.0


def test_survivorship_stamp_flags_narrow_pass() -> None:
    provisional, note = survivorship_stamp(
        Verdict.PASS, _kg_inputs(cpcv_path_sharpes=(1.05,) * 10), THRESH
    )
    assert provisional
    assert "upper bound" in note and "cpcv_median" in note


def test_survivorship_stamp_ignores_wide_pass() -> None:
    provisional, note = survivorship_stamp(Verdict.PASS, _kg_inputs(), THRESH)
    assert not provisional and note == ""


def test_survivorship_stamp_never_stamps_kill() -> None:
    provisional, _ = survivorship_stamp(
        Verdict.KILL, _kg_inputs(cpcv_path_sharpes=(1.05,) * 10), THRESH
    )
    assert not provisional


def test_render_report_shows_provisional_stamp_only_when_set(tmp_path: Path) -> None:
    candles = _series("SYN", 20, days=6)
    report = run_study(
        ThresholdMomentumSpec(0.0),
        candles,
        load_cost_model(REPO_CONFIG),
        THRESH,
        TrialLedger(tmp_path / "trials"),
        cpcv_embargo=NO_EMBARGO,
        regime_labeler=_by_day,
    )
    assert not report.provisional  # a non-PASS verdict is never stamped
    assert "PROVISIONAL" not in render_report(report)
    stamped = replace(report, provisional=True, provisional_note="narrow pass; upper bound")
    assert "PROVISIONAL" in render_report(stamped)
