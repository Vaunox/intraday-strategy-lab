"""Tests for the kill-gate, report, and paper updater (P2.6), incl. end-to-end."""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from lab.core.types import BarInterval, Candle, Side, Verdict
from lab.research.reports.killgate import (
    KillGateInputs,
    evaluate_kill_gate,
    load_kill_gate_thresholds,
)
from lab.research.reports.paper import append_study_section
from lab.research.reports.report import StudyReport, render_report, trade_statistics
from lab.research.strategies.adapter import signals_to_targets
from lab.research.strategies.reference import ReferenceMomentumSpec
from lab.research.trials.ledger import TrialLedger
from lab.research.validation.backtester import Trade, run_backtest
from lab.research.validation.costs import load_cost_model
from lab.research.validation.cpcv import combinatorial_purged_cv
from lab.research.validation.pbo import probability_of_backtest_overfitting
from lab.research.validation.robustness import (
    inject_ohlc_noise,
    monte_carlo_sign_flip,
    two_engines_agree,
    vectorized_backtest,
)
from lab.research.validation.sharpe import annualized_sharpe, return_stats

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config"
IST = ZoneInfo("Asia/Kolkata")
PERIODS = 18750.0

_PASSING = KillGateInputs(
    cpcv_median_path_sharpe=1.5,
    cpcv_positive_fraction=0.95,
    cpcv_tenth_percentile=0.2,
    dsr=0.99,
    pbo=0.10,
    profit_factor=1.5,
    top5_winners_fraction=0.30,
    expectancy=0.0020,
    round_trip_cost=0.0018,
    param_sensitivity_min_net_sharpe=0.6,
    mc_shuffle_beat_fraction=0.97,
    cross_symbol_positive_fraction=0.6,
    two_engines_reconcile=True,
    noise_survives=True,
    regime_bucket_medians=(0.5, 0.6, 0.7, 0.8),
    regime_positive_without_best=True,
)


def test_thresholds_load_from_config() -> None:
    thresholds = load_kill_gate_thresholds(REPO_CONFIG)
    assert thresholds.dsr_min == 0.95
    assert thresholds.pbo_max == 0.20
    assert thresholds.cpcv_median_path_sharpe_min == 1.0


def test_kill_gate_passes_when_all_criteria_met() -> None:
    result = evaluate_kill_gate(_PASSING, load_kill_gate_thresholds(REPO_CONFIG))
    assert result.verdict is Verdict.PASS
    assert result.passed
    assert all(c.passed for c in result.criteria)


def test_kill_gate_kills_on_any_single_failure() -> None:
    thresholds = load_kill_gate_thresholds(REPO_CONFIG)
    failed = replace(_PASSING, dsr=0.50)  # fails only criterion 2
    result = evaluate_kill_gate(failed, thresholds)
    assert result.verdict is Verdict.KILL
    assert not result.criteria[1].passed  # DSR criterion
    assert result.criteria[0].passed  # others still pass


def test_nan_metric_fails_its_criterion() -> None:
    thresholds = load_kill_gate_thresholds(REPO_CONFIG)
    result = evaluate_kill_gate(replace(_PASSING, dsr=float("nan")), thresholds)
    assert result.verdict is Verdict.KILL


def _trade(net: float) -> Trade:
    ts = datetime(2024, 7, 15, 9, 15, tzinfo=IST)
    return Trade(Side.LONG, ts, ts + timedelta(minutes=5), 100.0, 100.0 * (1 + net), net, 0.0)


def test_trade_statistics_hand_computed() -> None:
    stats = trade_statistics(
        [_trade(0.02), _trade(-0.01), _trade(0.03), _trade(-0.02), _trade(0.01)]
    )
    assert stats.profit_factor == pytest.approx(2.0)  # 0.06 / 0.03
    assert stats.expectancy == pytest.approx(0.006)  # 0.03 / 5
    assert stats.top5_winners_fraction == pytest.approx(1.0)  # only 3 winners
    assert stats.win_rate == pytest.approx(0.6)


def _synthetic(seed: int = 20) -> list[Candle]:
    rng = np.random.default_rng(seed)
    candles: list[Candle] = []
    price = 100.0
    for day in (date(2024, 7, 15), date(2024, 7, 16), date(2024, 7, 18), date(2024, 7, 19)):
        open_ts = datetime(day.year, day.month, day.day, 9, 15, tzinfo=IST)
        for bar in range(45):
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


def test_gate2_end_to_end_reference_spec_through_killgate(tmp_path: Path) -> None:
    """A spec runs through the unchanged harness; DSR auto-deflates from the
    ledger's effective-N; the verdict flows into the paper (Gate 2)."""
    costs = load_cost_model(REPO_CONFIG)
    thresholds = load_kill_gate_thresholds(REPO_CONFIG)
    spec = ReferenceMomentumSpec()
    candles = _synthetic()

    # Backtest -> returns -> CPCV (the validation engine is untouched by strategy code).
    targets = signals_to_targets(candles, spec.generate_signals(candles))
    result = run_backtest(candles, targets, costs)
    net = result.net_returns
    cpcv = combinatorial_purged_cv(net, n_groups=6, k_test_groups=2, periods_per_year=PERIODS)

    # Ledger: log variant runs; DSR pulls the EFFECTIVE trial count automatically.
    ledger = TrialLedger(tmp_path / "trials")
    variant_streams: list[np.ndarray] = []
    for i in range(4):
        stream = run_backtest(inject_ohlc_noise(candles, seed=i), targets, costs).net_returns
        variant_streams.append(stream)
        ledger.log_trial(spec.name, {"variant": i}, stream.tolist())
    stats_ = return_stats(net)
    dsr = ledger.deflated_sharpe(stats_.sharpe, stats_.n, stats_.skew, stats_.kurtosis)

    # PBO across the variant configs.
    length = min(s.size for s in variant_streams)
    matrix = np.column_stack([s[:length] for s in variant_streams])
    pbo = probability_of_backtest_overfitting(matrix, n_splits=8).pbo

    # Robustness pieces.
    vector = vectorized_backtest(candles, targets, costs)
    noise_run = run_backtest(inject_ohlc_noise(candles, seed=99), targets, costs)
    inputs = KillGateInputs(
        cpcv_median_path_sharpe=cpcv.median_path_sharpe,
        cpcv_positive_fraction=cpcv.positive_fraction,
        cpcv_tenth_percentile=cpcv.tenth_percentile,
        dsr=dsr,
        pbo=pbo,
        profit_factor=trade_statistics(result.trades).profit_factor,
        top5_winners_fraction=trade_statistics(result.trades).top5_winners_fraction,
        expectancy=trade_statistics(result.trades).expectancy,
        round_trip_cost=costs.round_trip_cost_fraction(100_000),
        param_sensitivity_min_net_sharpe=annualized_sharpe(net, PERIODS),
        mc_shuffle_beat_fraction=monte_carlo_sign_flip(net, n_shuffles=200),
        cross_symbol_positive_fraction=float(annualized_sharpe(noise_run.net_returns, PERIODS) > 0),
        two_engines_reconcile=two_engines_agree(result, vector),
        noise_survives=bool(np.sign(noise_run.net_returns.sum()) == np.sign(net.sum())),
        regime_bucket_medians=(cpcv.median_path_sharpe,),
        regime_positive_without_best=cpcv.median_path_sharpe > 0,
    )
    kg = evaluate_kill_gate(inputs, thresholds)
    report = StudyReport(
        strategy=spec.name,
        cpcv=cpcv,
        dsr=dsr,
        pbo=pbo,
        effective_trials=ledger.effective_trials(),
        trades=trade_statistics(result.trades),
        kill_gate=kg,
    )

    # The two engines reconcile, and the honest verdict for a no-edge rule is KILL.
    assert inputs.two_engines_reconcile
    assert kg.verdict is Verdict.KILL

    # Results flow into the research paper.
    findings = tmp_path / "RESEARCH_FINDINGS.md"
    findings.write_text("# Findings\n", encoding="utf-8")
    append_study_section(report, findings_path=findings)
    text = findings.read_text(encoding="utf-8")
    assert spec.name in text
    assert "KILL" in text
    assert "Seven-point kill-gate" in render_report(report)
