"""Tests for the kill-gate, report, and paper updater (P2.6), incl. end-to-end.

Covers the seven-point gate logic, the B-1 stub guard (evidence shape ->
INSUFFICIENT), and the Gate-2 end-to-end path through ``run_study`` (B-3), which
feeds criteria 6/7 from the real machinery — not the hand-passed stubs the earlier
version used.
"""

from __future__ import annotations

import math
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
from lab.research.reports.report import render_report, trade_statistics
from lab.research.strategies.reference import ReferenceMomentumSpec
from lab.research.study import run_study
from lab.research.trials.ledger import TrialLedger
from lab.research.validation.backtester import Trade
from lab.research.validation.costs import load_cost_model

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config"
IST = ZoneInfo("Asia/Kolkata")
PERIODS = 18750.0
NO_EMBARGO = timedelta(0)
THRESH = load_kill_gate_thresholds(REPO_CONFIG)


def _passing_inputs(**over: object) -> KillGateInputs:
    """A fully-passing KillGateInputs with well-shaped evidence; override to break one."""
    base: dict[str, object] = {
        "cpcv_path_sharpes": (1.2, 1.3, 1.4, 1.5, 1.5, 1.6, 1.6, 1.7, 1.8, 1.9),
        "dsr": 0.99,
        "pbo": 0.10,
        "profit_factor": 1.5,
        "top5_winners_fraction": 0.30,
        "expectancy": 0.0020,
        "round_trip_cost": 0.0018,
        "param_config_sharpes": {"base": 0.7, "threshold+": 0.6, "threshold-": 0.65},
        "has_tunable_params": True,
        "cross_symbol_sharpes": {"AAA": 0.6, "BBB": 0.7, "CCC": 0.55},
        "primary_symbol": "XYZ",
        "mc_shuffle_beat_fraction": 0.97,
        "two_engines_reconcile": True,
        "noise_survives": True,
        "regime_bucket_sharpes": {
            "2019|lovol|up": 0.6,
            "2020|hivol|down": 0.7,
            "2021|lovol|down": 0.55,
            "2022|hivol|up": 0.8,
        },
        "regime_positive_without_best": True,
    }
    base.update(over)
    return KillGateInputs(**base)  # type: ignore[arg-type]


# --- thresholds & core grading ---------------------------------------------- #
def test_thresholds_load_from_config() -> None:
    assert THRESH.dsr_min == 0.95
    assert THRESH.pbo_max == 0.20
    assert THRESH.cpcv_median_path_sharpe_min == 1.0
    assert THRESH.min_regime_buckets == 4
    assert THRESH.min_cross_symbols == 3


def test_kill_gate_passes_when_all_criteria_met() -> None:
    result = evaluate_kill_gate(_passing_inputs(), THRESH)
    assert result.verdict is Verdict.PASS
    assert result.passed
    assert all(c.passed for c in result.criteria)


def test_kill_gate_kills_on_any_single_failure() -> None:
    result = evaluate_kill_gate(_passing_inputs(dsr=0.50), THRESH)
    assert result.verdict is Verdict.KILL
    assert not result.passed


def test_nan_metric_fails_its_criterion() -> None:
    assert evaluate_kill_gate(_passing_inputs(dsr=float("nan")), THRESH).verdict is Verdict.KILL


# --- B-1 stub guard (evidence shape, not value) ----------------------------- #
def test_single_fake_regime_bucket_is_insufficient() -> None:
    # The (1.0,) single-bucket stub: a plausible number that must NOT be graded.
    result = evaluate_kill_gate(_passing_inputs(regime_bucket_sharpes={"only": 1.0}), THRESH)
    assert result.verdict is Verdict.INSUFFICIENT
    assert not result.passed  # INSUFFICIENT is not a pass


def test_too_few_cross_symbols_is_insufficient() -> None:
    result = evaluate_kill_gate(_passing_inputs(cross_symbol_sharpes={"AAA": 0.6}), THRESH)
    assert result.verdict is Verdict.INSUFFICIENT


def test_primary_symbol_as_holdout_is_insufficient() -> None:
    inputs = _passing_inputs(
        cross_symbol_sharpes={"XYZ": 0.6, "BBB": 0.7, "CCC": 0.5}, primary_symbol="XYZ"
    )
    assert evaluate_kill_gate(inputs, THRESH).verdict is Verdict.INSUFFICIENT


def test_tunable_strategy_with_only_base_config_is_insufficient() -> None:
    inputs = _passing_inputs(param_config_sharpes={"base": 0.7}, has_tunable_params=True)
    assert evaluate_kill_gate(inputs, THRESH).verdict is Verdict.INSUFFICIENT


def test_too_few_cpcv_paths_is_insufficient() -> None:
    # A CPCV "distribution" of 2 finite paths is a stub, not a distribution (< min 8).
    result = evaluate_kill_gate(_passing_inputs(cpcv_path_sharpes=(1.5, 1.6)), THRESH)
    assert result.verdict is Verdict.INSUFFICIENT


def test_parameter_free_single_base_config_is_sufficient() -> None:
    # A parameter-free strategy legitimately has one 'base' config — not a stub.
    inputs = _passing_inputs(param_config_sharpes={"base": 0.7}, has_tunable_params=False)
    assert evaluate_kill_gate(inputs, THRESH).verdict is Verdict.PASS


def test_extreme_but_real_values_are_not_flagged_as_stubs() -> None:
    # Shape is well-formed; the guard must not reject on magnitude — a real edge can
    # be strong in every bucket. Grading proceeds (to PASS here).
    strong = _passing_inputs(
        regime_bucket_sharpes={f"2020|b{k}|up": 3.0 for k in range(4)},
        param_config_sharpes={"base": 2.5, "threshold+": 2.4, "threshold-": 2.6},
    )
    assert evaluate_kill_gate(strong, THRESH).verdict is Verdict.PASS


# --- criterion 5: expectancy hurdle counted once (net > 0, not net > cost) --- #
def test_criterion5_expectancy_is_net_positive_not_double_counted() -> None:
    """Criterion 5's expectancy leg requires the per-trade edge to clear the round-trip
    cost COUNTED ONCE. `expectancy` arrives net of cost (mean Trade.net_return), so the
    leg is `net > 0` (equivalently gross expectancy > round-trip cost) — NOT the earlier
    accidental `net > cost` (gross > ~2x cost) hurdle. Guards killgate.py's criterion-5
    comparison from drifting back to the double-count.
    """
    cost = 0.0018

    # Gross edge in (1x, 2x) cost -> net in (0, cost): PASSES now; would have KILLed at 2x.
    marginal = _passing_inputs(expectancy=0.5 * cost, round_trip_cost=cost)
    marginal_result = evaluate_kill_gate(marginal, THRESH)
    assert next(c for c in marginal_result.criteria if c.number == 5).passed
    assert marginal_result.verdict is Verdict.PASS

    # Gross edge below 1x cost -> net < 0: cost-dead, FAILS.
    dead = _passing_inputs(expectancy=-0.2 * cost, round_trip_cost=cost)
    dead_result = evaluate_kill_gate(dead, THRESH)
    assert not next(c for c in dead_result.criteria if c.number == 5).passed
    assert dead_result.verdict is Verdict.KILL

    # Break-even (net == 0) is not a positive edge (strict >): FAILS.
    breakeven = evaluate_kill_gate(_passing_inputs(expectancy=0.0, round_trip_cost=cost), THRESH)
    assert not next(c for c in breakeven.criteria if c.number == 5).passed


# --- trade statistics ------------------------------------------------------- #
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


# --- Gate-2 end-to-end through run_study (B-3) ------------------------------ #
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


def test_gate2_end_to_end_reference_spec_through_run_study(tmp_path: Path) -> None:
    """A spec runs through the harness via ``run_study``: the DSR auto-deflates from
    the ledger's effective-N, criteria 6/7 are the REAL machinery's outputs (not
    stubs — this test would fail if that machinery were removed), and the verdict
    flows into the paper (Gate 2 / B-3)."""
    costs = load_cost_model(REPO_CONFIG)
    ledger = TrialLedger(tmp_path / "trials")
    candles = _synthetic()
    cross = {"S2": _synthetic(21), "S3": _synthetic(22), "S4": _synthetic(23)}

    report = run_study(
        ReferenceMomentumSpec(),
        candles,
        costs,
        THRESH,
        ledger,
        periods_per_year=PERIODS,
        cross_symbol_candles=cross,
        cpcv_embargo=NO_EMBARGO,
        # 4 day-buckets from the 4-day series -> satisfies the regime evidence floor.
        regime_labeler=lambda t: str(t.entry_time.astimezone(IST).date()),
        pbo_splits=4,
    )

    # Evidence is well-shaped (3 held-out symbols, 4 regime buckets), so the gate
    # GRADES rather than returning INSUFFICIENT; a no-edge rule is an honest KILL.
    assert report.kill_gate.verdict is Verdict.KILL
    assert ledger.count() == 1  # parameter-free -> a single base trial logged

    findings = tmp_path / "RESEARCH_FINDINGS.md"
    findings.write_text("# Findings\n", encoding="utf-8")
    append_study_section(report, findings_path=findings)
    text = findings.read_text(encoding="utf-8")
    assert "reference_momentum" in text
    assert "KILL" in text
    assert "Seven-point kill-gate" in render_report(report)
