"""Tests for the panel-scope foundation: frozen-config + threshold loaders, aggregation.

The orchestrator (run_panel_study) is covered separately; these pin the pieces that must
be exactly right BEFORE any panel scoring runs -- the Lock-A frozen sets, the pinned panel
thresholds, and the contribute-zero equal-weight aggregation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from lab.core.types import BarInterval, Candle, Side, StrategySignal, Verdict
from lab.research.panel import (
    PanelThresholds,
    StudyPanel,
    equal_weight_portfolio_stream,
    load_panel_thresholds,
    load_study_panel,
    render_panel_report,
    run_panel_study,
)
from lab.research.reports.killgate import load_kill_gate_thresholds
from lab.research.trials.ledger import TrialLedger
from lab.research.validation.backtester import Trade
from lab.research.validation.costs import load_cost_model

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config"
IST = ZoneInfo("Asia/Kolkata")

# The Lock-A frozen sets (operator ruling, 2026-07-09). This test is the regression pin:
# the config on main must match the ruling, verbatim and in order.
FROZEN_PANEL = (
    "HDFCBANK",
    "RELIANCE",
    "ICICIBANK",
    "TCS",
    "TATASTEEL",
    "ULTRACEMCO",
    "TITAN",
    "COALINDIA",
    "DRREDDY",
    "TATACONSUM",
)
FROZEN_HOLDOUT_6D = ("INFY", "SBIN", "SUNPHARMA", "ADANIPORTS", "NESTLEIND")


def test_load_study_panel_matches_the_frozen_ruling() -> None:
    panel = load_study_panel(REPO_CONFIG)
    assert panel.panel == FROZEN_PANEL  # exact names + order
    assert panel.holdout_6d == FROZEN_HOLDOUT_6D
    assert not set(panel.panel) & set(panel.holdout_6d)  # disjoint (6d = never-scored)
    assert "large-cap" in panel.scope_caveat.lower()  # scope stamp present


def test_study_panel_rejects_holdout_overlapping_panel() -> None:
    with pytest.raises(ValueError, match="OUTSIDE the panel"):
        StudyPanel(panel=("AAA", "BBB"), holdout_6d=("BBB", "CCC"), scope_caveat="x")


def test_load_panel_thresholds_are_the_pinned_values() -> None:
    thresholds = load_panel_thresholds(REPO_CONFIG)
    assert thresholds == PanelThresholds(
        breadth_median_path_sharpe_min=1.0,
        breadth_positive_fraction_min=0.60,
        noise_survives_fraction_min=0.60,
        two_engine_reconcile_fraction_min=1.0,
        min_panel_symbols=6,
        min_portfolio_days=250,
    )


def _trade(day: int, net: float) -> Trade:
    entry = datetime(2024, 7, day, 10, 0, tzinfo=IST)
    # gross_return = net, cost_fraction = 0.0 -> the net_return property equals net.
    return Trade(Side.LONG, entry, entry + timedelta(minutes=5), 100.0, 100.0 * (1 + net), net, 0.0)


def test_equal_weight_portfolio_is_contribute_zero_with_fixed_divisor() -> None:
    # Symbol A trades both days; symbol B only day 15. With a FIXED divisor = panel size 2,
    # B contributing nothing on day 16 dilutes toward zero (not re-weighted onto A).
    per_symbol = {
        "A": [_trade(15, 0.10), _trade(16, 0.20)],
        "B": [_trade(15, 0.30)],
    }
    stream = equal_weight_portfolio_stream(per_symbol, n_panel_symbols=2)
    assert [d.day for d in stream.days] == [15, 16]
    # day 15: (0.10 + 0.30) / 2 = 0.20 ; day 16: (0.20 + 0.0) / 2 = 0.10 (B contributes zero)
    assert stream.returns == pytest.approx((0.20, 0.10))
    assert [t.hour for t in stream.entry_times] == [0, 0]  # day-start labels


def test_equal_weight_portfolio_sums_same_day_trades_per_symbol() -> None:
    # Two trades for A on the same day are summed before the equal-weight average.
    per_symbol = {"A": [_trade(15, 0.10), _trade(15, 0.05)], "B": [_trade(15, 0.03)]}
    stream = equal_weight_portfolio_stream(per_symbol, n_panel_symbols=2)
    assert stream.returns == pytest.approx(((0.15 + 0.03) / 2,))  # (0.10+0.05) summed for A


# --- orchestrator (end-to-end) ---------------------------------------------- #
@dataclass(frozen=True, slots=True)
class _ThresholdSpec:
    """A no-edge parametrized reference: trade the bar's direction if |move| > threshold."""

    threshold: float
    name: str = "threshold_panel"
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


def _spec_factory(params: Mapping[str, float]) -> _ThresholdSpec:
    return _ThresholdSpec(threshold=float(params["threshold"]))


def _panel_series(
    symbol: str, seed: int, *, days: int = 30, bars_per_day: int = 20
) -> list[Candle]:
    rng = np.random.default_rng(seed)
    candles: list[Candle] = []
    price = 100.0
    start = datetime(2024, 1, 1, 9, 15, tzinfo=IST)
    for day in range(days):
        open_ts = start + timedelta(days=day)
        for bar in range(bars_per_day):
            prev = price
            price = prev * float(np.exp(rng.normal(0.0, 0.003)))
            candles.append(
                Candle(
                    symbol,
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


# min_portfolio_days lowered to 20 so the synthetic ~30-day panels certify; the other
# thresholds are the pinned values.
_TEST_PANEL_THRESH = PanelThresholds(
    breadth_median_path_sharpe_min=1.0,
    breadth_positive_fraction_min=0.60,
    noise_survives_fraction_min=0.60,
    two_engine_reconcile_fraction_min=1.0,
    min_panel_symbols=6,
    min_portfolio_days=20,
)


def _panels(
    n_panel: int = 6, n_holdout: int = 3, *, days: int = 30
) -> tuple[dict[str, list[Candle]], dict[str, list[Candle]]]:
    panel = {f"P{i}": _panel_series(f"P{i}", 100 + i, days=days) for i in range(n_panel)}
    holdout = {f"H{i}": _panel_series(f"H{i}", 200 + i, days=days) for i in range(n_holdout)}
    return panel, holdout


def test_run_panel_study_no_edge_does_not_pass_and_logs_k_streams(tmp_path: Path) -> None:
    panel, holdout = _panels()
    ledger = TrialLedger(tmp_path / "trials")
    report = run_panel_study(
        _spec_factory,
        {"threshold": 0.0},
        {"threshold": 0.002},
        panel,
        holdout,
        load_cost_model(REPO_CONFIG),
        load_kill_gate_thresholds(REPO_CONFIG),
        _TEST_PANEL_THRESH,
        ledger,
        scope_caveat="Scoped to NIFTY-50 large-caps (test caveat).",
        cpcv_embargo=timedelta(0),  # compressed synthetic span
        noise_seeds=2,
        mc_shuffles=50,
    )
    # A directionless rule on random panels must NOT pass -- honest KILL or INSUFFICIENT.
    assert report.verdict in {Verdict.KILL, Verdict.INSUFFICIENT}
    assert report.verdict is not Verdict.PASS
    # The scope caveat is stamped on the result and rendered.
    assert report.scope_caveat
    assert "SCOPE" in render_panel_report(report)
    # LEDGER RULE: K aggregate-portfolio streams (base + 2 threshold neighbours = 3), NOT
    # N*K per-symbol streams (6*3 = 18). The panel is scope, not extra trials.
    assert ledger.count() == 3
    # Breadth is computed per panel symbol.
    assert set(report.breadth.per_symbol_path_sharpe) == set(panel)


def test_run_panel_study_insufficient_when_panel_too_thin(tmp_path: Path) -> None:
    # Fewer scored panel symbols than min_panel_symbols -> cannot certify (INSUFFICIENT),
    # short-circuited before any scoring; fail-closed like the single-symbol floors.
    panel, holdout = _panels(n_panel=3)
    report = run_panel_study(
        _spec_factory,
        {"threshold": 0.0},
        {"threshold": 0.002},
        panel,
        holdout,
        load_cost_model(REPO_CONFIG),
        load_kill_gate_thresholds(REPO_CONFIG),
        _TEST_PANEL_THRESH,
        TrialLedger(tmp_path / "trials"),
        scope_caveat="scope",
        cpcv_embargo=timedelta(0),
        noise_seeds=1,
        mc_shuffles=10,
    )
    assert report.verdict is Verdict.INSUFFICIENT
    assert report.n_panel_symbols == 3
