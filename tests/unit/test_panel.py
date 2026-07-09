"""Tests for the panel-scope foundation: frozen-config + threshold loaders, aggregation.

The orchestrator (run_panel_study) is covered separately; these pin the pieces that must
be exactly right BEFORE any panel scoring runs -- the Lock-A frozen sets, the pinned panel
thresholds, and the contribute-zero equal-weight aggregation.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from lab.core.types import Side
from lab.research.panel import (
    PanelThresholds,
    StudyPanel,
    equal_weight_portfolio_stream,
    load_panel_thresholds,
    load_study_panel,
)
from lab.research.validation.backtester import Trade

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
