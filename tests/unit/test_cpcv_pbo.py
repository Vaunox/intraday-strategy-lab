"""Tests for CPCV path distribution and PBO via CSCV (P2.2)."""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from lab.research.validation.cpcv import combinatorial_purged_cv
from lab.research.validation.pbo import probability_of_backtest_overfitting

PERIODS = 18750.0
IST = ZoneInfo("Asia/Kolkata")
NO_EMBARGO = timedelta(0)


def _sequential_times(
    n: int, *, spacing: timedelta = timedelta(minutes=5), hold: timedelta = timedelta(minutes=4)
) -> tuple[list[datetime], list[datetime]]:
    """Time-ordered, non-overlapping label windows (a 1-min gap between trades)."""
    entries = [datetime(2024, 7, 15, 9, 15, tzinfo=IST) + i * spacing for i in range(n)]
    exits = [e + hold for e in entries]
    return entries, exits


# --- CPCV ------------------------------------------------------------------- #
def test_cpcv_path_count_and_size() -> None:
    rng = np.random.default_rng(0)
    returns = rng.normal(0.001, 0.002, size=180)
    entries, exits = _sequential_times(180)
    result = combinatorial_purged_cv(
        returns,
        entries,
        exits,
        n_groups=6,
        k_test_groups=2,
        periods_per_year=PERIODS,
        embargo=NO_EMBARGO,
    )
    assert len(result.path_sharpes) == math.comb(6, 2)  # 15 combinations
    assert result.n_paths == pytest.approx(15 * 2 / 6)  # phi = 5


def test_cpcv_positive_series_has_positive_distribution() -> None:
    rng = np.random.default_rng(1)
    returns = rng.normal(0.005, 0.005, size=200)  # clearly positive edge
    entries, exits = _sequential_times(200)
    result = combinatorial_purged_cv(
        returns,
        entries,
        exits,
        n_groups=6,
        k_test_groups=2,
        periods_per_year=PERIODS,
        embargo=NO_EMBARGO,
    )
    assert result.median_path_sharpe > 0
    assert result.positive_fraction == 1.0
    assert result.tenth_percentile > 0


def test_cpcv_noise_series_is_centered_near_zero() -> None:
    rng = np.random.default_rng(2)
    returns = rng.normal(0.0, 0.01, size=300)
    entries, exits = _sequential_times(300)
    result = combinatorial_purged_cv(
        returns,
        entries,
        exits,
        n_groups=8,
        k_test_groups=2,
        periods_per_year=PERIODS,
        embargo=NO_EMBARGO,
    )
    assert 0.2 < result.positive_fraction < 0.8  # no persistent edge


def test_cpcv_rejects_bad_params() -> None:
    entries, exits = _sequential_times(10)
    with pytest.raises(ValueError, match="k_test_groups"):
        combinatorial_purged_cv(
            [0.1] * 10, entries, exits, n_groups=4, k_test_groups=4, periods_per_year=PERIODS
        )


def test_cpcv_rejects_mismatched_time_lengths() -> None:
    entries, exits = _sequential_times(10)
    with pytest.raises(ValueError, match="equal length"):
        combinatorial_purged_cv(
            [0.1] * 10, entries[:9], exits, n_groups=4, k_test_groups=2, periods_per_year=PERIODS
        )


def test_cpcv_no_embargo_is_a_noop_for_sequential_trades() -> None:
    # With non-overlapping labels and no embargo, purging removes nothing: each
    # path-Sharpe equals the plain Sharpe over its two pooled groups.
    rng = np.random.default_rng(7)
    returns = rng.normal(0.003, 0.004, size=60)
    entries, exits = _sequential_times(60)
    result = combinatorial_purged_cv(
        returns,
        entries,
        exits,
        n_groups=6,
        k_test_groups=2,
        periods_per_year=PERIODS,
        embargo=NO_EMBARGO,
    )
    assert result.n_finite_paths == math.comb(6, 2)  # nothing purged to unscorable


def test_cpcv_embargo_purges_boundary_trades() -> None:
    # Day-spaced trades so a one-day embargo straddles group boundaries and drops
    # boundary trades from the pooled test set, changing the distribution.
    rng = np.random.default_rng(8)
    returns = rng.normal(0.004, 0.004, size=36)
    entries = [datetime(2024, 7, 1, 9, 15, tzinfo=IST) + timedelta(days=i) for i in range(36)]
    exits = [e + timedelta(minutes=5) for e in entries]
    unpurged = combinatorial_purged_cv(
        returns,
        entries,
        exits,
        n_groups=6,
        k_test_groups=2,
        periods_per_year=PERIODS,
        embargo=NO_EMBARGO,
    )
    embargoed = combinatorial_purged_cv(
        returns,
        entries,
        exits,
        n_groups=6,
        k_test_groups=2,
        periods_per_year=PERIODS,
        embargo=timedelta(days=1),
    )
    assert embargoed.path_sharpes != unpurged.path_sharpes  # the embargo actually purged


# --- PBO -------------------------------------------------------------------- #
def _daily_times(n: int) -> tuple[list[datetime], list[datetime]]:
    """One point-interval per row on consecutive days (rows are time-ordered)."""
    days = [datetime(2024, 1, 1, tzinfo=IST) + timedelta(days=i) for i in range(n)]
    return days, list(days)


def test_pbo_low_for_persistently_best_config() -> None:
    rng = np.random.default_rng(3)
    strong = rng.normal(0.02, 0.01, size=(240, 1))  # config 0: real, persistent edge
    noise = rng.normal(0.0, 0.01, size=(240, 4))
    matrix = np.hstack([strong, noise])
    entries, exits = _daily_times(240)
    result = probability_of_backtest_overfitting(
        matrix, entries, exits, n_splits=8, embargo=timedelta(0)
    )
    assert result.pbo < 0.2  # the IS-best (config 0) stays best OOS -> not overfit


def test_pbo_high_for_pure_noise_configs() -> None:
    rng = np.random.default_rng(4)
    matrix = rng.normal(0.0, 0.01, size=(240, 6))  # no config has a real edge
    entries, exits = _daily_times(240)
    result = probability_of_backtest_overfitting(
        matrix, entries, exits, n_splits=8, embargo=timedelta(0)
    )
    assert result.pbo > 0.35  # chasing the IS-best buys ~nothing OOS


def test_pbo_embargo_purges_boundary_rows() -> None:
    # A 1-day embargo purges OOS rows adjacent to an IS block, via the SAME shared
    # primitive CPCV uses — so the CSCV logit distribution shifts vs no embargo.
    rng = np.random.default_rng(9)
    matrix = rng.normal(0.0, 0.01, size=(240, 4))
    entries, exits = _daily_times(240)
    plain = probability_of_backtest_overfitting(
        matrix, entries, exits, n_splits=8, embargo=timedelta(0)
    )
    embargoed = probability_of_backtest_overfitting(
        matrix, entries, exits, n_splits=8, embargo=timedelta(days=1)
    )
    assert not np.array_equal(plain.logits, embargoed.logits)  # the embargo actually purged


def test_pbo_requires_multiple_configs() -> None:
    entries, exits = _daily_times(100)
    with pytest.raises(ValueError, match="at least 2 config"):
        probability_of_backtest_overfitting(np.zeros((100, 1)), entries, exits)


def test_pbo_rejects_mismatched_row_times() -> None:
    entries, exits = _daily_times(10)
    with pytest.raises(ValueError, match="one entry per matrix row"):
        probability_of_backtest_overfitting(np.zeros((240, 4)), entries, exits, n_splits=8)
