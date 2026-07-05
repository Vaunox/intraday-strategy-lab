"""Tests for CPCV path distribution and PBO via CSCV (P2.2)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from lab.research.validation.cpcv import combinatorial_purged_cv
from lab.research.validation.pbo import probability_of_backtest_overfitting

PERIODS = 18750.0


# --- CPCV ------------------------------------------------------------------- #
def test_cpcv_path_count_and_size() -> None:
    rng = np.random.default_rng(0)
    returns = rng.normal(0.001, 0.002, size=180)
    result = combinatorial_purged_cv(returns, n_groups=6, k_test_groups=2, periods_per_year=PERIODS)
    assert len(result.path_sharpes) == math.comb(6, 2)  # 15 combinations
    assert result.n_paths == pytest.approx(15 * 2 / 6)  # phi = 5


def test_cpcv_positive_series_has_positive_distribution() -> None:
    rng = np.random.default_rng(1)
    returns = rng.normal(0.005, 0.005, size=200)  # clearly positive edge
    result = combinatorial_purged_cv(returns, n_groups=6, k_test_groups=2, periods_per_year=PERIODS)
    assert result.median_path_sharpe > 0
    assert result.positive_fraction == 1.0
    assert result.tenth_percentile > 0


def test_cpcv_noise_series_is_centered_near_zero() -> None:
    rng = np.random.default_rng(2)
    returns = rng.normal(0.0, 0.01, size=300)
    result = combinatorial_purged_cv(returns, n_groups=8, k_test_groups=2, periods_per_year=PERIODS)
    assert 0.2 < result.positive_fraction < 0.8  # no persistent edge


def test_cpcv_rejects_bad_params() -> None:
    with pytest.raises(ValueError, match="k_test_groups"):
        combinatorial_purged_cv([0.1] * 10, n_groups=4, k_test_groups=4, periods_per_year=PERIODS)


# --- PBO -------------------------------------------------------------------- #
def test_pbo_low_for_persistently_best_config() -> None:
    rng = np.random.default_rng(3)
    strong = rng.normal(0.02, 0.01, size=(240, 1))  # config 0: real, persistent edge
    noise = rng.normal(0.0, 0.01, size=(240, 4))
    matrix = np.hstack([strong, noise])
    result = probability_of_backtest_overfitting(matrix, n_splits=8)
    assert result.pbo < 0.2  # the IS-best (config 0) stays best OOS -> not overfit


def test_pbo_high_for_pure_noise_configs() -> None:
    rng = np.random.default_rng(4)
    matrix = rng.normal(0.0, 0.01, size=(240, 6))  # no config has a real edge
    result = probability_of_backtest_overfitting(matrix, n_splits=8)
    assert result.pbo > 0.35  # chasing the IS-best buys ~nothing OOS


def test_pbo_requires_multiple_configs() -> None:
    with pytest.raises(ValueError, match="at least 2 config"):
        probability_of_backtest_overfitting(np.zeros((100, 1)))
