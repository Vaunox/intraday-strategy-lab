"""Tests for the Sharpe convention and PSR/DSR math (P2.2)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from lab.research.validation.metrics import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
)
from lab.research.validation.sharpe import (
    annualized_sharpe,
    per_period_sharpe,
    realized_periods_per_year,
    return_stats,
)


# --- Sharpe convention ------------------------------------------------------ #
def test_per_period_sharpe_hand_computed() -> None:
    # mean=3, sample std=sqrt(2.5); Sharpe = 3 / sqrt(2.5).
    assert per_period_sharpe([1, 2, 3, 4, 5]) == pytest.approx(3.0 / math.sqrt(2.5))


def test_annualized_scales_by_sqrt_periods() -> None:
    assert annualized_sharpe([1, 2, 3, 4, 5], periods_per_year=4) == pytest.approx(
        2.0 * 3.0 / math.sqrt(2.5)
    )


def test_sharpe_is_nan_on_degenerate_input() -> None:
    assert math.isnan(per_period_sharpe([1.0]))
    assert math.isnan(per_period_sharpe([2.0, 2.0, 2.0]))  # zero variance


def test_realized_periods_per_year_is_observations_over_span() -> None:
    assert realized_periods_per_year(1000, 5.0) == pytest.approx(200.0)
    assert realized_periods_per_year(93_750, 5.0) == pytest.approx(18_750.0)


def test_realized_periods_per_year_fails_closed_on_degenerate_input() -> None:
    assert math.isnan(realized_periods_per_year(0, 5.0))  # no trades
    assert math.isnan(realized_periods_per_year(100, 0.0))  # zero span
    assert math.isnan(realized_periods_per_year(100, -1.0))  # negative span


def test_annualization_uses_realized_not_fixed_frequency() -> None:
    # PINS the realized-frequency convention against a regression to the fixed 18_750
    # constant. A low-frequency strategy (500 trades / 5y = 100/yr) and a near-every-bar
    # one (93_750 / 5y = 18_750/yr) must each annualize by sqrt(their OWN rate) -- a
    # ~13.7x gap the old fixed constant would erase.
    rng = np.random.default_rng(0)
    returns = rng.normal(0.001, 0.01, size=500)
    per_period = per_period_sharpe(returns)

    low_ppy = realized_periods_per_year(500, 5.0)  # 100/yr
    high_ppy = realized_periods_per_year(93_750, 5.0)  # 18_750/yr
    assert annualized_sharpe(returns, low_ppy) == pytest.approx(per_period * math.sqrt(100.0))
    assert annualized_sharpe(returns, high_ppy) == pytest.approx(per_period * math.sqrt(18_750.0))
    # The low-frequency Sharpe must NOT be scaled by the old fixed 18_750 (the bug):
    assert annualized_sharpe(returns, low_ppy) != pytest.approx(per_period * math.sqrt(18_750.0))
    # ...and its scale factor is exactly sqrt(realized rate), proving realized scaling.
    assert annualized_sharpe(returns, low_ppy) / per_period == pytest.approx(math.sqrt(100.0))


def test_return_stats_symmetric() -> None:
    stats_ = return_stats([-2.0, -1.0, 0.0, 1.0, 2.0])
    assert stats_.n == 5
    assert stats_.skew == pytest.approx(0.0, abs=1e-9)


# --- PSR -------------------------------------------------------------------- #
def test_psr_is_half_when_observed_equals_benchmark() -> None:
    assert probabilistic_sharpe_ratio(0.1, 0.1, 100, 0.0, 3.0) == pytest.approx(0.5)


def test_psr_increases_with_observed_sharpe() -> None:
    low = probabilistic_sharpe_ratio(0.1, 0.0, 100, 0.0, 3.0)
    high = probabilistic_sharpe_ratio(0.2, 0.0, 100, 0.0, 3.0)
    assert 0.5 < low < high < 1.0


def test_psr_reference_value() -> None:
    # observed 0.1, benchmark 0, n=101, Gaussian returns -> Phi(0.9975) ~ 0.8407.
    assert probabilistic_sharpe_ratio(0.1, 0.0, 101, 0.0, 3.0) == pytest.approx(0.8407, abs=1e-3)


# --- expected max Sharpe / DSR ---------------------------------------------- #
def test_expected_max_sharpe_monotonic_in_trials() -> None:
    assert expected_max_sharpe(1, 0.5) == 0.0  # a single trial has no inflation
    assert 0.0 < expected_max_sharpe(5, 0.5) < expected_max_sharpe(50, 0.5)


def test_dsr_deflates_with_more_effective_trials() -> None:
    few = deflated_sharpe_ratio(0.15, 500, 0.0, 3.0, effective_trials=2, trial_sharpe_std=0.1)
    many = deflated_sharpe_ratio(0.15, 500, 0.0, 3.0, effective_trials=200, trial_sharpe_std=0.1)
    assert 0.0 <= many < few <= 1.0  # more trials -> harder bar -> lower DSR


def test_dsr_single_trial_equals_psr_against_zero() -> None:
    dsr = deflated_sharpe_ratio(0.12, 300, 0.0, 3.0, effective_trials=1, trial_sharpe_std=0.1)
    psr = probabilistic_sharpe_ratio(0.12, 0.0, 300, 0.0, 3.0)
    assert dsr == pytest.approx(psr)
