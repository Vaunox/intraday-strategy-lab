"""Probabilistic and Deflated Sharpe Ratios (Phase 2, P2.2).

Implements Bailey & López de Prado's PSR and DSR. The DSR deflates the observed
Sharpe by the expected maximum Sharpe under a null of *N independent trials* —
where **N is the effective (cluster-adjusted) trial count** from the honest
ledger, never a raw variant count (Inviolable Rule 4). All Sharpes here are
per-period (not annualized); annualization is a reporting convention applied
separately.
"""

from __future__ import annotations

import math

from scipy import stats

#: Euler-Mascheroni constant, used in the expected-maximum-Sharpe estimate.
EULER_MASCHERONI = 0.5772156649015329


def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    benchmark_sharpe: float,
    n: int,
    skew: float,
    kurtosis: float,
) -> float:
    """Return PSR: P(true Sharpe > ``benchmark_sharpe``) given the sample moments.

    Args:
        observed_sharpe: Per-period Sharpe estimate.
        benchmark_sharpe: Per-period Sharpe to test against.
        n: Number of return observations.
        skew: Sample skewness of returns.
        kurtosis: Sample non-excess kurtosis of returns (normal == 3).
    """
    if n < 2:
        return float("nan")
    variance = 1.0 - skew * observed_sharpe + (kurtosis - 1.0) / 4.0 * observed_sharpe**2
    denominator = math.sqrt(max(variance, 1e-12))
    z = (observed_sharpe - benchmark_sharpe) * math.sqrt(n - 1) / denominator
    return float(stats.norm.cdf(z))


def expected_max_sharpe(n_trials: float, trial_sharpe_std: float) -> float:
    """Expected maximum Sharpe under a null of ``n_trials`` independent trials.

    This is the multiple-testing-adjusted benchmark the DSR must clear. A single
    trial has no inflation; more (independent) trials raise the bar.
    """
    if n_trials <= 1.0 or trial_sharpe_std <= 0.0:
        return 0.0
    left = (1.0 - EULER_MASCHERONI) * stats.norm.ppf(1.0 - 1.0 / n_trials)
    right = EULER_MASCHERONI * stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(trial_sharpe_std * (left + right))


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n: int,
    skew: float,
    kurtosis: float,
    *,
    effective_trials: float,
    trial_sharpe_std: float,
) -> float:
    """Return the Deflated Sharpe Ratio (a probability in ``[0, 1]``).

    Args:
        observed_sharpe: Per-period Sharpe of the strategy under test.
        n: Number of return observations.
        skew: Sample skewness of returns.
        kurtosis: Sample non-excess kurtosis of returns.
        effective_trials: The EFFECTIVE (cluster-adjusted) number of independent
            trials, from the honest ledger — never a raw variant count.
        trial_sharpe_std: Standard deviation of the per-period Sharpes across the
            trials (the dispersion of the search).
    """
    benchmark = expected_max_sharpe(effective_trials, trial_sharpe_std)
    return probabilistic_sharpe_ratio(observed_sharpe, benchmark, n, skew, kurtosis)
