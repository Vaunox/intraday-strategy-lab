"""Sharpe convention (realized frequency) and return statistics (Phase 2, Part III Layer 2).

A bare "Sharpe" is meaningless intraday, so the annualization convention is pinned and
applied identically to every study: a series' per-period Sharpe is annualized by
``sqrt(`` its REALIZED per-year frequency ``)`` — the number of return observations the
strategy ACTUALLY produced over its operating span (:func:`realized_periods_per_year`),
not a fixed calendar constant. This REPLACES the earlier fixed ``periods_per_year = 18750``
(all 5-min bars in a year), which silently over-annualized every strategy that did not
trade on ~every bar: a few-hundred-trades-a-year strategy was scaled as though it traded
~18750 times a year, inflating the Sharpe-magnitude kill-gate criteria (1/4/6a/7). That was
code-vs-blueprint drift — the blueprint's convention scales on **in-market (position-held)
events, not calendar time**. The per-period Sharpe, sample length, skew, and kurtosis feed
the Probabilistic/Deflated Sharpe math in :mod:`lab.research.validation.metrics`, which
consumes the NON-annualized per-period Sharpe (annualization is only the reporting scale on
the kill-gate's Sharpe criteria).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats

FloatArray = npt.NDArray[np.float64]
Returns = Sequence[float] | FloatArray


def realized_periods_per_year(n_obs: int, span_years: float) -> float:
    """Return the REALIZED annualization frequency: return observations per year.

    The Sharpe annualization factor is ``sqrt(realized_periods_per_year)``. Unlike a
    fixed calendar constant, this reflects how often the strategy ACTUALLY produced a
    return — ``n_obs`` trade returns over ``span_years`` of operation — so a strategy
    trading a few hundred times a year is annualized by ``sqrt(hundreds)``, never
    ``sqrt(all 5-min bars)``. Returns NaN for a degenerate span or empty series, so the
    dependent annualized Sharpe (and its kill-gate criterion) fails closed.
    """
    if span_years <= 0.0 or n_obs < 1:
        return float("nan")
    return n_obs / span_years


@dataclass(frozen=True, slots=True)
class ReturnStats:
    """Per-period Sharpe plus the moments the (D/P)SR math needs."""

    sharpe: float  # per-period (NOT annualized)
    n: int
    skew: float
    kurtosis: float  # non-excess (normal == 3)


def per_period_sharpe(returns: Returns) -> float:
    """Return the non-annualized Sharpe (mean / sample std) of ``returns``."""
    values = np.asarray(returns, dtype=np.float64)
    if values.size < 2:
        return float("nan")
    std = float(values.std(ddof=1))
    if std == 0.0:
        return float("nan")
    return float(values.mean()) / std


def annualized_sharpe(returns: Returns, periods_per_year: float) -> float:
    """Annualize a per-period Sharpe by ``sqrt(periods_per_year)``.

    ``periods_per_year`` is the series' REALIZED frequency from
    :func:`realized_periods_per_year` (observations per operating year), not a fixed
    calendar constant — see the module docstring.
    """
    return per_period_sharpe(returns) * math.sqrt(periods_per_year)


def return_stats(returns: Returns) -> ReturnStats:
    """Compute per-period Sharpe, sample length, skew, and non-excess kurtosis."""
    values = np.asarray(returns, dtype=np.float64)
    n = int(values.size)
    skew = float(stats.skew(values)) if n > 2 else 0.0
    kurtosis = float(stats.kurtosis(values, fisher=False)) if n > 3 else 3.0
    return ReturnStats(sharpe=per_period_sharpe(values), n=n, skew=skew, kurtosis=kurtosis)
