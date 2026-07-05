"""Fixed Sharpe convention and return statistics (Phase 2, Part III Layer 2).

A bare "Sharpe" is meaningless intraday, so the convention is pinned in config
and applied identically to every study: Sharpes are annualized by
sqrt(``periods_per_year``) and scaled on **in-market** (position-held) periods,
not calendar time. The per-period Sharpe, sample length, skew, and kurtosis feed
the Probabilistic/Deflated Sharpe math in :mod:`lab.research.validation.metrics`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats

from lab.core.config import Settings

FloatArray = npt.NDArray[np.float64]
Returns = Sequence[float] | FloatArray


@dataclass(frozen=True, slots=True)
class SharpeConvention:
    """The pinned Sharpe convention: annualization factor and scaling basis."""

    periods_per_year: float
    basis: str = "in_market"

    @classmethod
    def from_settings(cls, settings: Settings) -> SharpeConvention:
        """Load the convention from the ``sharpe`` config section."""
        raw = settings.raw.get("sharpe", {})
        return cls(
            periods_per_year=float(raw["periods_per_year"]),
            basis=str(raw.get("basis", "in_market")),
        )

    def annualize(self, per_period_sharpe: float) -> float:
        """Annualize a per-period Sharpe by sqrt(periods_per_year)."""
        return per_period_sharpe * math.sqrt(self.periods_per_year)


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
    """Return the annualized Sharpe under the fixed convention."""
    return per_period_sharpe(returns) * math.sqrt(periods_per_year)


def return_stats(returns: Returns) -> ReturnStats:
    """Compute per-period Sharpe, sample length, skew, and non-excess kurtosis."""
    values = np.asarray(returns, dtype=np.float64)
    n = int(values.size)
    skew = float(stats.skew(values)) if n > 2 else 0.0
    kurtosis = float(stats.kurtosis(values, fisher=False)) if n > 3 else 3.0
    return ReturnStats(sharpe=per_period_sharpe(values), n=n, skew=skew, kurtosis=kurtosis)
