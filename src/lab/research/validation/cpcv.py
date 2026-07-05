"""Combinatorial Purged Cross-Validation (Phase 2, P2.2).

Partition the trade series into ``N`` time-ordered groups; for every combination
of ``k`` test groups (``C(N,k)`` of them) pool the out-of-sample returns and score
their Sharpe. The distribution of those path-Sharpes is what the kill-gate judges:
narrow & positive = robust; wild variance = fragile (Part III Layer 2). The number
of reconstructed paths is ``phi = C(N,k)·k/N``.

Note: the strategies here are deterministic rules (no per-fold model fitting), so
reconstructed full-coverage paths would be identical; the informative distribution
is therefore taken over the purged ``C(N,k)`` test-group combinations. The
purge/embargo splitter governs the training-based path (meta-labeling, Phase 4.5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import numpy.typing as npt

from lab.research.validation.sharpe import annualized_sharpe


@dataclass(frozen=True, slots=True)
class CPCVResult:
    """The path-Sharpe distribution from a CPCV run."""

    path_sharpes: tuple[float, ...]
    n_groups: int
    k_test_groups: int
    n_paths: float  # phi = C(N,k)·k/N

    def _finite(self) -> np.ndarray:
        arr = np.array(self.path_sharpes, dtype=np.float64)
        return arr[np.isfinite(arr)]

    @property
    def median_path_sharpe(self) -> float:
        """Median across finite path-Sharpes (kill-gate criterion 1)."""
        finite = self._finite()
        return float(np.median(finite)) if finite.size else float("nan")

    @property
    def positive_fraction(self) -> float:
        """Fraction of finite paths with a positive Sharpe (criterion 4a)."""
        finite = self._finite()
        return float(np.mean(finite > 0.0)) if finite.size else float("nan")

    @property
    def tenth_percentile(self) -> float:
        """10th-percentile path-Sharpe (criterion 4b)."""
        finite = self._finite()
        return float(np.percentile(finite, 10)) if finite.size else float("nan")


def combinatorial_purged_cv(
    returns: npt.ArrayLike,
    *,
    n_groups: int,
    k_test_groups: int,
    periods_per_year: float,
) -> CPCVResult:
    """Run CPCV over a return series and return the path-Sharpe distribution.

    Args:
        returns: Per-trade (or per-period) net returns, time-ordered.
        n_groups: Number of groups ``N`` (>= 2).
        k_test_groups: Test groups per combination ``k`` (1 <= k < N).
        periods_per_year: Annualization factor (fixed Sharpe convention).
    """
    if n_groups < 2:
        raise ValueError(f"n_groups must be >= 2; got {n_groups}")
    if not 1 <= k_test_groups < n_groups:
        raise ValueError(f"k_test_groups must be in [1, n_groups); got {k_test_groups}")
    values = np.asarray(returns, dtype=np.float64)
    if values.size < n_groups:
        raise ValueError(f"need at least {n_groups} returns; got {values.size}")

    groups = np.array_split(np.arange(values.size), n_groups)
    path_sharpes = [
        annualized_sharpe(values[np.concatenate([groups[g] for g in combo])], periods_per_year)
        for combo in combinations(range(n_groups), k_test_groups)
    ]
    n_paths = math.comb(n_groups, k_test_groups) * k_test_groups / n_groups
    return CPCVResult(tuple(path_sharpes), n_groups, k_test_groups, n_paths)
