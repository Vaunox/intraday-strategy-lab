"""Combinatorial Purged Cross-Validation (Phase 2, P2.2).

Partition the trade series into ``N`` time-ordered groups; for every combination
of ``k`` test groups (``C(N,k)`` of them) pool the out-of-sample returns and score
their Sharpe. The distribution of those path-Sharpes is what the kill-gate judges:
narrow & positive = robust; wild variance = fragile (Part III Layer 2). The number
of reconstructed paths is ``phi = C(N,k)·k/N``.

**Purging & embargo.** Each combination's pooled test trades are purged of any
observation whose label window overlaps a *train* group's span, and a further
embargo buffer (default one trading day) is applied at the test/train boundary —
both via the single shared :func:`~lab.research.validation.splitter.label_overlaps`
primitive, so CPCV and the purged k-fold splitter cannot drift apart. For a single
symbol whose intraday positions square off within the session, trades are
sequential and rarely overlap, so the embargo buffer is the active guard, thinning
the pool near boundaries; with concurrent/overlapping labels the overlap purge also
bites. (Deterministic rules do no per-fold fitting, so full-coverage reconstructed
paths would be identical — hence the distribution is taken over the purged
``C(N,k)`` test-group combinations, not those degenerate paths.)
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import combinations

import numpy as np
import numpy.typing as npt

from lab.research.validation.sharpe import annualized_sharpe
from lab.research.validation.splitter import ONE_TRADING_DAY, label_overlaps


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
    def n_finite_paths(self) -> int:
        """Number of combinations that retained a scorable pool after purging."""
        return int(self._finite().size)

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


def _group_span(
    group: np.ndarray, entry_times: Sequence[datetime], exit_times: Sequence[datetime]
) -> tuple[datetime, datetime]:
    """The ``[min entry, max exit]`` time span covered by a group's observations."""
    start = min(entry_times[int(i)] for i in group)
    end = max(exit_times[int(i)] for i in group)
    return start, end


def combinatorial_purged_cv(
    returns: npt.ArrayLike,
    entry_times: Sequence[datetime],
    exit_times: Sequence[datetime],
    *,
    n_groups: int,
    k_test_groups: int,
    periods_per_year: float,
    embargo: timedelta = ONE_TRADING_DAY,
) -> CPCVResult:
    """Run purged, embargoed CPCV over a return series; return the path-Sharpe distribution.

    Args:
        returns: Per-trade net returns, time-ordered.
        entry_times: Label-window start (observation entry) per return.
        exit_times: Label-window end (observation exit) per return.
        n_groups: Number of groups ``N`` (>= 2).
        k_test_groups: Test groups per combination ``k`` (1 <= k < N).
        periods_per_year: Annualization factor (fixed Sharpe convention).
        embargo: Buffer applied at each test/train boundary (default one trading
            day); a test trade within this buffer of a train group is purged.
    """
    if n_groups < 2:
        raise ValueError(f"n_groups must be >= 2; got {n_groups}")
    if not 1 <= k_test_groups < n_groups:
        raise ValueError(f"k_test_groups must be in [1, n_groups); got {k_test_groups}")
    values = np.asarray(returns, dtype=np.float64)
    if not len(entry_times) == len(exit_times) == values.size:
        raise ValueError("returns, entry_times, and exit_times must have equal length")
    if values.size < n_groups:
        raise ValueError(f"need at least {n_groups} returns; got {values.size}")

    groups = np.array_split(np.arange(values.size), n_groups)
    path_sharpes: list[float] = []
    for combo in combinations(range(n_groups), k_test_groups):
        combo_set = set(combo)
        # Embargo-widened spans of the train groups this combination excludes.
        train_spans = [
            _group_span(groups[g], entry_times, exit_times)
            for g in range(n_groups)
            if g not in combo_set
        ]
        kept = [
            int(i)
            for g in combo
            for i in groups[g]
            if not any(
                label_overlaps(
                    entry_times[int(i)], exit_times[int(i)], start - embargo, end + embargo
                )
                for start, end in train_spans
            )
        ]
        pooled = values[np.array(kept, dtype=np.int64)] if kept else np.empty(0, dtype=np.float64)
        path_sharpes.append(annualized_sharpe(pooled, periods_per_year))
    n_paths = math.comb(n_groups, k_test_groups) * k_test_groups / n_groups
    return CPCVResult(tuple(path_sharpes), n_groups, k_test_groups, n_paths)
