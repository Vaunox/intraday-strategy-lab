"""Probability of Backtest Overfitting via CSCV (Phase 2, P2.2).

Combinatorially Symmetric Cross-Validation (Bailey et al.): given a matrix of
per-period performance for N candidate configurations (a strategy's parameter
variants), split time into S blocks; for every symmetric split into in-sample /
out-of-sample halves, pick the IS-best configuration and record how it ranks OOS.
PBO is the fraction of splits where the IS-best lands below the OOS median — i.e.
how often chasing the best backtest buys you nothing out of sample. Kill-gate
criterion 3 pins ``PBO < 0.20``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import numpy.typing as npt
from scipy import stats

FloatArray = npt.NDArray[np.float64]

#: Default number of CSCV blocks (must be even). C(16,8) = 12870 symmetric splits.
DEFAULT_N_SPLITS = 16


@dataclass(frozen=True, slots=True)
class PBOResult:
    """The PBO estimate and the underlying logit distribution."""

    pbo: float
    logits: FloatArray


def _sharpe_per_config(block: FloatArray) -> FloatArray:
    """Per-column (per-config) Sharpe over a block of period returns."""
    mean = block.mean(axis=0)
    std = block.std(axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        result: FloatArray = np.where(std > 0.0, mean / std, np.nan)
    return result


def probability_of_backtest_overfitting(
    performance_matrix: npt.ArrayLike, *, n_splits: int = DEFAULT_N_SPLITS
) -> PBOResult:
    """Estimate PBO from a ``(T periods, N configs)`` performance matrix.

    Args:
        performance_matrix: Per-period returns per configuration (columns are the
            strategy's variants).
        n_splits: Number of CSCV time blocks (even).
    """
    matrix = np.asarray(performance_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("performance_matrix must be 2-D (T periods x N configs)")
    n_periods, n_configs = matrix.shape
    if n_configs < 2:
        raise ValueError("PBO needs at least 2 configurations")
    if n_splits % 2 != 0:
        raise ValueError("n_splits must be even")
    if n_periods < n_splits:
        raise ValueError("need at least n_splits periods")

    blocks = np.array_split(np.arange(n_periods), n_splits)
    logits: list[float] = []
    for is_blocks in combinations(range(n_splits), n_splits // 2):
        is_set = set(is_blocks)
        is_idx = np.concatenate([blocks[b] for b in range(n_splits) if b in is_set])
        oos_idx = np.concatenate([blocks[b] for b in range(n_splits) if b not in is_set])

        is_perf = _sharpe_per_config(matrix[is_idx])
        oos_perf = _sharpe_per_config(matrix[oos_idx])
        if np.all(np.isnan(is_perf)):
            continue
        best = int(np.nanargmax(is_perf))
        # Rank the IS-best config out of sample (nan configs rank lowest).
        ranks = stats.rankdata(np.nan_to_num(oos_perf, nan=-np.inf))
        relative_rank = ranks[best] / (n_configs + 1)
        relative_rank = min(max(relative_rank, 1e-6), 1.0 - 1e-6)
        logits.append(math.log(relative_rank / (1.0 - relative_rank)))

    logit_array = np.array(logits, dtype=np.float64)
    pbo = float(np.mean(logit_array <= 0.0)) if logit_array.size else float("nan")
    return PBOResult(pbo=pbo, logits=logit_array)
