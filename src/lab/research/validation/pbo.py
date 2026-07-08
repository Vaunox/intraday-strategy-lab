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
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import combinations

import numpy as np
import numpy.typing as npt
from scipy import stats

from lab.research.validation.splitter import ONE_TRADING_DAY, purge_indices

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
    performance_matrix: npt.ArrayLike,
    entry_times: Sequence[datetime],
    exit_times: Sequence[datetime],
    *,
    n_splits: int = DEFAULT_N_SPLITS,
    embargo: timedelta = ONE_TRADING_DAY,
) -> PBOResult:
    """Estimate PBO from a ``(T periods, N configs)`` performance matrix.

    The matrix rows are time-aligned periods (the SAME period across every config)
    and ``entry_times``/``exit_times`` are each row's label window, so CSCV's
    in-sample/out-of-sample blocks are coherent time partitions. Each split purges
    the OOS rows whose label window overlaps an in-sample block span (embargo-widened)
    through the shared :func:`~lab.research.validation.splitter.purge_indices`
    primitive — the same one CPCV uses — so the two cannot drift.

    Args:
        performance_matrix: Per-period returns per configuration (columns are the
            strategy's variants; rows are aligned periods).
        entry_times: Label-window start per row (length == number of periods).
        exit_times: Label-window end per row.
        n_splits: Number of CSCV time blocks (even).
        embargo: Buffer applied at each IS/OOS boundary (default one trading day).
    """
    matrix = np.asarray(performance_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("performance_matrix must be 2-D (T periods x N configs)")
    n_periods, n_configs = matrix.shape
    if n_configs < 2:
        raise ValueError("PBO needs at least 2 configurations")
    if not len(entry_times) == len(exit_times) == n_periods:
        raise ValueError("entry_times and exit_times must have one entry per matrix row")
    if n_splits % 2 != 0:
        raise ValueError("n_splits must be even")
    if n_periods < n_splits:
        raise ValueError("need at least n_splits periods")

    blocks = np.array_split(np.arange(n_periods), n_splits)
    block_spans = [
        (min(entry_times[int(i)] for i in block), max(exit_times[int(i)] for i in block))
        for block in blocks
    ]
    logits: list[float] = []
    for is_blocks in combinations(range(n_splits), n_splits // 2):
        is_set = set(is_blocks)
        is_idx = np.concatenate([blocks[b] for b in range(n_splits) if b in is_set])
        oos_all = np.concatenate([blocks[b] for b in range(n_splits) if b not in is_set])
        # Purge OOS rows overlapping an in-sample block span (embargoed), via the
        # shared primitive so CSCV and CPCV purge identically.
        oos_kept = purge_indices(
            (int(i) for i in oos_all),
            entry_times,
            exit_times,
            [block_spans[b] for b in is_blocks],
            embargo=embargo,
        )
        if len(oos_kept) < 2:
            continue  # a 0/1-row OOS (after purging) cannot yield a Sharpe
        oos_idx = np.array(oos_kept, dtype=np.int64)

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
