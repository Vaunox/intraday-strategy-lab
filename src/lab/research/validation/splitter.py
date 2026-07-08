"""Purged k-fold cross-validation with embargo (Phase 2, P2.1).

Standard k-fold leaks across the train/test boundary when observations have
overlapping evaluation windows (López de Prado). Purging removes training
observations whose label window overlaps the test window; the embargo drops a
further buffer *after* each test fold. Because every intraday position squares
off within one session, a **1-trading-day embargo** fully removes overlap leakage
(pinned, not implicit — Part III Layer 2).

Each observation is an interval ``[entry_time, exit_time]`` (feature observed at
entry, label realized at exit). Observations are assumed time-ordered.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

#: Pinned embargo: one trading day (intraday holding horizon <= 1 session).
ONE_TRADING_DAY = timedelta(days=1)


def label_overlaps(
    entry: datetime, exit_: datetime, span_start: datetime, span_end: datetime
) -> bool:
    """Whether label window ``[entry, exit_]`` intersects ``[span_start, span_end]``.

    The single shared overlap primitive used by both the purged k-fold splitter
    and the combinatorial purged CV, so their purge semantics cannot drift. To
    apply an embargo, pass a span already widened by the embargo on both ends.
    """
    return entry <= span_end and exit_ >= span_start


def purge_indices(
    candidate_idx: Iterable[int],
    entry_times: Sequence[datetime],
    exit_times: Sequence[datetime],
    exclude_spans: Iterable[tuple[datetime, datetime]],
    *,
    embargo: timedelta,
) -> list[int]:
    """Return the candidate observations whose label window survives purging.

    A candidate index is kept unless its ``[entry, exit]`` window overlaps any
    ``exclude_span`` widened by ``embargo`` on both ends. This is the single purge
    primitive shared by the purged k-fold splitter, CPCV, and CSCV/PBO, so their
    purge/embargo semantics cannot drift (Part III Layer 2) — the one place the
    overlap rule lives.
    """
    widened = [(start - embargo, end + embargo) for start, end in exclude_spans]
    return [
        int(i)
        for i in candidate_idx
        if not any(
            label_overlaps(entry_times[int(i)], exit_times[int(i)], lo, hi) for lo, hi in widened
        )
    ]


@dataclass(frozen=True, slots=True)
class Fold:
    """One train/test split (index positions into the observation series)."""

    train: tuple[int, ...]
    test: tuple[int, ...]


class PurgedKFold:
    """Time-ordered k-fold splitter that purges overlap and embargoes a buffer."""

    def __init__(self, n_splits: int, embargo: timedelta = ONE_TRADING_DAY) -> None:
        """Configure the splitter.

        Args:
            n_splits: Number of folds (>= 2).
            embargo: Buffer after each test fold in which training observations
                are dropped (default: one trading day).
        """
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2; got {n_splits}")
        self._n_splits = n_splits
        self._embargo = embargo

    def split(self, entry_times: Sequence[datetime], exit_times: Sequence[datetime]) -> list[Fold]:
        """Return the purged, embargoed train/test folds for the observations."""
        if len(entry_times) != len(exit_times):
            raise ValueError("entry_times and exit_times must have equal length")
        n = len(entry_times)
        if n < self._n_splits:
            raise ValueError(f"cannot make {self._n_splits} folds from {n} observations")

        folds: list[Fold] = []
        for chunk in np.array_split(np.arange(n), self._n_splits):
            test = {int(i) for i in chunk}
            test_start = min(entry_times[i] for i in test)
            test_end = max(exit_times[i] for i in test)
            embargo_end = test_end + self._embargo

            train: list[int] = []
            for i in range(n):
                if i in test:
                    continue
                if label_overlaps(entry_times[i], exit_times[i], test_start, test_end):
                    continue  # purge: label window overlaps the test window
                if test_end < entry_times[i] <= embargo_end:
                    continue  # embargo: enters within the buffer after the test fold
                train.append(i)
            folds.append(Fold(train=tuple(train), test=tuple(sorted(test))))
        return folds
