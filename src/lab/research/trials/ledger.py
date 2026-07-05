"""Honest, program-wide effective-trial ledger (Phase 2, P2.3).

Persists **every trial's realized return stream** to durable storage (one file
per trial), across all sessions and phases — never an integer counter
(Inviolable Rule 4). The Deflated Sharpe is deflated by the **effective** number
of independent trials, obtained by the correlation participation ratio of the
stored streams:

    effective_N = (trace C)^2 / sum(C_ij^2) = N^2 / ||C||_F^2

For N perfectly-correlated variants (a one-at-a-time parameter sweep) this is ~1;
for N independent strategies it is ~N; a mix lands in between. This is the same
cluster-adjusted effective-trial-count pattern as López de Prado's
covariance/clustering treatment. No caller ever passes a literal N — the DSR
pulls it from :meth:`TrialLedger.effective_trials`.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from lab.research.validation.metrics import deflated_sharpe_ratio
from lab.research.validation.sharpe import per_period_sharpe

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class TrialRecord:
    """One logged trial: its identity, parameters, and realized return stream."""

    trial_id: str
    strategy: str
    params: Mapping[str, Any]
    returns: tuple[float, ...]


class TrialLedger:
    """Append-only store of per-trial return streams feeding the DSR's effective-N."""

    def __init__(self, storage_dir: Path) -> None:
        """Bind the ledger to a durable directory (persists across sessions)."""
        self._dir = storage_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def log_trial(
        self,
        strategy: str,
        params: Mapping[str, Any],
        returns: Sequence[float],
        *,
        trial_id: str | None = None,
    ) -> str:
        """Append a trial's return stream to the ledger; return its id."""
        tid = trial_id or uuid.uuid4().hex
        payload = {
            "trial_id": tid,
            "strategy": strategy,
            "params": dict(params),
            "returns": [float(r) for r in returns],
        }
        (self._dir / f"{tid}.json").write_text(json.dumps(payload), encoding="utf-8")
        return tid

    def trials(self) -> list[TrialRecord]:
        """Load all logged trials (order is by trial-id filename)."""
        records: list[TrialRecord] = []
        for path in sorted(self._dir.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            records.append(
                TrialRecord(
                    trial_id=str(data["trial_id"]),
                    strategy=str(data["strategy"]),
                    params=dict(data["params"]),
                    returns=tuple(float(r) for r in data["returns"]),
                )
            )
        return records

    def count(self) -> int:
        """Return the RAW number of logged trials (not the effective count)."""
        return sum(1 for _ in self._dir.glob("*.json"))

    def _stream_matrix(self) -> FloatArray:
        streams = [np.asarray(r.returns, dtype=np.float64) for r in self.trials()]
        streams = [s for s in streams if s.size > 0]
        if not streams:
            return np.empty((0, 0), dtype=np.float64)
        length = max(s.size for s in streams)
        return np.vstack([np.pad(s, (0, length - s.size)) for s in streams])

    def effective_trials(self) -> float:
        """Return the effective (cluster-adjusted) number of independent trials."""
        matrix = self._stream_matrix()
        n = int(matrix.shape[0])
        if n <= 1:
            return float(n)
        corr: FloatArray = np.asarray(np.corrcoef(matrix), dtype=np.float64)
        np.nan_to_num(corr, copy=False, nan=0.0)  # constant streams -> no correlation
        np.fill_diagonal(corr, 1.0)
        frobenius_sq = float(np.sum(corr**2))
        return float(n * n / frobenius_sq) if frobenius_sq > 0 else float(n)

    def trial_sharpe_std(self) -> float:
        """Standard deviation of per-period Sharpes across trials (for the DSR)."""
        sharpes = [per_period_sharpe(r.returns) for r in self.trials()]
        finite = np.array([s for s in sharpes if np.isfinite(s)], dtype=np.float64)
        return float(finite.std(ddof=1)) if finite.size >= 2 else 0.0

    def deflated_sharpe(
        self, observed_sharpe: float, n: int, skew: float, kurtosis: float
    ) -> float:
        """Deflated Sharpe for a strategy, using THIS ledger's effective-N.

        The effective trial count and trial-Sharpe dispersion come from the
        ledger — callers never pass a literal N (Inviolable Rule 4).
        """
        return deflated_sharpe_ratio(
            observed_sharpe,
            n,
            skew,
            kurtosis,
            effective_trials=self.effective_trials(),
            trial_sharpe_std=self.trial_sharpe_std(),
        )
