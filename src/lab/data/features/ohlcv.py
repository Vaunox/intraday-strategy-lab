"""OHLCV array container for feature computation (Phase 1, P1.5).

Adapts an ordered ``Candle`` series into parallel NumPy arrays (the form TA-Lib
and vectorized math want), keeping timestamps for session-aware features (VWAP,
pivots, opening range). ``prefix`` yields the point-in-time view up to and
including a bar — the basis of the incremental path in the dual-path skew test.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import numpy.typing as npt

from lab.core.types import Candle

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class OHLCV:
    """Parallel OHLCV arrays plus per-bar timestamps, in ascending time order."""

    timestamps: tuple[datetime, ...]
    open: FloatArray
    high: FloatArray
    low: FloatArray
    close: FloatArray
    volume: FloatArray

    @classmethod
    def from_candles(cls, candles: Sequence[Candle]) -> OHLCV:
        """Build an :class:`OHLCV` from an ascending ``Candle`` series."""
        return cls(
            timestamps=tuple(c.timestamp for c in candles),
            open=np.array([c.open for c in candles], dtype=np.float64),
            high=np.array([c.high for c in candles], dtype=np.float64),
            low=np.array([c.low for c in candles], dtype=np.float64),
            close=np.array([c.close for c in candles], dtype=np.float64),
            volume=np.array([float(c.volume) for c in candles], dtype=np.float64),
        )

    def __len__(self) -> int:
        """Return the number of bars."""
        return len(self.timestamps)

    def prefix(self, count: int) -> OHLCV:
        """Return the point-in-time view of the first ``count`` bars."""
        return OHLCV(
            timestamps=self.timestamps[:count],
            open=self.open[:count],
            high=self.high[:count],
            low=self.low[:count],
            close=self.close[:count],
            volume=self.volume[:count],
        )
