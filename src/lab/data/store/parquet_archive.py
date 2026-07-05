"""Parquet-backed candle storage (Phase 1, P1.2).

The concrete ``Repository``: an immutable **raw** layer plus a derived
**adjusted** layer, partitioned by symbol / interval / IST trading date. This is
the ONLY module that imports the storage engine (pyarrow) — everything else
depends on the ``Repository`` Protocol (Part I §1, Part III Layer 1).

Raw immutability: a raw partition is never silently overwritten. Writing a date
that already exists raises :class:`PartitionExistsError`; corrections go to a new
version via the adjusted layer, which is regenerated from raw + corporate actions
(P1.4) and is therefore overwritable. Partition writes are atomic (temp file +
replace), so a partition is either fully present or absent — which is what makes
the backfill (P1.3) safely resumable.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pyarrow as pa
import pyarrow.parquet as pq

from lab.core.constants import INDIA_TZ
from lab.core.logging import get_logger
from lab.core.types import BarInterval, Candle

_log = get_logger("data.store.parquet")

_RAW = "raw"
_ADJUSTED = "adjusted"


class PartitionExistsError(FileExistsError):
    """Raised when writing a raw partition that already exists (raw is immutable)."""


class ParquetArchive:
    """Immutable raw + derived adjusted candle storage on local Parquet files."""

    def __init__(self, root: Path, *, timezone: str = INDIA_TZ) -> None:
        """Bind the archive to ``root`` and the IST trading-date timezone."""
        self._root = root
        self._tz = ZoneInfo(timezone)

    # -- paths --------------------------------------------------------------- #
    def _partition_path(self, layer: str, symbol: str, interval: BarInterval, day: date) -> Path:
        return (
            self._root
            / layer
            / f"symbol={symbol}"
            / f"interval={interval.value}"
            / f"{day.isoformat()}.parquet"
        )

    def _trading_date(self, moment: datetime) -> date:
        return moment.astimezone(self._tz).date()

    # -- serialization ------------------------------------------------------- #
    def _table(self, candles: Sequence[Candle]) -> pa.Table:
        return pa.table(
            {
                "timestamp": pa.array(
                    [c.timestamp for c in candles], type=pa.timestamp("us", tz=str(self._tz))
                ),
                "open": pa.array([c.open for c in candles], type=pa.float64()),
                "high": pa.array([c.high for c in candles], type=pa.float64()),
                "low": pa.array([c.low for c in candles], type=pa.float64()),
                "close": pa.array([c.close for c in candles], type=pa.float64()),
                "volume": pa.array([c.volume for c in candles], type=pa.int64()),
                "open_interest": pa.array([c.open_interest for c in candles], type=pa.int64()),
            }
        )

    def _write_partition(
        self, path: Path, symbol: str, interval: BarInterval, candles: Sequence[Candle]
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        table = self._table(sorted(candles, key=lambda c: c.timestamp))
        tmp = path.with_suffix(".parquet.tmp")
        pq.write_table(table, tmp)
        tmp.replace(path)  # atomic: partition is fully present or absent

    def _read_partition(self, path: Path, symbol: str, interval: BarInterval) -> list[Candle]:
        table = pq.read_table(path)
        columns = table.to_pydict()
        candles: list[Candle] = []
        for i in range(table.num_rows):
            oi = columns["open_interest"][i]
            candles.append(
                Candle(
                    symbol=symbol,
                    interval=interval,
                    timestamp=columns["timestamp"][i].astimezone(self._tz),
                    open=float(columns["open"][i]),
                    high=float(columns["high"][i]),
                    low=float(columns["low"][i]),
                    close=float(columns["close"][i]),
                    volume=int(columns["volume"][i]),
                    open_interest=int(oi) if oi is not None else None,
                )
            )
        return candles

    # -- raw layer (immutable, Repository contract) -------------------------- #
    def write_candles(
        self,
        symbol: str,
        interval: BarInterval,
        candles: Sequence[Candle],
        *,
        overwrite: bool = False,
    ) -> None:
        """Persist ``candles`` to the raw layer, one partition per IST trading date.

        Raises:
            PartitionExistsError: If a target raw partition already exists and
                ``overwrite`` is False (raw is append-only / immutable).
        """
        self._write_layer(_RAW, symbol, interval, candles, overwrite=overwrite, immutable=True)

    def read_candles(
        self, symbol: str, interval: BarInterval, start: datetime, end: datetime
    ) -> Sequence[Candle]:
        """Return raw candles for ``symbol``/``interval`` within ``[start, end]``."""
        return self._read_layer(_RAW, symbol, interval, start, end)

    # -- adjusted layer (derived, regenerable) ------------------------------- #
    def write_adjusted(self, symbol: str, interval: BarInterval, candles: Sequence[Candle]) -> None:
        """Write corp-action-adjusted candles to the derived layer (overwritable)."""
        self._write_layer(_ADJUSTED, symbol, interval, candles, overwrite=True, immutable=False)

    def read_adjusted(
        self, symbol: str, interval: BarInterval, start: datetime, end: datetime
    ) -> Sequence[Candle]:
        """Return adjusted candles for ``symbol``/``interval`` within ``[start, end]``."""
        return self._read_layer(_ADJUSTED, symbol, interval, start, end)

    # -- shared layer helpers ------------------------------------------------ #
    def _write_layer(
        self,
        layer: str,
        symbol: str,
        interval: BarInterval,
        candles: Sequence[Candle],
        *,
        overwrite: bool,
        immutable: bool,
    ) -> None:
        by_date: dict[date, list[Candle]] = defaultdict(list)
        for candle in candles:
            by_date[self._trading_date(candle.timestamp)].append(candle)
        for day, day_candles in by_date.items():
            path = self._partition_path(layer, symbol, interval, day)
            if path.exists() and immutable and not overwrite:
                raise PartitionExistsError(f"raw partition already exists (immutable): {path}")
            self._write_partition(path, symbol, interval, day_candles)
        _log.info(
            "candles_written",
            layer=layer,
            symbol=symbol,
            interval=interval.value,
            partitions=len(by_date),
            candles=len(candles),
        )

    def _read_layer(
        self, layer: str, symbol: str, interval: BarInterval, start: datetime, end: datetime
    ) -> list[Candle]:
        candles: list[Candle] = []
        for day in self.stored_dates(symbol, interval, layer=layer):
            if start.astimezone(self._tz).date() <= day <= end.astimezone(self._tz).date():
                path = self._partition_path(layer, symbol, interval, day)
                candles.extend(self._read_partition(path, symbol, interval))
        in_range = [c for c in candles if start <= c.timestamp <= end]
        return sorted(in_range, key=lambda c: c.timestamp)

    def stored_dates(self, symbol: str, interval: BarInterval, *, layer: str = _RAW) -> list[date]:
        """Return the IST trading dates already stored for ``symbol``/``interval``.

        Used by the backfill (P1.3) to resume without re-fetching or duplicating
        completed partitions.
        """
        directory = self._partition_path(layer, symbol, interval, date(2000, 1, 1)).parent
        if not directory.exists():
            return []
        return sorted(date.fromisoformat(p.stem) for p in directory.glob("*.parquet"))
