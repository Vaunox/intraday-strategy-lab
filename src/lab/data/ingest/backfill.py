"""Resumable historical backfill (Phase 1, P1.3).

Pulls historical candles for a universe over a date range and writes them to the
immutable raw archive, one partition per IST trading day. It is:

* **calendar-aware** — only trading days are fetched (weekends/holidays skipped);
* **paginated** — consecutive missing days are fetched in bounded windows, so
  long ranges respect Kite's per-request window limit;
* **resumable & idempotent** — already-stored days are skipped, so an interrupted
  run resumes without re-fetching or duplicating (raw immutability guarantees no
  silent overwrite).

Depends on the ``BrokerAdapter`` Protocol; the concrete store is the partitioned
``ParquetArchive`` (whose ``stored_dates`` drives resume).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date

from lab.core.interfaces import BrokerAdapter
from lab.core.logging import get_logger
from lab.core.nse_calendar import NseCalendar
from lab.core.types import BarInterval
from lab.data.store.parquet_archive import ParquetArchive

_log = get_logger("data.ingest.backfill")

#: Conservative default request window (trading days). Kite minute history is
#: served in windows of roughly this size; larger ranges are split.
DEFAULT_MAX_WINDOW_DAYS = 60


@dataclass(frozen=True, slots=True)
class BackfillPlan:
    """What to backfill: a universe, an interval, and an inclusive date range."""

    symbols: tuple[str, ...]
    interval: BarInterval
    start: date
    end: date


@dataclass(frozen=True, slots=True)
class BackfillReport:
    """Outcome of a backfill run."""

    symbols: int
    fetched_days: int
    skipped_days: int
    candles_written: int


class Backfiller:
    """Fetches historical candles per plan and writes them to the raw archive."""

    def __init__(
        self,
        adapter: BrokerAdapter,
        archive: ParquetArchive,
        calendar: NseCalendar,
        *,
        max_window_days: int = DEFAULT_MAX_WINDOW_DAYS,
    ) -> None:
        """Wire the backfiller to a broker, a store, and a trading calendar."""
        self._adapter = adapter
        self._archive = archive
        self._calendar = calendar
        self._max_window_days = max_window_days

    @staticmethod
    def _windows(
        trading_days: list[date], stored: set[date], max_window: int
    ) -> Iterator[list[date]]:
        """Yield runs of consecutive missing trading days, each <= ``max_window``."""
        run: list[date] = []
        for day in trading_days:
            if day in stored:
                if run:
                    yield run
                    run = []
                continue
            run.append(day)
            if len(run) >= max_window:
                yield run
                run = []
        if run:
            yield run

    def run(self, plan: BackfillPlan) -> BackfillReport:
        """Execute ``plan``, skipping already-stored days; return a report."""
        fetched_days = skipped_days = candles_written = 0
        for symbol in plan.symbols:
            trading_days = self._calendar.trading_days(plan.start, plan.end)
            stored = set(self._archive.stored_dates(symbol, plan.interval))
            skipped_days += sum(1 for day in trading_days if day in stored)

            for window in self._windows(trading_days, stored, self._max_window_days):
                window_start = self._calendar.session_bounds(window[0])[0]
                window_end = self._calendar.session_bounds(window[-1])[1]
                candles = self._adapter.fetch_historical_candles(
                    symbol, plan.interval, window_start, window_end
                )
                if candles:
                    self._archive.write_candles(symbol, plan.interval, candles)
                fetched_days += len(window)
                candles_written += len(candles)

            _log.info(
                "backfill_symbol_done",
                symbol=symbol,
                interval=plan.interval.value,
                stored_before=len(stored),
            )

        report = BackfillReport(
            symbols=len(plan.symbols),
            fetched_days=fetched_days,
            skipped_days=skipped_days,
            candles_written=candles_written,
        )
        _log.info(
            "backfill_done",
            symbols=report.symbols,
            fetched_days=report.fetched_days,
            skipped_days=report.skipped_days,
            candles_written=report.candles_written,
        )
        return report
