"""Tests for the resumable historical backfill (P1.3)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from lab.core.config import load_settings
from lab.core.nse_calendar import NseCalendar
from lab.core.types import BarInterval, Candle
from lab.data.ingest.backfill import Backfiller, BackfillPlan
from lab.data.store.parquet_archive import ParquetArchive

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config"
IST = ZoneInfo("Asia/Kolkata")


class FakeBackfillBroker:
    """Generates two 5-min candles per trading day within the requested range."""

    def __init__(self, calendar: NseCalendar) -> None:
        self._cal = calendar
        self.fetch_calls = 0

    def fetch_historical_candles(
        self, symbol: str, interval: BarInterval, start: datetime, end: datetime
    ) -> Sequence[Candle]:
        self.fetch_calls += 1
        candles: list[Candle] = []
        for day in self._cal.trading_days(start.date(), end.date()):
            for minute in (15, 20):
                ts = datetime(day.year, day.month, day.day, 9, minute, tzinfo=IST)
                if start <= ts <= end:
                    candles.append(Candle(symbol, interval, ts, 100.0, 101.0, 99.0, 100.5, 1000))
        return candles


def _calendar() -> NseCalendar:
    return NseCalendar.from_settings(load_settings("dev", config_dir=REPO_CONFIG, environ={}))


# A clean week with no holidays (Mon 2024-07-22 .. Fri 2024-07-26).
WEEK_START = date(2024, 7, 22)
WEEK_END = date(2024, 7, 26)


def test_full_backfill_writes_all_trading_days(tmp_path: Path) -> None:
    cal = _calendar()
    archive = ParquetArchive(tmp_path / "store")
    backfiller = Backfiller(FakeBackfillBroker(cal), archive, cal)
    report = backfiller.run(BackfillPlan(("RELIANCE",), BarInterval.MIN_5, WEEK_START, WEEK_END))

    assert report.fetched_days == 5
    assert report.skipped_days == 0
    assert report.candles_written == 10  # 5 days x 2 bars
    assert len(archive.stored_dates("RELIANCE", BarInterval.MIN_5)) == 5


def test_resume_skips_stored_without_duplication(tmp_path: Path) -> None:
    cal = _calendar()
    archive = ParquetArchive(tmp_path / "store")
    backfiller = Backfiller(FakeBackfillBroker(cal), archive, cal)
    plan = BackfillPlan(("RELIANCE",), BarInterval.MIN_5, WEEK_START, WEEK_END)

    backfiller.run(plan)
    second = backfiller.run(plan)  # resume — everything already stored

    assert second.fetched_days == 0
    assert second.skipped_days == 5
    assert second.candles_written == 0
    # No duplication: the stored candle count is unchanged.
    read = archive.read_candles(
        "RELIANCE",
        BarInterval.MIN_5,
        datetime(2024, 7, 22, 0, 0, tzinfo=IST),
        datetime(2024, 7, 26, 23, 59, tzinfo=IST),
    )
    assert len(read) == 10


def test_backfill_resumes_after_interruption(tmp_path: Path) -> None:
    cal = _calendar()
    archive = ParquetArchive(tmp_path / "store")
    backfiller = Backfiller(FakeBackfillBroker(cal), archive, cal)

    # Simulate an interrupted run that only got the first two days.
    backfiller.run(BackfillPlan(("RELIANCE",), BarInterval.MIN_5, WEEK_START, date(2024, 7, 23)))
    # Resume the full range: the two stored days are skipped, three are fetched.
    report = backfiller.run(BackfillPlan(("RELIANCE",), BarInterval.MIN_5, WEEK_START, WEEK_END))

    assert report.skipped_days == 2
    assert report.fetched_days == 3


def test_pagination_splits_long_ranges(tmp_path: Path) -> None:
    cal = _calendar()
    archive = ParquetArchive(tmp_path / "store")
    broker = FakeBackfillBroker(cal)
    backfiller = Backfiller(broker, archive, cal, max_window_days=2)
    backfiller.run(BackfillPlan(("RELIANCE",), BarInterval.MIN_5, WEEK_START, WEEK_END))

    # 5 missing days in windows of 2 -> [22,23], [24,25], [26] = 3 requests.
    assert broker.fetch_calls == 3


def test_backfill_skips_holidays(tmp_path: Path) -> None:
    cal = _calendar()
    archive = ParquetArchive(tmp_path / "store")
    backfiller = Backfiller(FakeBackfillBroker(cal), archive, cal)
    # 2024-07-17 is Muharram (holiday); range Mon 15 .. Thu 18.
    backfiller.run(
        BackfillPlan(("RELIANCE",), BarInterval.MIN_5, date(2024, 7, 15), date(2024, 7, 18))
    )

    stored = archive.stored_dates("RELIANCE", BarInterval.MIN_5)
    assert stored == [date(2024, 7, 15), date(2024, 7, 16), date(2024, 7, 18)]
