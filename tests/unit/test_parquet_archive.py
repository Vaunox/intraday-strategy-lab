"""Tests for the Parquet candle archive (P1.2)."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from lab.core.interfaces import Repository
from lab.core.types import BarInterval, Candle
from lab.data.store.parquet_archive import ParquetArchive, PartitionExistsError

IST = ZoneInfo("Asia/Kolkata")


def _candle(day: int, hour: int, minute: int, base: float = 100.0, oi: int | None = None) -> Candle:
    ts = datetime(2024, 7, day, hour, minute, tzinfo=IST)
    return Candle(
        "RELIANCE", BarInterval.MIN_5, ts, base, base + 2.0, base - 1.0, base + 1.0, 1000, oi
    )


def _archive(tmp_path: Path) -> ParquetArchive:
    return ParquetArchive(tmp_path / "store")


def test_raw_round_trip(tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    written = [_candle(15, 9, 15), _candle(15, 9, 20), _candle(16, 9, 15)]
    archive.write_candles("RELIANCE", BarInterval.MIN_5, written)

    read = archive.read_candles(
        "RELIANCE",
        BarInterval.MIN_5,
        datetime(2024, 7, 15, 0, 0, tzinfo=IST),
        datetime(2024, 7, 16, 23, 59, tzinfo=IST),
    )
    assert list(read) == sorted(written, key=lambda c: c.timestamp)


def test_open_interest_none_round_trips(tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    archive.write_candles("RELIANCE", BarInterval.MIN_5, [_candle(15, 9, 15, oi=None)])
    (read,) = archive.read_candles(
        "RELIANCE",
        BarInterval.MIN_5,
        datetime(2024, 7, 15, 0, 0, tzinfo=IST),
        datetime(2024, 7, 15, 23, 59, tzinfo=IST),
    )
    assert read.open_interest is None


def test_raw_is_immutable(tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    archive.write_candles("RELIANCE", BarInterval.MIN_5, [_candle(15, 9, 15)])
    with pytest.raises(PartitionExistsError):
        archive.write_candles("RELIANCE", BarInterval.MIN_5, [_candle(15, 9, 20)])


def test_read_filters_by_range(tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    archive.write_candles(
        "RELIANCE", BarInterval.MIN_5, [_candle(15, 9, 15), _candle(15, 9, 20), _candle(16, 9, 15)]
    )
    read = archive.read_candles(
        "RELIANCE",
        BarInterval.MIN_5,
        datetime(2024, 7, 15, 9, 18, tzinfo=IST),
        datetime(2024, 7, 15, 15, 30, tzinfo=IST),
    )
    assert [c.timestamp.day for c in read] == [15]
    assert [c.timestamp.minute for c in read] == [20]


def test_stored_dates_for_resume(tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    assert archive.stored_dates("RELIANCE", BarInterval.MIN_5) == []
    archive.write_candles("RELIANCE", BarInterval.MIN_5, [_candle(15, 9, 15), _candle(16, 9, 15)])
    assert archive.stored_dates("RELIANCE", BarInterval.MIN_5) == [
        date(2024, 7, 15),
        date(2024, 7, 16),
    ]


def test_adjusted_layer_is_overwritable(tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    archive.write_adjusted("RELIANCE", BarInterval.MIN_5, [_candle(15, 9, 15, base=100.0)])
    # Re-running adjustment overwrites the derived layer without error.
    archive.write_adjusted("RELIANCE", BarInterval.MIN_5, [_candle(15, 9, 15, base=50.0)])
    (read,) = archive.read_adjusted(
        "RELIANCE",
        BarInterval.MIN_5,
        datetime(2024, 7, 15, 0, 0, tzinfo=IST),
        datetime(2024, 7, 15, 23, 59, tzinfo=IST),
    )
    assert read.open == 50.0


def test_raw_and_adjusted_are_separate(tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    archive.write_candles("RELIANCE", BarInterval.MIN_5, [_candle(15, 9, 15, base=100.0)])
    archive.write_adjusted("RELIANCE", BarInterval.MIN_5, [_candle(15, 9, 15, base=50.0)])
    (raw,) = archive.read_candles(
        "RELIANCE",
        BarInterval.MIN_5,
        datetime(2024, 7, 15, 0, 0, tzinfo=IST),
        datetime(2024, 7, 15, 23, 59, tzinfo=IST),
    )
    assert raw.open == 100.0  # raw untouched by the adjusted write


def test_archive_satisfies_repository_protocol(tmp_path: Path) -> None:
    archive: Repository = _archive(tmp_path)
    assert isinstance(archive, Repository)
