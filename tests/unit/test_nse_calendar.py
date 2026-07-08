"""Tests for the NSE trading calendar: day classification, session phases,
bounds, square-off, and timezone conversion — all IST.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from lab.core.config import load_settings
from lab.core.nse_calendar import NseCalendar
from lab.core.types import SessionPhase

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config"
IST = ZoneInfo("Asia/Kolkata")


@pytest.fixture(scope="module")
def calendar() -> NseCalendar:
    settings = load_settings("dev", config_dir=REPO_CONFIG, environ={})
    return NseCalendar.from_settings(settings)


def _ist(year: int, month: int, day: int, hour: int, minute: int) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=IST)


def test_normal_weekday_is_trading_day(calendar: NseCalendar) -> None:
    assert calendar.is_trading_day(date(2024, 7, 15))  # Monday
    assert calendar.is_trading_day(date(2025, 1, 1))  # New Year is a trading day in India


@pytest.mark.parametrize(
    "holiday",
    [date(2024, 1, 26), date(2024, 3, 29), date(2024, 8, 15), date(2024, 12, 25)],
)
def test_configured_holidays_are_not_trading_days(calendar: NseCalendar, holiday: date) -> None:
    assert calendar.is_holiday(holiday)
    assert not calendar.is_trading_day(holiday)


@pytest.mark.parametrize(
    "holiday",
    [date(2018, 8, 15), date(2021, 1, 26), date(2023, 12, 25), date(2026, 1, 26)],
)
def test_expanded_range_holidays_are_covered(calendar: NseCalendar, holiday: date) -> None:
    # The holiday list now spans 2018-2026 (generated from exchange_calendars XBOM).
    assert not calendar.is_trading_day(holiday)


def test_weekends_are_not_trading_days(calendar: NseCalendar) -> None:
    assert not calendar.is_trading_day(date(2024, 8, 17))  # Saturday
    assert not calendar.is_trading_day(date(2024, 8, 18))  # Sunday


def test_next_and_previous_trading_day_skip_holiday_and_weekend(calendar: NseCalendar) -> None:
    # 2024-08-15 (Thu) is Independence Day.
    assert calendar.next_trading_day(date(2024, 8, 15)) == date(2024, 8, 16)
    assert calendar.previous_trading_day(date(2024, 8, 15)) == date(2024, 8, 14)
    # From Fri 2024-08-16, the next trading day is Mon 2024-08-19 (skip weekend).
    assert calendar.next_trading_day(date(2024, 8, 16)) == date(2024, 8, 19)


def test_trading_days_range_excludes_closures(calendar: NseCalendar) -> None:
    days = calendar.trading_days(date(2024, 8, 14), date(2024, 8, 19))
    assert days == [date(2024, 8, 14), date(2024, 8, 16), date(2024, 8, 19)]


def test_trading_days_rejects_inverted_range(calendar: NseCalendar) -> None:
    with pytest.raises(ValueError, match="must not be after"):
        calendar.trading_days(date(2024, 8, 19), date(2024, 8, 14))


@pytest.mark.parametrize(
    ("moment", "expected"),
    [
        (_ist(2024, 7, 15, 8, 0), SessionPhase.CLOSED),
        (_ist(2024, 7, 15, 9, 5), SessionPhase.PRE_OPEN),
        (_ist(2024, 7, 15, 10, 0), SessionPhase.REGULAR),
        (_ist(2024, 7, 15, 15, 35), SessionPhase.CLOSED),  # gap before post-close
        (_ist(2024, 7, 15, 15, 45), SessionPhase.POST_CLOSE),
        (_ist(2024, 8, 15, 10, 0), SessionPhase.CLOSED),  # holiday
    ],
)
def test_phase_at(calendar: NseCalendar, moment: datetime, expected: SessionPhase) -> None:
    assert calendar.phase_at(moment) is expected


def test_phase_at_converts_from_other_timezone(calendar: NseCalendar) -> None:
    # 04:30 UTC on a trading day == 10:00 IST -> REGULAR.
    utc_moment = datetime(2024, 7, 15, 4, 30, tzinfo=UTC)
    assert calendar.phase_at(utc_moment) is SessionPhase.REGULAR


def test_phase_at_rejects_naive_datetime(calendar: NseCalendar) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        calendar.phase_at(datetime(2024, 7, 15, 10, 0))


def test_session_bounds(calendar: NseCalendar) -> None:
    open_dt, close_dt = calendar.session_bounds(date(2024, 7, 15))
    assert open_dt == _ist(2024, 7, 15, 9, 15)
    assert close_dt == _ist(2024, 7, 15, 15, 30)


def test_session_bounds_rejects_non_trading_day(calendar: NseCalendar) -> None:
    with pytest.raises(ValueError, match="not a trading day"):
        calendar.session_bounds(date(2024, 8, 15))


def test_square_off_and_past_square_off(calendar: NseCalendar) -> None:
    assert calendar.square_off_at(date(2024, 7, 15)).time() == time(15, 20)
    assert calendar.is_past_square_off(_ist(2024, 7, 15, 15, 25))
    assert not calendar.is_past_square_off(_ist(2024, 7, 15, 15, 0))
    # Non-trading day is never "past square-off".
    assert not calendar.is_past_square_off(_ist(2024, 8, 15, 15, 25))


def test_is_open(calendar: NseCalendar) -> None:
    assert calendar.is_open(_ist(2024, 7, 15, 10, 0))
    assert not calendar.is_open(_ist(2024, 7, 15, 16, 30))


def test_is_regular_session_time(calendar: NseCalendar) -> None:
    # Pure time-of-day window [09:15, 15:30) — the intraday-grid filter.
    assert calendar.is_regular_session_time(_ist(2024, 7, 15, 10, 0))
    assert calendar.is_regular_session_time(_ist(2024, 7, 15, 9, 15))  # open inclusive
    assert calendar.is_regular_session_time(_ist(2024, 7, 15, 15, 25))  # last 5-min bar
    assert not calendar.is_regular_session_time(_ist(2024, 7, 15, 9, 0))  # pre-open
    assert not calendar.is_regular_session_time(_ist(2024, 7, 15, 15, 30))  # close exclusive
    assert not calendar.is_regular_session_time(_ist(2024, 7, 15, 15, 40))  # post-close
    assert not calendar.is_regular_session_time(_ist(2024, 11, 1, 18, 30))  # Muhurat evening


def test_is_regular_session_time_rejects_naive(calendar: NseCalendar) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        calendar.is_regular_session_time(datetime(2024, 7, 15, 10, 0))  # naive
