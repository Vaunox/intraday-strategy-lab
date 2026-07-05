"""Authoritative NSE trading-calendar and session utility (Part III Layer 1).

Answers, entirely in IST: is a date a trading day? what are the session bounds?
when is intraday square-off? which session phase is a given instant in? The
timezone, session boundaries, square-off time, and holiday list are all
configuration (``config``), never literals in code (Part I §2) — so the exact
calendar used by any run is versioned and reproducible.

The holiday list is exchange data curated in ``config/default.yaml`` and extended
against official NSE circulars as the backfill range is fixed (Phase 1); weekends
are structural. All datetime inputs must be timezone-aware.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from lab.core.config import CalendarSettings, Settings
from lab.core.types import SessionPhase

# Weekly market closure: Saturday (5) and Sunday (6) in ``date.weekday()`` terms.
_SATURDAY = 5

# Defensive bound on trading-day search; real calendars never approach it.
_MAX_SCAN_DAYS = 366


class NseCalendar:
    """A trading calendar bound to a configured timezone, session, and holidays."""

    def __init__(self, settings: CalendarSettings) -> None:
        """Build the calendar from resolved :class:`CalendarSettings`."""
        self._settings = settings
        self._tz = ZoneInfo(settings.timezone)
        self._holidays: frozenset[date] = frozenset(settings.holidays)

    @classmethod
    def from_settings(cls, settings: Settings) -> NseCalendar:
        """Convenience constructor from the top-level :class:`Settings`."""
        return cls(settings.calendar)

    @property
    def timezone(self) -> ZoneInfo:
        """The IST timezone this calendar operates in."""
        return self._tz

    # -- day classification -------------------------------------------------- #
    def is_weekend(self, day: date) -> bool:
        """Return whether ``day`` falls on a Saturday or Sunday."""
        return day.weekday() >= _SATURDAY

    def is_holiday(self, day: date) -> bool:
        """Return whether ``day`` is a configured exchange holiday."""
        return day in self._holidays

    def is_trading_day(self, day: date) -> bool:
        """Return whether the exchange trades on ``day`` (not weekend, not holiday)."""
        return not self.is_weekend(day) and not self.is_holiday(day)

    def next_trading_day(self, day: date) -> date:
        """Return the first trading day strictly after ``day``."""
        candidate = day
        for _ in range(_MAX_SCAN_DAYS):
            candidate = candidate + timedelta(days=1)
            if self.is_trading_day(candidate):
                return candidate
        raise RuntimeError(f"no trading day found within {_MAX_SCAN_DAYS} days after {day}")

    def previous_trading_day(self, day: date) -> date:
        """Return the last trading day strictly before ``day``."""
        candidate = day
        for _ in range(_MAX_SCAN_DAYS):
            candidate = candidate - timedelta(days=1)
            if self.is_trading_day(candidate):
                return candidate
        raise RuntimeError(f"no trading day found within {_MAX_SCAN_DAYS} days before {day}")

    def trading_days(self, start: date, end: date) -> list[date]:
        """Return every trading day in the inclusive range ``[start, end]``."""
        if start > end:
            raise ValueError(f"start {start} must not be after end {end}")
        days: list[date] = []
        current = start
        while current <= end:
            if self.is_trading_day(current):
                days.append(current)
            current = current + timedelta(days=1)
        return days

    # -- intraday session ---------------------------------------------------- #
    def session_bounds(self, day: date) -> tuple[datetime, datetime]:
        """Return the regular-session open and close as IST-aware datetimes.

        Raises:
            ValueError: If ``day`` is not a trading day.
        """
        if not self.is_trading_day(day):
            raise ValueError(f"{day} is not a trading day; it has no session bounds")
        session = self._settings.session
        return (
            datetime.combine(day, session.open, tzinfo=self._tz),
            datetime.combine(day, session.close, tzinfo=self._tz),
        )

    def square_off_at(self, day: date) -> datetime:
        """Return the intraday square-off cutoff for ``day`` as an IST-aware datetime.

        Raises:
            ValueError: If ``day`` is not a trading day.
        """
        if not self.is_trading_day(day):
            raise ValueError(f"{day} is not a trading day; it has no square-off time")
        return datetime.combine(day, self._settings.session.square_off, tzinfo=self._tz)

    def phase_at(self, moment: datetime) -> SessionPhase:
        """Classify ``moment`` (any timezone) into a :class:`SessionPhase`.

        The instant is converted to IST before classification. Non-trading days,
        and any time outside the pre-open/regular/post-close windows (including
        the gap between the regular close and the post-close session), are
        ``CLOSED``.

        Raises:
            ValueError: If ``moment`` is timezone-naive.
        """
        if moment.tzinfo is None or moment.tzinfo.utcoffset(moment) is None:
            raise ValueError(f"moment must be timezone-aware; got naive {moment!r}")
        local = moment.astimezone(self._tz)
        if not self.is_trading_day(local.date()):
            return SessionPhase.CLOSED

        now = local.time()
        session = self._settings.session
        if session.pre_open_start <= now < session.open:
            return SessionPhase.PRE_OPEN
        if session.open <= now < session.close:
            return SessionPhase.REGULAR
        if session.post_close_start <= now < session.post_close_end:
            return SessionPhase.POST_CLOSE
        return SessionPhase.CLOSED

    def is_open(self, moment: datetime) -> bool:
        """Return whether the regular session is live at ``moment``."""
        return self.phase_at(moment) is SessionPhase.REGULAR

    def is_past_square_off(self, moment: datetime) -> bool:
        """Return whether ``moment`` is at/after the square-off cutoff on a trading day.

        Raises:
            ValueError: If ``moment`` is timezone-naive.
        """
        if moment.tzinfo is None or moment.tzinfo.utcoffset(moment) is None:
            raise ValueError(f"moment must be timezone-aware; got naive {moment!r}")
        local = moment.astimezone(self._tz)
        if not self.is_trading_day(local.date()):
            return False
        return local.time() >= self._settings.session.square_off
