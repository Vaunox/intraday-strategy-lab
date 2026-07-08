"""Domain value types shared across every layer.

All types are immutable (frozen dataclasses) and validate their invariants at
construction — inputs are checked at the boundary and bad data fails loudly and
early (Part I §7). Timestamps are timezone-aware; the data layer guarantees they
are in IST.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class Side(Enum):
    """Direction of a position or trade."""

    LONG = "long"
    SHORT = "short"


class SessionPhase(Enum):
    """Phase of the NSE equity trading day at a given instant (all IST).

    ``CLOSED`` covers non-trading days and any time outside the defined sessions,
    including the gap between the regular close and the post-close session.
    """

    CLOSED = "closed"
    PRE_OPEN = "pre_open"
    REGULAR = "regular"
    POST_CLOSE = "post_close"


class BarInterval(Enum):
    """Candle interval.

    Values mirror Kite Connect's ``interval`` strings so the broker adapter
    (Phase 1) can map them without a second lookup table.
    """

    MINUTE = "minute"
    MIN_3 = "3minute"
    MIN_5 = "5minute"
    MIN_15 = "15minute"
    MIN_60 = "60minute"
    DAY = "day"


class Verdict(Enum):
    """Outcome of the seven-point kill-gate for a strategy study."""

    PASS = "pass"  # noqa: S105 — kill-gate verdict label, not a credential
    KILL = "kill"
    #: A criterion's input arrived under-shaped (a stub, not a genuinely computed
    #: value); the gate refuses to certify. Neither PASS nor KILL (Phase-3 B-1).
    INSUFFICIENT = "insufficient"


def _require_nonempty(value: str, field: str) -> None:
    """Raise ``ValueError`` if ``value`` is empty or whitespace-only."""
    if not value or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")


def _require_tz_aware(moment: datetime, field: str) -> None:
    """Raise ``ValueError`` if ``moment`` is naive (no usable UTC offset)."""
    if moment.tzinfo is None or moment.tzinfo.utcoffset(moment) is None:
        raise ValueError(f"{field} must be timezone-aware (IST); got naive {moment!r}")


def _require_positive_price(value: float, field: str) -> None:
    """Raise ``ValueError`` if ``value`` is not a finite, strictly positive price."""
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{field} must be a finite positive price; got {value!r}")


@dataclass(frozen=True, slots=True)
class Candle:
    """A single OHLCV(+OI) price bar for one symbol at one interval.

    Also referred to as a "bar". This is the fundamental unit of market data
    the whole program is built on. ``timestamp`` is the timezone-aware start of
    the bar (IST).

    Invariants (validated): prices are finite and positive; ``high`` is the
    period maximum and ``low`` the period minimum; ``volume`` and
    ``open_interest`` are non-negative.
    """

    symbol: str
    interval: BarInterval
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    open_interest: int | None = None

    def __post_init__(self) -> None:
        """Validate the bar's invariants; raise ``ValueError`` on violation."""
        _require_nonempty(self.symbol, "symbol")
        _require_tz_aware(self.timestamp, "timestamp")
        for field_name, price in (
            ("open", self.open),
            ("high", self.high),
            ("low", self.low),
            ("close", self.close),
        ):
            _require_positive_price(price, field_name)
        if self.high < self.low:
            raise ValueError(f"high ({self.high}) must be >= low ({self.low})")
        if self.high < max(self.open, self.close):
            raise ValueError(f"high ({self.high}) must be >= open/close")
        if self.low > min(self.open, self.close):
            raise ValueError(f"low ({self.low}) must be <= open/close")
        if self.volume < 0:
            raise ValueError(f"volume must be non-negative; got {self.volume}")
        if self.open_interest is not None and self.open_interest < 0:
            raise ValueError(f"open_interest must be non-negative; got {self.open_interest}")


@dataclass(frozen=True, slots=True)
class StrategySignal:
    """A decision emitted by a strategy at a decision time.

    Point-in-time: ``asof`` is bar *t*'s close (the decision instant); execution
    happens at bar *t+1*'s open (Inviolable Rule 2). ``strength`` is a unit-scaled
    conviction/target-weight in ``[0, 1]``; direction is carried by ``side``.
    """

    asof: datetime
    symbol: str
    side: Side
    strength: float = 1.0
    reason: str | None = None

    def __post_init__(self) -> None:
        """Validate the signal; raise ``ValueError`` on violation."""
        _require_nonempty(self.symbol, "symbol")
        _require_tz_aware(self.asof, "asof")
        if not math.isfinite(self.strength) or not (0.0 <= self.strength <= 1.0):
            raise ValueError(f"strength must be within [0, 1]; got {self.strength!r}")
