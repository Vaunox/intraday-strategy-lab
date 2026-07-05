"""Liquidity screening and restricted-list exclusion (Phase 1, P1.4).

The universe is restricted to liquid names and excludes surveillance segments,
so studies are not run on untradeable or manipulation-prone symbols (Part III
Layer 1). With OHLCV-only data, liquidity is proxied by average daily turnover
(price x volume); spread is unavailable and is out of scope. ESM (Enhanced
Surveillance Measure) and T2T (Trade-to-Trade) names are excluded by list.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date
from zoneinfo import ZoneInfo

from lab.core.constants import INDIA_TZ
from lab.core.types import Candle


def average_daily_turnover(candles: Sequence[Candle], *, timezone: str = INDIA_TZ) -> float:
    """Return mean daily turnover (sum of price x volume per IST day, averaged)."""
    tz = ZoneInfo(timezone)
    by_day: dict[date, float] = defaultdict(float)
    for candle in candles:
        by_day[candle.timestamp.astimezone(tz).date()] += candle.close * candle.volume
    return sum(by_day.values()) / len(by_day) if by_day else 0.0


def passes_liquidity(
    candles: Sequence[Candle], *, min_daily_turnover: float, timezone: str = INDIA_TZ
) -> bool:
    """Return whether average daily turnover clears ``min_daily_turnover``."""
    return average_daily_turnover(candles, timezone=timezone) >= min_daily_turnover


@dataclass(frozen=True, slots=True)
class RestrictedList:
    """The set of symbols under surveillance segments (ESM / T2T)."""

    esm: frozenset[str] = field(default_factory=frozenset)
    t2t: frozenset[str] = field(default_factory=frozenset)

    def is_restricted(self, symbol: str) -> bool:
        """Return whether ``symbol`` is on the ESM or T2T list."""
        return symbol in self.esm or symbol in self.t2t


def exclude_restricted(symbols: Iterable[str], restricted: RestrictedList) -> list[str]:
    """Return ``symbols`` with ESM/T2T names removed, order preserved."""
    return [symbol for symbol in symbols if not restricted.is_restricted(symbol)]
