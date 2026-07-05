"""Bad-tick filtering and gap detection (Phase 1, P1.4).

Both jobs are pure and idempotent. Bad-tick filtering never silently mutates
prices: suspicious bars are dropped and each removal is returned as an auditable
:class:`TickCorrection` (and logged). Gap detection reports missing intraday bars
without altering the series.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import pairwise
from zoneinfo import ZoneInfo

from lab.core.constants import INDIA_TZ
from lab.core.logging import get_logger
from lab.core.types import BarInterval, Candle

_log = get_logger("data.hygiene.quality")

#: Default per-bar absolute return above which a tick is treated as suspect.
DEFAULT_MAX_RETURN = 0.20

_INTERVAL_STEP: dict[BarInterval, timedelta] = {
    BarInterval.MINUTE: timedelta(minutes=1),
    BarInterval.MIN_3: timedelta(minutes=3),
    BarInterval.MIN_5: timedelta(minutes=5),
    BarInterval.MIN_15: timedelta(minutes=15),
    BarInterval.MIN_60: timedelta(minutes=60),
    BarInterval.DAY: timedelta(days=1),
}


@dataclass(frozen=True, slots=True)
class TickCorrection:
    """An auditable record of one dropped suspect bar."""

    timestamp: datetime
    reason: str
    detail: str


@dataclass(frozen=True, slots=True)
class Gap:
    """A detected run of missing intraday bars between two present bars."""

    previous: datetime
    following: datetime
    missing_bars: int


def filter_bad_ticks(
    candles: Sequence[Candle], *, max_return: float = DEFAULT_MAX_RETURN
) -> tuple[list[Candle], list[TickCorrection]]:
    """Drop bars whose bar-to-bar return exceeds ``max_return`` as likely errors.

    Returns the cleaned series and the list of corrections. The return of each
    bar is measured against the last *accepted* close, so a lone spike does not
    also reject the bar that follows it.
    """
    clean: list[Candle] = []
    corrections: list[TickCorrection] = []
    last_good_close: float | None = None
    for candle in candles:
        if last_good_close is not None:
            move = abs(candle.close / last_good_close - 1.0)
            if move > max_return:
                correction = TickCorrection(
                    timestamp=candle.timestamp,
                    reason="return_jump",
                    detail=f"|move|={move:.4f} exceeds max_return={max_return}",
                )
                corrections.append(correction)
                _log.warning(
                    "bad_tick_dropped",
                    symbol=candle.symbol,
                    timestamp=candle.timestamp.isoformat(),
                    move=round(move, 4),
                )
                continue
        clean.append(candle)
        last_good_close = candle.close
    return clean, corrections


def detect_gaps(
    candles: Sequence[Candle], interval: BarInterval, *, timezone: str = INDIA_TZ
) -> list[Gap]:
    """Return intraday gaps (missing bars) in an ascending ``candles`` series.

    Overnight boundaries between trading days are expected and never reported for
    sub-daily intervals.
    """
    step = _INTERVAL_STEP[interval]
    tz = ZoneInfo(timezone)
    gaps: list[Gap] = []
    for previous, following in pairwise(candles):
        if interval is not BarInterval.DAY:
            prev_day = previous.timestamp.astimezone(tz).date()
            next_day = following.timestamp.astimezone(tz).date()
            if prev_day != next_day:
                continue  # overnight boundary — expected, not a gap
        delta = following.timestamp - previous.timestamp
        if delta > step:
            missing = int(delta / step) - 1
            if missing > 0:
                gaps.append(Gap(previous.timestamp, following.timestamp, missing))
    return gaps
