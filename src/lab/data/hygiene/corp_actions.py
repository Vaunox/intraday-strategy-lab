"""Corporate-action adjustment (Phase 1, P1.4).

Back-adjusts a raw OHLCV series for splits, bonuses, and dividends so prices form
a continuous series across ex-dates. Raw candles are stored untouched; the
adjusted series is a *derived* layer (Part III Layer 1) — always recomputed from
raw + the action list, so the job is idempotent (re-running yields the same
adjusted output; never adjust already-adjusted data).

Convention: each action carries a ``price_factor`` (< 1 pushes pre-ex prices down
onto the post-ex scale) and a ``volume_factor`` (splits/bonuses inflate share
count, so pre-ex volume scales up; dividends leave volume unchanged). A candle's
adjustment is the product of the factors of every action whose ex-date is *after*
that candle's trading date.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from zoneinfo import ZoneInfo

from lab.core.constants import INDIA_TZ
from lab.core.logging import get_logger
from lab.core.types import Candle

_log = get_logger("data.hygiene.corp_actions")


@dataclass(frozen=True, slots=True)
class CorporateAction:
    """A corporate action: an ex-date plus price/volume back-adjustment factors."""

    ex_date: date
    price_factor: float
    volume_factor: float = 1.0

    @classmethod
    def split(cls, ex_date: date, ratio: float) -> CorporateAction:
        """A stock split of ``ratio``-for-1 (e.g. ``ratio=2`` for a 2:1 split)."""
        if ratio <= 0:
            raise ValueError(f"split ratio must be positive; got {ratio}")
        return cls(ex_date=ex_date, price_factor=1.0 / ratio, volume_factor=ratio)

    @classmethod
    def bonus(cls, ex_date: date, held: int, received: int) -> CorporateAction:
        """A bonus issue of ``received`` new shares per ``held`` existing shares."""
        ratio = (held + received) / held
        return cls(ex_date=ex_date, price_factor=1.0 / ratio, volume_factor=ratio)

    @classmethod
    def dividend(cls, ex_date: date, dividend: float, reference_close: float) -> CorporateAction:
        """A cash dividend of ``dividend`` against the pre-ex ``reference_close``."""
        if reference_close <= 0:
            raise ValueError("reference_close must be positive")
        return cls(ex_date=ex_date, price_factor=1.0 - dividend / reference_close)


def adjust_candles(
    candles: Sequence[Candle], actions: Sequence[CorporateAction], *, timezone: str = INDIA_TZ
) -> list[Candle]:
    """Return ``candles`` back-adjusted for ``actions`` (raw input is not mutated).

    A candle is scaled by the product of the factors of every action whose
    ex-date is strictly after the candle's IST trading date.
    """
    tz = ZoneInfo(timezone)
    adjusted: list[Candle] = []
    for candle in candles:
        trading_date = candle.timestamp.astimezone(tz).date()
        price_factor = 1.0
        volume_factor = 1.0
        for action in actions:
            if action.ex_date > trading_date:
                price_factor *= action.price_factor
                volume_factor *= action.volume_factor
        adjusted.append(
            Candle(
                symbol=candle.symbol,
                interval=candle.interval,
                timestamp=candle.timestamp,
                open=candle.open * price_factor,
                high=candle.high * price_factor,
                low=candle.low * price_factor,
                close=candle.close * price_factor,
                volume=round(candle.volume * volume_factor),
                open_interest=candle.open_interest,
            )
        )
    if actions:
        _log.info("candles_adjusted", candles=len(candles), actions=len(actions))
    return adjusted
