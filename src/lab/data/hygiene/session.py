"""Regular-session filter (Phase 1 hygiene): keep only in-session bars.

Kite historical candles include out-of-session bars — notably Diwali **Muhurat**
evening sessions (~18:15-19:15 IST, one per year) — that must not enter feature or
backtest computation (they would contaminate intraday-cumulative and rolling
features like VWAP / opening-range / gap). This filters the intraday grid to the
configured regular session (``[session.open, session.close)``, e.g. 09:15-15:30)
at the ingest boundary, leaving the immutable raw Parquet store whole — corrections
downstream, never a silent mutation of source, the same boundary-filtering
principle as the intraday square-off cutoff.
"""

from __future__ import annotations

from collections.abc import Sequence

from lab.core.nse_calendar import NseCalendar
from lab.core.types import Candle


def regular_session_candles(candles: Sequence[Candle], calendar: NseCalendar) -> list[Candle]:
    """Return only the candles whose timestamp falls within the regular session.

    Out-of-session bars (Muhurat evening sessions, pre-open, post-close) are dropped
    so they cannot reach ``OHLCV.from_candles`` / ``compute_features`` or the
    backtester. Uses the config-driven session window
    (:meth:`~lab.core.nse_calendar.NseCalendar.is_regular_session_time`); the raw
    store is never touched.
    """
    return [c for c in candles if calendar.is_regular_session_time(c.timestamp)]
