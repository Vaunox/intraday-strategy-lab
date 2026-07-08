"""Tests for the regular-session filter (Muhurat / out-of-session bars, P1 hygiene).

The key guarantee: a Diwali Muhurat evening bar cannot enter feature computation —
it is dropped at the ingest boundary before it can reach ``OHLCV.from_candles`` /
``compute_features``.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from lab.core.config import load_settings
from lab.core.nse_calendar import NseCalendar
from lab.core.types import BarInterval, Candle
from lab.data.features.ohlcv import OHLCV
from lab.data.hygiene.session import regular_session_candles

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config"
IST = ZoneInfo("Asia/Kolkata")


def _calendar() -> NseCalendar:
    return NseCalendar.from_settings(load_settings("dev", config_dir=REPO_CONFIG, environ={}))


def _candle(ts: datetime, close: float) -> Candle:
    return Candle("RELIANCE", BarInterval.MIN_5, ts, close, close + 1.0, close - 1.0, close, 1000)


def test_muhurat_evening_bar_cannot_enter_feature_computation() -> None:
    """A Muhurat evening bar (~18:30) is dropped by the filter, so it never reaches
    OHLCV.from_candles / compute_features."""
    cal = _calendar()
    regular = [
        _candle(datetime(2024, 11, 4, 9, 15, tzinfo=IST), 100.0),
        _candle(datetime(2024, 11, 4, 15, 25, tzinfo=IST), 101.0),  # last regular 5-min bar
    ]
    muhurat = _candle(datetime(2024, 11, 1, 18, 30, tzinfo=IST), 555.0)  # Diwali Muhurat evening

    filtered = regular_session_candles([regular[0], muhurat, regular[1]], cal)

    assert filtered == regular  # order preserved, Muhurat dropped
    assert muhurat not in filtered
    # The filtered series is exactly what becomes features — the Muhurat bar is absent.
    ohlcv = OHLCV.from_candles(filtered)
    assert muhurat.timestamp not in ohlcv.timestamps
    assert 555.0 not in ohlcv.close.tolist()


def test_filter_also_drops_pre_open_and_post_close() -> None:
    cal = _calendar()
    base = datetime(2024, 7, 15, tzinfo=IST)
    bars = [
        _candle(base.replace(hour=9, minute=0), 100.0),  # pre-open   -> dropped
        _candle(base.replace(hour=9, minute=15), 101.0),  # open       -> kept
        _candle(base.replace(hour=15, minute=25), 102.0),  # last bar  -> kept
        _candle(base.replace(hour=15, minute=40), 103.0),  # post-close-> dropped
    ]
    kept = regular_session_candles(bars, cal)
    assert [c.close for c in kept] == [101.0, 102.0]


def test_filter_keeps_a_full_regular_session_untouched() -> None:
    cal = _calendar()
    open_ts = datetime(2024, 7, 15, 9, 15, tzinfo=IST)
    day = [_candle(open_ts + timedelta(minutes=5 * i), 100.0 + i) for i in range(75)]
    # 09:15 + 74*5min = 15:25, all within [09:15, 15:30): nothing dropped.
    assert regular_session_candles(day, cal) == day
