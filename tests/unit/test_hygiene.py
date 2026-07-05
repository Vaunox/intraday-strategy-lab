"""Tests for the data-hygiene jobs (P1.4): corp actions, survivorship,
bad-tick filtering, gap detection, liquidity screen, ESM/T2T exclusion.
"""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

from lab.core.types import BarInterval, Candle
from lab.data.hygiene.corp_actions import CorporateAction, adjust_candles
from lab.data.hygiene.quality import detect_gaps, filter_bad_ticks
from lab.data.hygiene.screening import (
    RestrictedList,
    average_daily_turnover,
    exclude_restricted,
    passes_liquidity,
)
from lab.data.hygiene.survivorship import ConstituentRecord, PointInTimeUniverse

IST = ZoneInfo("Asia/Kolkata")


def _day_candle(day: int, close: float, volume: int) -> Candle:
    ts = datetime(2024, 7, day, 9, 15, tzinfo=IST)
    return Candle("RELIANCE", BarInterval.MIN_5, ts, close, close + 1.0, close - 1.0, close, volume)


def _bar(minute: int, close: float) -> Candle:
    ts = datetime(2024, 7, 15, 9, minute, tzinfo=IST)
    return Candle("X", BarInterval.MIN_5, ts, close, close + 1.0, close - 1.0, close, 1000)


# --- corporate actions ------------------------------------------------------ #
def test_split_adjusts_prices_and_volume() -> None:
    raw = [_day_candle(15, 200.0, 1000), _day_candle(16, 100.0, 2000)]
    actions = [CorporateAction.split(date(2024, 7, 16), ratio=2)]
    adjusted = adjust_candles(raw, actions)
    # Pre-ex day is scaled onto the post-split scale; ex day is untouched.
    assert adjusted[0].close == 100.0
    assert adjusted[0].volume == 2000
    assert adjusted[1].close == 100.0
    assert adjusted[1].volume == 2000


def test_dividend_adjusts_price_only() -> None:
    raw = [_day_candle(15, 200.0, 1000), _day_candle(16, 190.0, 1000)]
    actions = [CorporateAction.dividend(date(2024, 7, 16), dividend=10.0, reference_close=200.0)]
    adjusted = adjust_candles(raw, actions)
    assert adjusted[0].close == 190.0  # 200 * (1 - 10/200)
    assert adjusted[0].volume == 1000  # dividends do not adjust volume


def test_adjustment_is_idempotent_from_raw() -> None:
    raw = [_day_candle(15, 200.0, 1000), _day_candle(16, 100.0, 2000)]
    actions = [CorporateAction.split(date(2024, 7, 16), ratio=2)]
    assert adjust_candles(raw, actions) == adjust_candles(raw, actions)


def test_no_actions_leaves_series_unchanged() -> None:
    raw = [_day_candle(15, 200.0, 1000)]
    assert adjust_candles(raw, []) == raw


# --- survivorship ----------------------------------------------------------- #
def _universe() -> PointInTimeUniverse:
    return PointInTimeUniverse(
        [
            ConstituentRecord("ACTIVE", date(2020, 1, 1)),
            ConstituentRecord("DELISTED", date(2020, 1, 1), date(2023, 6, 30)),
            ConstituentRecord(
                "OLDNAME", date(2019, 1, 1), date(2021, 12, 31), renamed_to="NEWNAME"
            ),
            ConstituentRecord("NEWNAME", date(2022, 1, 1)),
        ]
    )


def test_delisted_names_included_historically() -> None:
    universe = _universe()
    assert "DELISTED" in universe.constituents_on(date(2023, 1, 1))
    assert "DELISTED" not in universe.constituents_on(date(2024, 1, 1))
    assert "ACTIVE" in universe.constituents_on(date(2024, 1, 1))
    assert "DELISTED" in universe.all_symbols()


def test_rename_chain_resolves() -> None:
    assert _universe().resolve_current_symbol("OLDNAME") == "NEWNAME"


# --- bad ticks -------------------------------------------------------------- #
def test_bad_tick_dropped_and_reported() -> None:
    series = [_bar(15, 100.0), _bar(20, 101.0), _bar(25, 160.0), _bar(30, 102.0), _bar(35, 103.0)]
    clean, corrections = filter_bad_ticks(series, max_return=0.20)
    assert [c.close for c in clean] == [100.0, 101.0, 102.0, 103.0]
    assert len(corrections) == 1
    assert corrections[0].reason == "return_jump"


def test_clean_series_unchanged() -> None:
    series = [_bar(15, 100.0), _bar(20, 100.5), _bar(25, 101.0)]
    clean, corrections = filter_bad_ticks(series)
    assert clean == series
    assert corrections == []


# --- gaps ------------------------------------------------------------------- #
def test_gap_detected() -> None:
    series = [_bar(15, 100.0), _bar(20, 101.0), _bar(30, 102.0)]  # missing 09:25
    gaps = detect_gaps(series, BarInterval.MIN_5)
    assert len(gaps) == 1
    assert gaps[0].missing_bars == 1


def test_contiguous_series_has_no_gaps() -> None:
    series = [_bar(15, 100.0), _bar(20, 101.0), _bar(25, 102.0)]
    assert detect_gaps(series, BarInterval.MIN_5) == []


def test_overnight_boundary_is_not_a_gap() -> None:
    overnight = [
        Candle(
            "X",
            BarInterval.MIN_5,
            datetime(2024, 7, 15, 15, 30, tzinfo=IST),
            100,
            101,
            99,
            100,
            1000,
        ),
        Candle(
            "X",
            BarInterval.MIN_5,
            datetime(2024, 7, 16, 9, 15, tzinfo=IST),
            100,
            101,
            99,
            100,
            1000,
        ),
    ]
    assert detect_gaps(overnight, BarInterval.MIN_5) == []


# --- screening -------------------------------------------------------------- #
def test_liquidity_screen() -> None:
    liquid = [_day_candle(15, 100.0, 100_000)]  # daily turnover = 10,000,000
    assert average_daily_turnover(liquid) == 10_000_000.0
    assert passes_liquidity(liquid, min_daily_turnover=1_000_000)
    assert not passes_liquidity(liquid, min_daily_turnover=100_000_000)


def test_restricted_exclusion() -> None:
    restricted = RestrictedList(esm=frozenset({"BADSTOCK"}), t2t=frozenset({"T2TSTOCK"}))
    assert exclude_restricted(["GOOD", "BADSTOCK", "T2TSTOCK"], restricted) == ["GOOD"]
    assert restricted.is_restricted("BADSTOCK")
    assert not restricted.is_restricted("GOOD")
