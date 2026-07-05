"""Tests for the validation core: purged CV, cost model, backtester (P2.1)."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from lab.core.types import BarInterval, Candle, Side
from lab.research.validation.backtester import run_backtest
from lab.research.validation.costs import CostModel, load_cost_model
from lab.research.validation.splitter import PurgedKFold

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config"
IST = ZoneInfo("Asia/Kolkata")

ZERO_COST = CostModel(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)


# --- purged k-fold ---------------------------------------------------------- #
def _daily_obs(day_nums: list[int]) -> tuple[list[datetime], list[datetime]]:
    entries = [datetime(2024, 7, d, 9, 15, tzinfo=IST) for d in day_nums]
    exits = [datetime(2024, 7, d, 15, 20, tzinfo=IST) for d in day_nums]
    return entries, exits


def test_purged_kfold_embargoes_adjacent_days() -> None:
    entries, exits = _daily_obs([15, 16, 17, 18, 19, 20])
    folds = PurgedKFold(n_splits=3, embargo=timedelta(days=1)).split(entries, exits)
    assert len(folds) == 3
    # First fold tests days [15,16]; day 17 (index 2) is embargoed out of training.
    first = folds[0]
    assert first.test == (0, 1)
    assert 2 not in first.train
    assert set(first.train) == {3, 4, 5}
    # Middle fold tests [17,18]; earlier days remain, day 19 is embargoed.
    middle = folds[1]
    assert middle.test == (2, 3)
    assert set(middle.train) == {0, 1, 5}


def test_purge_removes_overlapping_window() -> None:
    # obs 0 spans days 15-19 (a long label); the test fold is day 17 -> overlap.
    entries = [datetime(2024, 7, d, 9, 15, tzinfo=IST) for d in (15, 16, 17, 18, 19)]
    exits = [datetime(2024, 7, d, 15, 20, tzinfo=IST) for d in (19, 16, 17, 18, 19)]
    folds = PurgedKFold(n_splits=5, embargo=timedelta(0)).split(entries, exits)
    day17_fold = next(f for f in folds if f.test == (2,))
    assert 0 not in day17_fold.train  # obs 0's window overlaps the test day -> purged


def test_purged_kfold_rejects_too_few_observations() -> None:
    entries, exits = _daily_obs([15, 16])
    with pytest.raises(ValueError, match="cannot make"):
        PurgedKFold(n_splits=3).split(entries, exits)


# --- cost model ------------------------------------------------------------- #
def test_round_trip_cost_hand_computed() -> None:
    costs = load_cost_model(REPO_CONFIG)
    # buy=sell=1,00,000: brokerage 2x min(30,20)=40; STT 25; exch 5.94; SEBI 0.2;
    # GST 0.18*46.14=8.3052; stamp 3; slippage 100 -> 182.4452.
    assert costs.round_trip_cost(100_000, 100_000) == pytest.approx(182.4452, abs=1e-3)


def test_round_trip_cost_fraction_in_realistic_band() -> None:
    costs = load_cost_model(REPO_CONFIG)
    fraction = costs.round_trip_cost_fraction(100_000)
    assert 0.0012 <= fraction <= 0.0020  # ~0.12-0.20% round trip


def test_stress_widens_slippage() -> None:
    costs = load_cost_model(REPO_CONFIG)
    assert costs.round_trip_cost_fraction(100_000, stressed=True) > costs.round_trip_cost_fraction(
        100_000
    )


def test_slippage_grows_with_participation() -> None:
    costs = load_cost_model(REPO_CONFIG)
    small = costs.round_trip_cost_fraction(100_000, participation=0.001)
    large = costs.round_trip_cost_fraction(100_000, participation=0.05)
    flat = costs.round_trip_cost_fraction(100_000, participation=0.0)
    assert flat < small < large  # bigger order in the same bar pays more slippage


def test_participation_is_capped() -> None:
    costs = load_cost_model(REPO_CONFIG)
    at_cap = costs.round_trip_cost_fraction(100_000, participation=costs.slippage_participation_cap)
    beyond = costs.round_trip_cost_fraction(100_000, participation=10_000.0)
    assert beyond == pytest.approx(at_cap)  # a thin/zero-volume bar cannot explode the cost


def test_trade_cost_fraction_uses_bar_liquidity() -> None:
    from lab.research.validation.costs import trade_cost_fraction

    costs = load_cost_model(REPO_CONFIG)
    thick = trade_cost_fraction(costs, 100_000, entry_price=100.0, entry_volume=1_000_000.0)
    thin = trade_cost_fraction(costs, 100_000, entry_price=100.0, entry_volume=10_000.0)
    assert thin > thick  # same order, thinner bar -> higher participation -> higher cost


# --- backtester ------------------------------------------------------------- #
def _bar(minute: int, open_: float, high: float, low: float, close: float) -> Candle:
    ts = datetime(2024, 7, 15, 9, minute, tzinfo=IST)
    return Candle("X", BarInterval.MIN_5, ts, open_, high, low, close, 1000)


def test_long_trade_next_bar_open_fill_and_squareoff() -> None:
    candles = [
        _bar(15, 100, 103, 99, 101),
        _bar(20, 102, 104, 101, 103),
        _bar(25, 101, 104, 100, 103),
    ]
    # Long decided at bar 0 close -> filled at bar 1 open (102); square-off at
    # the last bar's close (103).
    result = run_backtest(candles, [1.0, 1.0, 0.0], ZERO_COST)
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.side is Side.LONG
    assert trade.entry_price == 102.0
    assert trade.exit_price == 103.0
    assert trade.gross_return == pytest.approx(103 / 102 - 1)
    assert trade.net_return == pytest.approx(trade.gross_return)  # zero-cost model


def test_short_trade_direction() -> None:
    candles = [_bar(15, 100, 101, 99, 100), _bar(20, 100, 101, 98, 99)]
    result = run_backtest(candles, [-1.0, -1.0], ZERO_COST)
    (trade,) = result.trades
    assert trade.side is Side.SHORT
    # Short entered at bar1 open (100), squared off at bar1 close (99): +1%.
    assert trade.gross_return == pytest.approx(1 - 99 / 100)


def test_flat_signal_produces_no_trades() -> None:
    candles = [_bar(15, 100, 101, 99, 100), _bar(20, 100, 101, 99, 100)]
    assert run_backtest(candles, [0.0, 0.0], ZERO_COST).trades == ()


def test_no_overnight_signal_carry() -> None:
    day1 = [_bar(15, 100, 101, 99, 100), _bar(20, 100, 101, 99, 100)]
    day2 = [
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
        Candle(
            "X",
            BarInterval.MIN_5,
            datetime(2024, 7, 16, 9, 20, tzinfo=IST),
            100,
            101,
            99,
            100,
            1000,
        ),
    ]
    # Long signalled on day1's last bar must NOT open a position at day2's open.
    result = run_backtest(day1 + day2, [0.0, 1.0, 0.0, 0.0], ZERO_COST)
    assert all(t.entry_time.day == 15 for t in result.trades) or result.trades == ()


def test_costs_reduce_net_return() -> None:
    costs = load_cost_model(REPO_CONFIG)
    candles = [_bar(15, 100, 110, 99, 101), _bar(20, 100, 110, 99, 110)]
    (trade,) = run_backtest(candles, [1.0, 1.0], costs).trades
    assert trade.net_return < trade.gross_return
    assert trade.gross_return - trade.net_return == pytest.approx(trade.cost_fraction)
