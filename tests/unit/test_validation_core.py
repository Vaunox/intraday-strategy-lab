"""Tests for the validation core: purged CV, cost model, backtester (P2.1)."""

from __future__ import annotations

from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from lab.core.types import BarInterval, Candle, Side
from lab.research.validation.backtester import run_backtest
from lab.research.validation.costs import CostModel, load_cost_model
from lab.research.validation.robustness import two_engines_agree, vectorized_backtest
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


# --- square-off honors the configured MIS cutoff (Phase-3 square-off fix) ---- #
def _bar_hm(hour: int, minute: int, open_: float, high: float, low: float, close: float) -> Candle:
    ts = datetime(2024, 7, 15, hour, minute, tzinfo=IST)
    return Candle("X", BarInterval.MIN_5, ts, open_, high, low, close, 1000)


def test_square_off_honors_cutoff_when_grid_runs_past() -> None:
    # Kite's 5-min grid runs to 15:25, past the 15:20 MIS cutoff. A long held all day
    # must square off at the last bar BEFORE 15:20 (the 15:15 bar's close = the price
    # at 15:20) and never trade the 15:20/15:25 bars.
    candles = [
        _bar_hm(15, 10, 100, 100.5, 99.5, 100),
        _bar_hm(15, 15, 100, 102.5, 99.5, 102),  # last pre-cutoff bar -> square-off here
        _bar_hm(15, 20, 102, 105.5, 101.5, 105),  # >= 15:20: never held
        _bar_hm(15, 25, 105, 108.5, 104.5, 108),
    ]
    targets = [1.0, 1.0, 1.0, 1.0]
    result = run_backtest(candles, targets, ZERO_COST, square_off=time(15, 20))
    assert result.trades
    for tr in result.trades:
        assert tr.exit_time.astimezone(IST).time() <= time(15, 20)  # never past the cutoff
    last = result.trades[-1]
    assert last.exit_time.astimezone(IST).time() == time(15, 15)  # last pre-cutoff bar
    assert last.exit_price == 102.0  # its close = the 15:20 price
    # Control: WITHOUT the cutoff the same day exits at the 15:25 last bar (price 108) —
    # a materially different trade. So the reconciliation below is on a cutoff-CHANGED
    # exit, and the two engines agree because BOTH honor 15:20 — not because both fell
    # through to the last bar (that failure mode is ruled out by the next line).
    legacy = run_backtest(candles, targets, ZERO_COST)  # no cutoff -> last-bar exit
    assert legacy.trades[-1].exit_time.astimezone(IST).time() == time(15, 25)
    assert legacy.trades[-1].exit_price == 108.0
    assert not two_engines_agree(result, legacy)  # cutoff-honored trade != last-bar trade
    # Both engines, under the cutoff, must reconcile on that cutoff-changed trade.
    vec = vectorized_backtest(candles, targets, ZERO_COST, square_off=time(15, 20))
    assert two_engines_agree(result, vec)  # a cutoff-ignoring vec would equal `legacy` and fail


def test_no_new_position_opens_at_or_after_cutoff() -> None:
    # The only long signal (target index 1, the 15:15 close) would fill at the 15:20
    # open — at the cutoff — so no position is opened, and no trade results.
    candles = [
        _bar_hm(15, 10, 100, 101, 99, 100),
        _bar_hm(15, 15, 100, 101, 99, 100),
        _bar_hm(15, 20, 100, 101, 99, 100),
        _bar_hm(15, 25, 100, 101, 99, 100),
    ]
    targets = [0.0, 1.0, 1.0, 1.0]
    assert run_backtest(candles, targets, ZERO_COST, square_off=time(15, 20)).trades == ()


def test_square_off_none_is_legacy_last_bar() -> None:
    # Without a cutoff, legacy behaviour: exit at the day's last bar even if it is 15:25.
    candles = [_bar_hm(15, 10, 100, 100.5, 99.5, 100), _bar_hm(15, 25, 100, 108.5, 99.5, 108)]
    (trade,) = run_backtest(candles, [1.0, 1.0], ZERO_COST).trades
    assert trade.exit_time.astimezone(IST).time() == time(15, 25)


def test_square_off_short_grid_falls_back_to_last_bar() -> None:
    # A short session whose grid ends BEFORE the 15:20 cutoff (a half-day / early close,
    # or a symbol with a truncated grid). With square_off SET it must degrade to last-bar
    # behaviour: entries not wrongly blocked, square-off at the ACTUAL last bar, no forced
    # close at a wrong bar — identical to the no-cutoff run, and both engines reconcile.
    candles = [
        _bar_hm(12, 45, 100, 100.5, 99.5, 100),
        _bar_hm(12, 50, 100, 101.5, 99.5, 101),
        _bar_hm(12, 55, 101, 103.5, 100.5, 103),  # day's last bar, well before 15:20
    ]
    targets = [1.0, 1.0, 1.0]
    result = run_backtest(candles, targets, ZERO_COST, square_off=time(15, 20))
    (tr,) = result.trades
    assert tr.entry_time.astimezone(IST).time() == time(12, 50)  # next-bar-open entry, not blocked
    assert tr.exit_time.astimezone(IST).time() == time(12, 55)  # squared off at the actual last bar
    assert tr.exit_price == 103.0  # no forced close at a wrong bar
    # Graceful: identical to the legacy (no-cutoff) run, and both engines reconcile.
    assert two_engines_agree(result, run_backtest(candles, targets, ZERO_COST))
    assert two_engines_agree(
        result, vectorized_backtest(candles, targets, ZERO_COST, square_off=time(15, 20))
    )


def test_square_off_session_entirely_past_cutoff_is_a_clean_skip() -> None:
    # A special evening session (e.g. Diwali Muhurat, ~18:00-19:00) is entirely past the
    # 15:20 MIS cutoff. The daily cutoff makes it untradeable -> no trades, no crash, no
    # forced close at a wrong bar (a clean skip), and both engines agree (empty == empty).
    candles = [
        _bar_hm(18, 15, 100, 100.5, 99.5, 100),
        _bar_hm(18, 20, 100, 101.5, 99.5, 101),
    ]
    targets = [1.0, 1.0]
    result = run_backtest(candles, targets, ZERO_COST, square_off=time(15, 20))
    assert result.trades == ()
    assert vectorized_backtest(candles, targets, ZERO_COST, square_off=time(15, 20)).trades == ()
    assert two_engines_agree(
        result, vectorized_backtest(candles, targets, ZERO_COST, square_off=time(15, 20))
    )
