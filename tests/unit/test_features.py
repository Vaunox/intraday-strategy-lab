"""Tests for the feature library and the dual-path skew harness (P1.5)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from lab.core.config import load_settings
from lab.core.types import BarInterval, Candle
from lab.data.features import indicators
from lab.data.features.compute import (
    FEATURE_NAMES,
    FeatureConfig,
    compute_features,
    feature_names,
)
from lab.data.features.harness import assert_point_in_time, find_skew
from lab.data.features.ohlcv import OHLCV

IST = ZoneInfo("Asia/Kolkata")
REPO_CONFIG = Path(__file__).resolve().parents[2] / "config"


def _ohlcv(candles: Sequence[Candle]) -> OHLCV:
    return OHLCV.from_candles(candles)


def _ramp(closes: Sequence[float]) -> OHLCV:
    candles = [
        Candle(
            "R",
            BarInterval.MIN_5,
            datetime(2024, 7, 15, 9, 15, tzinfo=IST) + timedelta(minutes=5 * i),
            close,
            close + 0.5,
            max(close - 0.5, 0.1),
            close,
            1000,
        )
        for i, close in enumerate(closes)
    ]
    return _ohlcv(candles)


def _synthetic(bars_per_day: int = 40) -> OHLCV:
    """A deterministic multi-day 5-min series long enough to warm up indicators."""
    rng = np.random.default_rng(42)
    trading_days = [date(2024, 7, 15), date(2024, 7, 16), date(2024, 7, 18)]  # 17 = holiday
    candles: list[Candle] = []
    price = 100.0
    for day in trading_days:
        open_ts = datetime(day.year, day.month, day.day, 9, 15, tzinfo=IST)
        for bar in range(bars_per_day):
            prev = price
            price = prev * math.exp(float(rng.normal(0.0, 0.003)))
            high = max(prev, price) + 0.4
            low = min(prev, price) - 0.4
            volume = 1000 + int(rng.integers(0, 500))
            candles.append(
                Candle(
                    "SYN",
                    BarInterval.MIN_5,
                    open_ts + timedelta(minutes=5 * bar),
                    prev,
                    high,
                    low,
                    price,
                    volume,
                )
            )
    return _ohlcv(candles)


# --- the dual-path skew test (the P1.5 acceptance centerpiece) --------------- #
def test_features_are_point_in_time() -> None:
    data = _synthetic()
    assert_point_in_time(data)  # raises on any vectorized/incremental disagreement


def test_skew_check_is_not_vacuous() -> None:
    # Ensure the series actually warms up features (so the skew test has teeth).
    data = _synthetic()
    sma = compute_features(data)["sma"]
    assert np.count_nonzero(~np.isnan(sma)) > 20


# --- known-value checks (correctness, not just self-consistency) ------------- #
def test_sma_known_values() -> None:
    sma = indicators.sma(_ramp([1, 2, 3, 4, 5]), period=3)
    assert math.isnan(sma[1])
    assert sma[2] == 2.0  # mean(1,2,3)
    assert sma[4] == 4.0  # mean(3,4,5)


def test_vwap_resets_and_accumulates() -> None:
    day = datetime(2024, 7, 15, 9, 15, tzinfo=IST)
    candles = [
        Candle("V", BarInterval.MIN_5, day, 10, 11, 9, 10, 100),  # typical 10
        Candle(
            "V", BarInterval.MIN_5, day + timedelta(minutes=5), 20, 22, 18, 20, 100
        ),  # typical 20
    ]
    vwap = indicators.vwap(_ohlcv(candles))
    assert vwap[0] == 10.0
    assert vwap[1] == 15.0  # (10*100 + 20*100) / 200


def test_gap_known_value() -> None:
    d1 = datetime(2024, 7, 15, 9, 15, tzinfo=IST)
    d2 = datetime(2024, 7, 16, 9, 15, tzinfo=IST)
    candles = [
        Candle("G", BarInterval.MIN_5, d1, 100, 101, 99, 100, 1000),  # day1 last close 100
        Candle("G", BarInterval.MIN_5, d2, 105, 106, 104, 105, 1000),  # day2 open 105
    ]
    gap = indicators.gap(_ohlcv(candles))
    assert math.isnan(gap[0])  # no prior day
    assert gap[1] == pytest.approx(0.05)  # 105/100 - 1


def test_pivot_uses_prior_day_hlc() -> None:
    d1 = datetime(2024, 7, 15, 9, 15, tzinfo=IST)
    d2 = datetime(2024, 7, 16, 9, 15, tzinfo=IST)
    candles = [
        Candle("P", BarInterval.MIN_5, d1, 100, 110, 90, 100, 1000),  # day1 H110 L90 C100
        Candle("P", BarInterval.MIN_5, d2, 100, 101, 99, 100, 1000),
    ]
    pivot = indicators.pivot(_ohlcv(candles))
    assert math.isnan(pivot[0])
    assert pivot[1] == (110 + 90 + 100) / 3


# --- contract / purity ------------------------------------------------------- #
def test_feature_names_match_output() -> None:
    data = _synthetic(bars_per_day=40)
    assert tuple(compute_features(data).keys()) == FEATURE_NAMES
    assert feature_names() == list(FEATURE_NAMES)


def test_all_features_aligned_to_input_length() -> None:
    data = _synthetic()
    features = compute_features(data)
    assert all(len(values) == len(data) for values in features.values())


def test_compute_features_does_not_mutate_input() -> None:
    data = _synthetic()
    close_before = data.close.copy()
    volume_before = data.volume.copy()
    compute_features(data)
    assert np.array_equal(data.close, close_before)
    assert np.array_equal(data.volume, volume_before)


def test_real_feature_set_has_no_skew() -> None:
    assert find_skew(_synthetic()) == []


# --- feature config wiring (P1 follow-up) ------------------------------------ #
def test_feature_config_from_repo_settings() -> None:
    settings = load_settings("dev", config_dir=REPO_CONFIG, environ={})
    cfg = FeatureConfig.from_settings(settings)
    assert cfg.sma_period == 20
    assert cfg.rsi_period == 14
    assert cfg.bb_num_std == 2.0


def test_feature_config_env_var_override_flows_through() -> None:
    settings = load_settings(
        "dev", config_dir=REPO_CONFIG, environ={"LAB__FEATURES__RSI_PERIOD": "7"}
    )
    assert FeatureConfig.from_settings(settings).rsi_period == 7


def test_feature_config_from_mapping_overrides_defaults() -> None:
    cfg = FeatureConfig.from_mapping({"rsi_period": 9})
    assert cfg.rsi_period == 9
    assert cfg.sma_period == 20  # untouched default


def test_feature_config_rejects_unknown_key() -> None:
    with pytest.raises(ValueError, match="unknown feature config"):
        FeatureConfig.from_mapping({"bogus_period": 3})


def test_feature_config_rejects_non_positive_value() -> None:
    with pytest.raises(ValueError, match="positive"):
        FeatureConfig.from_mapping({"sma_period": -5})


# --- new Layer-1 families (P1.5 completion) --------------------------------- #
def test_full_feature_family_count() -> None:
    assert len(FEATURE_NAMES) == 42
    assert len(compute_features(_synthetic())) == 42


def test_momentum_known_value() -> None:
    m = indicators.momentum(_ramp([1, 1, 2, 4]), period=2)
    assert math.isnan(m[1])
    assert m[2] == pytest.approx(2.0 / 1.0 - 1)  # close[2]/close[0] - 1
    assert m[3] == pytest.approx(4.0 / 1.0 - 1)  # close[3]/close[1] - 1


def test_fibonacci_pivot_known_value() -> None:
    d1 = datetime(2024, 7, 15, 9, 15, tzinfo=IST)
    d2 = datetime(2024, 7, 16, 9, 15, tzinfo=IST)
    candles = [
        Candle("P", BarInterval.MIN_5, d1, 100, 110, 90, 100, 1000),  # H110 L90 C100
        Candle("P", BarInterval.MIN_5, d2, 100, 101, 99, 100, 1000),
    ]
    r1, s1 = indicators.fibonacci_pivot_levels(_ohlcv(candles))
    assert math.isnan(r1[0])
    assert r1[1] == pytest.approx(100 + 0.382 * 20)  # pivot 100, range 20
    assert s1[1] == pytest.approx(100 - 0.382 * 20)


def test_swing_high_confirms_local_max() -> None:
    highs = [10, 11, 12, 20, 12, 11, 10]  # clear peak (20) at index 3
    candles = [
        Candle(
            "S",
            BarInterval.MIN_5,
            datetime(2024, 7, 15, 9, 15, tzinfo=IST) + timedelta(minutes=5 * i),
            h - 1.0,
            float(h),
            h - 2.0,
            h - 1.0,
            1000,
        )
        for i, h in enumerate(highs)
    ]
    swing = indicators.swing_high(_ohlcv(candles), window=2)
    assert math.isnan(swing[4])  # not yet confirmed
    assert swing[5] == 20.0  # confirmed 2 bars after the peak
    assert swing[6] == 20.0  # carried forward


def test_cross_sectional_rank_is_point_in_time() -> None:
    panel = {
        "A": np.array([1.0, 5.0]),
        "B": np.array([2.0, 4.0]),
        "C": np.array([3.0, 3.0]),
    }
    ranks = indicators.cross_sectional_rank(panel)
    assert (ranks["A"][0], ranks["B"][0], ranks["C"][0]) == (0.0, 0.5, 1.0)
    assert (ranks["A"][1], ranks["B"][1], ranks["C"][1]) == (1.0, 0.5, 0.0)
