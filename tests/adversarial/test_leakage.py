"""Adversarial leakage & skew suite (Phase 1, P1.6).

Runs in CI and must TRIP on injected leakage. It checks the three silent killers
around features: train/serve skew (dual-path harness), forward-shift invariance
(appending future bars must not change past values), trailing-only normalization
(a full-series statistic is leakage), and near-perfect future correlation (a sign
of a feature that has literally seen the future).

Each check is demonstrated both ways: the honest feature set passes, and a
deliberately leaky variant is caught — so the suite has teeth.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from lab.core.types import BarInterval, Candle
from lab.data.features.compute import FeatureConfig, compute_features
from lab.data.features.harness import find_skew
from lab.data.features.ohlcv import OHLCV, FloatArray

pytestmark = pytest.mark.adversarial
IST = ZoneInfo("Asia/Kolkata")


def _synthetic(bars_per_day: int = 30) -> OHLCV:
    rng = np.random.default_rng(7)
    candles: list[Candle] = []
    price = 100.0
    for day in (date(2024, 7, 15), date(2024, 7, 16)):
        open_ts = datetime(day.year, day.month, day.day, 9, 15, tzinfo=IST)
        for bar in range(bars_per_day):
            prev = price
            price = prev * math.exp(float(rng.normal(0.0, 0.003)))
            candles.append(
                Candle(
                    "SYN",
                    BarInterval.MIN_5,
                    open_ts + timedelta(minutes=5 * bar),
                    prev,
                    max(prev, price) + 0.4,
                    min(prev, price) - 0.4,
                    price,
                    1000 + int(rng.integers(0, 500)),
                )
            )
    return OHLCV.from_candles(candles)


# --- leaky and honest feature functions ------------------------------------- #
def _lookahead_compute(data: OHLCV, config: FeatureConfig | None = None) -> dict[str, FloatArray]:
    """Honest features plus a feature that peeks one bar into the future."""
    features = dict(compute_features(data, config))
    peek = np.full(len(data), np.nan, dtype=np.float64)
    if len(data) > 1:
        peek[:-1] = data.close[1:]  # value at i uses close[i+1] — lookahead
    features["lookahead_close"] = peek
    return features


def _global_zscore_compute(
    data: OHLCV, config: FeatureConfig | None = None
) -> dict[str, FloatArray]:
    """Honest features plus a z-score normalized on FULL-series stats (leakage)."""
    features = dict(compute_features(data, config))
    std = float(data.close.std())
    z = (data.close - data.close.mean()) / std if std > 0 else np.zeros(len(data))
    features["global_z"] = z
    return features


def _trailing_zscore_compute(
    data: OHLCV, config: FeatureConfig | None = None
) -> dict[str, FloatArray]:
    """Honest features plus a correctly trailing (expanding-window) z-score."""
    features = dict(compute_features(data, config))
    z = np.full(len(data), np.nan, dtype=np.float64)
    for i in range(len(data)):
        window = data.close[: i + 1]
        std = float(window.std())
        z[i] = (float(data.close[i]) - float(window.mean())) / std if std > 0 else 0.0
    features["trailing_z"] = z
    return features


def _abs_future_correlation(feature: FloatArray, forward_return: FloatArray) -> float:
    mask = ~(np.isnan(feature) | np.isnan(forward_return))
    if int(mask.sum()) < 2:
        return 0.0
    return float(abs(np.corrcoef(feature[mask], forward_return[mask])[0, 1]))


# --- train/serve skew (the core tripwire) ----------------------------------- #
def test_harness_passes_on_honest_features() -> None:
    assert find_skew(_synthetic()) == []


def test_harness_trips_on_lookahead_feature() -> None:
    skews = find_skew(_synthetic(), feature_fn=_lookahead_compute)
    assert skews, "the leakage tripwire failed to detect a lookahead feature"
    assert any(s.feature == "lookahead_close" for s in skews)


# --- trailing-only normalization -------------------------------------------- #
def test_global_normalization_is_flagged() -> None:
    skews = find_skew(_synthetic(), feature_fn=_global_zscore_compute)
    assert any(s.feature == "global_z" for s in skews)


def test_trailing_normalization_is_clean() -> None:
    skews = find_skew(_synthetic(), feature_fn=_trailing_zscore_compute)
    assert [s for s in skews if s.feature == "trailing_z"] == []


# --- forward-shift invariance ----------------------------------------------- #
def test_causal_features_are_forward_shift_invariant() -> None:
    data = _synthetic()
    cut = len(data) - 10
    full = compute_features(data)
    partial = compute_features(data.prefix(cut))
    for name, values in full.items():
        earlier = values[:cut]
        recomputed = partial[name]
        assert np.array_equal(np.isnan(earlier), np.isnan(recomputed))
        mask = ~np.isnan(earlier)
        assert np.allclose(earlier[mask], recomputed[mask], atol=1e-9)


def test_lookahead_feature_is_not_shift_invariant() -> None:
    data = _synthetic()
    cut = len(data) - 10
    full = _lookahead_compute(data)["lookahead_close"][:cut]
    partial = _lookahead_compute(data.prefix(cut))["lookahead_close"]
    # The boundary value changes when future bars are (un)available — that's the tell.
    assert not np.array_equal(np.nan_to_num(full), np.nan_to_num(partial))


# --- suspicious future correlation ------------------------------------------ #
def test_direct_future_leakage_has_near_perfect_correlation() -> None:
    data = _synthetic()
    forward_return = np.full(len(data), np.nan, dtype=np.float64)
    forward_return[:-1] = data.close[1:] / data.close[:-1] - 1.0
    leaky_feature = forward_return.copy()  # feature IS the future return
    assert _abs_future_correlation(leaky_feature, forward_return) > 0.99


def test_honest_feature_is_not_near_perfectly_future_correlated() -> None:
    data = _synthetic()
    forward_return = np.full(len(data), np.nan, dtype=np.float64)
    forward_return[:-1] = data.close[1:] / data.close[:-1] - 1.0
    realized_vol = compute_features(data)["realized_vol"]
    assert _abs_future_correlation(realized_vol, forward_return) < 0.95
