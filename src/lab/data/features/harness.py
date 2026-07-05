"""Dual-path skew harness (Phase 1, P1.5 / P1.6).

The train/serve skew tripwire. The **vectorized** path computes a feature over the
whole series at once; the **incremental** path computes it from only the bars up
to each point (a prefix). For a genuinely point-in-time feature the two must agree
at every bar — so any lookahead (a centered window, a full-series statistic, use
of a future bar) makes them diverge and is caught here, in CI (P1.6).

``find_skew`` accepts the feature function under test (default: the real
:func:`compute_features`), so the leakage suite can point it at a deliberately
leaky feature set and assert the skew is detected.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from lab.data.features.compute import FeatureConfig, compute_features
from lab.data.features.ohlcv import OHLCV, FloatArray

#: A feature function: OHLCV (+ optional config) -> named feature arrays.
FeatureFn = Callable[[OHLCV, FeatureConfig | None], dict[str, FloatArray]]

#: Absolute tolerance for the two paths to be considered equal.
DEFAULT_ATOL = 1e-9


@dataclass(frozen=True, slots=True)
class Skew:
    """A single bar where the vectorized and incremental paths disagree."""

    feature: str
    index: int
    vectorized: float
    incremental: float


def find_skew(
    data: OHLCV,
    config: FeatureConfig | None = None,
    *,
    feature_fn: FeatureFn = compute_features,
    atol: float = DEFAULT_ATOL,
) -> list[Skew]:
    """Return every bar/feature where the vectorized and incremental paths differ."""
    vectorized = feature_fn(data, config)
    skews: list[Skew] = []
    for index in range(len(data)):
        prefix = data.prefix(index + 1)
        incremental = {
            name: float(values[-1]) for name, values in feature_fn(prefix, config).items()
        }
        for name, values in vectorized.items():
            vec = float(values[index])
            inc = incremental[name]
            if np.isnan(vec) and np.isnan(inc):
                continue
            if np.isnan(vec) != np.isnan(inc) or abs(vec - inc) > atol:
                skews.append(Skew(feature=name, index=index, vectorized=vec, incremental=inc))
    return skews


def assert_point_in_time(
    data: OHLCV,
    config: FeatureConfig | None = None,
    *,
    feature_fn: FeatureFn = compute_features,
    atol: float = DEFAULT_ATOL,
) -> None:
    """Raise ``AssertionError`` if any feature is not point-in-time (skew found)."""
    skews = find_skew(data, config, feature_fn=feature_fn, atol=atol)
    if skews:
        preview = ", ".join(f"{s.feature}@{s.index}" for s in skews[:5])
        raise AssertionError(f"{len(skews)} vectorized/incremental skew(s); first: {preview}")
