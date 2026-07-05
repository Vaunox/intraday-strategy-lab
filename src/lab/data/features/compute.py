"""The feature-store contract: ``compute_features`` (Phase 1, P1.5).

Assembles the full Layer-1 feature family into a single named, versioned vector,
computed identically everywhere (backfill and any future serving). Because every
underlying indicator is causal, the whole vector is point-in-time: the value at
bar ``i`` depends only on bars ``0..i`` — verified bar-by-bar by the dual-path
skew test in :mod:`lab.data.features.harness`.

Feature parameters live in one typed :class:`FeatureConfig` (not scattered magic
numbers); Phase-3 study specs pin these from ``config/``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

from lab.core.config import Settings
from lab.data.features import indicators
from lab.data.features.ohlcv import OHLCV, FloatArray

#: Version of the feature set; bump when definitions change so results are traceable.
FEATURE_SET_VERSION = "1.0.0"

#: The ordered feature names produced by :func:`compute_features` (a test asserts
#: this stays in lockstep with the implementation).
FEATURE_NAMES: tuple[str, ...] = (
    "sma",
    "ema",
    "kama",
    "rsi",
    "adx",
    "atr",
    "macd_line",
    "macd_signal",
    "macd_hist",
    "bb_upper",
    "bb_middle",
    "bb_lower",
    "bb_percent_b",
    "donchian_upper",
    "donchian_lower",
    "relative_volume",
    "realized_vol",
    "vwap",
    "vwap_deviation",
    "pivot",
    "gap",
    "or_high",
    "or_low",
)


@dataclass(frozen=True, slots=True)
class FeatureConfig:
    """Parameters for the Layer-1 feature family (the single source of truth)."""

    sma_period: int = 20
    ema_period: int = 20
    kama_period: int = 10
    rsi_period: int = 14
    adx_period: int = 14
    atr_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_num_std: float = 2.0
    donchian_period: int = 20
    relative_volume_period: int = 20
    realized_vol_period: int = 20
    opening_range_minutes: int = 15

    def __post_init__(self) -> None:
        """Validate every parameter is a positive number (fail loudly on bad config)."""
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
                raise ValueError(
                    f"feature config {field.name!r} must be a positive number; got {value!r}"
                )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> FeatureConfig:
        """Build a config from a mapping, overriding defaults; reject unknown keys."""
        known = {field.name for field in fields(cls)}
        unknown = set(mapping) - known
        if unknown:
            raise ValueError(f"unknown feature config keys: {sorted(unknown)}")
        return cls(**{key: value for key, value in mapping.items() if key in known})

    @classmethod
    def from_settings(cls, settings: Settings) -> FeatureConfig:
        """Build a config from the ``features`` section of loaded settings."""
        raw = settings.raw.get("features")
        if not isinstance(raw, Mapping):
            return cls()
        return cls.from_mapping(raw)


def compute_features(data: OHLCV, config: FeatureConfig | None = None) -> dict[str, FloatArray]:
    """Compute the full named feature vector over ``data`` (point-in-time)."""
    cfg = config or FeatureConfig()
    macd_line, macd_signal, macd_hist = indicators.macd(
        data, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal
    )
    bb_upper, bb_middle, bb_lower = indicators.bollinger(data, cfg.bb_period, cfg.bb_num_std)
    donchian_upper, donchian_lower = indicators.donchian(data, cfg.donchian_period)
    or_high, or_low = indicators.opening_range(data, cfg.opening_range_minutes)

    return {
        "sma": indicators.sma(data, cfg.sma_period),
        "ema": indicators.ema(data, cfg.ema_period),
        "kama": indicators.kama(data, cfg.kama_period),
        "rsi": indicators.rsi(data, cfg.rsi_period),
        "adx": indicators.adx(data, cfg.adx_period),
        "atr": indicators.atr(data, cfg.atr_period),
        "macd_line": macd_line,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist,
        "bb_upper": bb_upper,
        "bb_middle": bb_middle,
        "bb_lower": bb_lower,
        "bb_percent_b": indicators.percent_b(data, cfg.bb_period, cfg.bb_num_std),
        "donchian_upper": donchian_upper,
        "donchian_lower": donchian_lower,
        "relative_volume": indicators.relative_volume(data, cfg.relative_volume_period),
        "realized_vol": indicators.realized_volatility(data, cfg.realized_vol_period),
        "vwap": indicators.vwap(data),
        "vwap_deviation": indicators.vwap_deviation(data),
        "pivot": indicators.pivot(data),
        "gap": indicators.gap(data),
        "or_high": or_high,
        "or_low": or_low,
    }


def feature_names() -> list[str]:
    """Return the ordered feature names produced by :func:`compute_features`."""
    return list(FEATURE_NAMES)
