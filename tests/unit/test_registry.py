"""Pins the frozen strategy registry: pre-registered params must match the studies.

The registry is the single source both drivers read, so this guards against the
registered frozen params drifting away from what a study's pre-registration committed.
"""

from __future__ import annotations

from lab.research.strategies.adaptive_ma import AdaptiveMaCrossSpec, AdaptiveMaSlopeSpec
from lab.research.strategies.breakout import BreakoutSpec
from lab.research.strategies.donchian_breakout import DonchianBreakoutSpec
from lab.research.strategies.mean_reversion import MeanReversionSpec
from lab.research.strategies.pivot_reversion import PivotReversionSpec
from lab.research.strategies.registry import STRATEGIES
from lab.research.strategies.reversal import ReversalSpec
from lab.research.strategies.volatility_filters import (
    VolContractionReversionSpec,
    VolExpansionBreakoutSpec,
)
from lab.research.strategies.vwap import VwapCrossSpec


def test_breakout_registered_with_frozen_prereg_params() -> None:
    entry = STRATEGIES["breakout"]
    assert entry.base_params == {"breakout_lookback": 20.0, "volume_mult": 1.5}
    assert entry.param_steps == {"breakout_lookback": 5.0, "volume_mult": 0.25}
    spec = entry.factory(entry.base_params)
    assert isinstance(spec, BreakoutSpec)  # narrows type + checks the factory built the right spec
    assert spec.name == "breakout"
    assert spec.breakout_lookback == 20  # rounded to int by the factory
    assert spec.volume_mult == 1.5


def test_vwap_registered_with_frozen_prereg_params() -> None:
    entry = STRATEGIES["vwap_mean_reversion"]
    assert entry.base_params == {"entry_threshold": 0.004, "exit_threshold": 0.001}
    assert entry.param_steps == {"entry_threshold": 0.001, "exit_threshold": 0.0005}


def test_vwap_cross_registered_with_frozen_prereg_params() -> None:
    # P3.1 V2 (owed cross variant): blind cross_threshold 0.002 (+/-0.001), pre-registered.
    entry = STRATEGIES["vwap_cross"]
    assert entry.base_params == {"cross_threshold": 0.002}
    assert entry.param_steps == {"cross_threshold": 0.001}
    spec = entry.factory(entry.base_params)
    assert isinstance(spec, VwapCrossSpec)  # narrows type + checks the factory built the right spec
    assert spec.name == "vwap_cross"
    assert spec.cross_threshold == 0.002


def test_mean_reversion_registered_with_frozen_prereg_params() -> None:
    # P3.3: blind z-score fade -- entry_z 2.0 (±0.5), exit_z 0.5 (±0.25), lookback 20 fixed.
    entry = STRATEGIES["mean_reversion"]
    assert entry.base_params == {"entry_z": 2.0, "exit_z": 0.5, "lookback": 20.0}
    assert entry.param_steps == {"entry_z": 0.5, "exit_z": 0.25}
    spec = entry.factory(entry.base_params)
    assert isinstance(spec, MeanReversionSpec)  # factory built the right spec
    assert spec.name == "mean_reversion"
    assert spec.entry_z == 2.0 and spec.exit_z == 0.5 and spec.lookback == 20


def test_reversal_registered_with_frozen_prereg_params() -> None:
    # P3.4: swing-failure (failed-breakout) fade -- swing_lookback 20 (±5), break_buffer 0.001.
    entry = STRATEGIES["reversal"]
    assert entry.base_params == {"swing_lookback": 20.0, "break_buffer": 0.001}
    assert entry.param_steps == {"swing_lookback": 5.0, "break_buffer": 0.0005}
    spec = entry.factory(entry.base_params)
    assert isinstance(spec, ReversalSpec)  # factory built the right spec
    assert spec.name == "reversal"
    assert spec.swing_lookback == 20 and spec.break_buffer == 0.001


def test_pivot_reversion_registered_with_frozen_prereg_params() -> None:
    # P3.5: classic pivot S/R reversion -- entry_band 0.001 (±0.0005), exit_band 0.001.
    entry = STRATEGIES["pivot_reversion"]
    assert entry.base_params == {"entry_band": 0.001, "exit_band": 0.001}
    assert entry.param_steps == {"entry_band": 0.0005, "exit_band": 0.0005}
    spec = entry.factory(entry.base_params)
    assert isinstance(spec, PivotReversionSpec)  # factory built the right spec
    assert spec.name == "pivot_reversion"
    assert spec.entry_band == 0.001 and spec.exit_band == 0.001


def test_donchian_breakout_registered_with_frozen_prereg_params() -> None:
    # P3.6: global multi-session channel breakout -- channel_lookback 55 (±10), one real knob.
    entry = STRATEGIES["donchian_breakout"]
    assert entry.base_params == {"channel_lookback": 55.0}
    assert entry.param_steps == {"channel_lookback": 10.0}
    spec = entry.factory(entry.base_params)
    assert isinstance(spec, DonchianBreakoutSpec)  # factory built the right spec
    assert spec.name == "donchian_breakout"
    assert spec.channel_lookback == 55


def test_adaptive_ma_both_variants_registered_with_frozen_prereg_params() -> None:
    # P3.7 both-owed dichotomy -- V1 fast/slow KAMA cross (proven-divergent from V2 slope),
    # V2 single-KAMA slope. Asymmetric params (different mechanisms).
    cross = STRATEGIES["adaptive_ma_cross"]
    assert cross.base_params == {"fast_period": 10.0, "slow_period": 30.0}
    assert cross.param_steps == {"fast_period": 5.0, "slow_period": 10.0}
    assert isinstance(cross.factory(cross.base_params), AdaptiveMaCrossSpec)
    assert cross.factory(cross.base_params).name == "adaptive_ma_cross"
    slope = STRATEGIES["adaptive_ma_slope"]
    assert slope.base_params == {"kama_period": 10.0} and slope.param_steps == {"kama_period": 5.0}
    assert isinstance(slope.factory(slope.base_params), AdaptiveMaSlopeSpec)
    assert slope.factory(slope.base_params).name == "adaptive_ma_slope"


def test_volatility_filters_both_registered_with_frozen_prereg_params() -> None:
    # P3.8 -- two INDEPENDENT studies (not a dichotomy): C1 expansion-breakout, C2 contraction.
    c1 = STRATEGIES["vol_expansion_breakout"]
    assert c1.base_params == {"breakout_lookback": 20.0, "atr_long": 100.0}
    assert c1.param_steps == {"breakout_lookback": 5.0, "atr_long": 20.0}
    assert isinstance(c1.factory(c1.base_params), VolExpansionBreakoutSpec)
    assert c1.factory(c1.base_params).name == "vol_expansion_breakout"
    c2 = STRATEGIES["vol_contraction_reversion"]
    assert c2.base_params == {"entry_z": 2.0, "atr_long": 100.0}
    assert c2.param_steps == {"entry_z": 0.5, "atr_long": 20.0}
    assert isinstance(c2.factory(c2.base_params), VolContractionReversionSpec)
    assert c2.factory(c2.base_params).name == "vol_contraction_reversion"


def test_every_registered_factory_builds_a_named_spec() -> None:
    for entry in STRATEGIES.values():
        spec = entry.factory(entry.base_params)
        assert spec.name  # each factory builds a named spec from its own frozen params
