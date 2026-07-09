"""Pins the frozen strategy registry: pre-registered params must match the studies.

The registry is the single source both drivers read, so this guards against the
registered frozen params drifting away from what a study's pre-registration committed.
"""

from __future__ import annotations

from lab.research.strategies.breakout import BreakoutSpec
from lab.research.strategies.registry import STRATEGIES
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


def test_every_registered_factory_builds_a_named_spec() -> None:
    for entry in STRATEGIES.values():
        spec = entry.factory(entry.base_params)
        assert spec.name  # each factory builds a named spec from its own frozen params
