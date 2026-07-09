"""The frozen strategy registry: each catalog study's spec factory + pre-committed params.

Shared by the single-symbol driver (``scripts/run_study.py``) and the panel driver
(``scripts/run_panel_study.py``) so both run the IDENTICAL frozen configuration.
``base_params`` is the frozen, pre-registered (blind) configuration; ``param_steps`` is
the +/- one-step neighbour per tunable parameter that drives criterion-6a sensitivity and
the PBO / config matrix. Every variant is charged to the trial ledger. Phase 3 adds one
entry per study; the values here must match the study's committed pre-registration.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from lab.core.interfaces import StrategySpec
from lab.research.strategies.adaptive_ma import adaptive_ma_cross_spec, adaptive_ma_slope_spec
from lab.research.strategies.breakout import breakout_spec
from lab.research.strategies.donchian_breakout import donchian_breakout_spec
from lab.research.strategies.mean_reversion import mean_reversion_spec
from lab.research.strategies.pivot_reversion import pivot_reversion_spec
from lab.research.strategies.reference import ReferenceMomentumSpec
from lab.research.strategies.reversal import reversal_spec
from lab.research.strategies.vwap import vwap_cross_spec, vwap_mean_reversion_spec

SpecFactory = Callable[[Mapping[str, float]], StrategySpec]


@dataclass(frozen=True)
class StrategyEntry:
    """A registered study: its spec factory plus the PRE-COMMITTED frozen parameters."""

    factory: SpecFactory
    base_params: Mapping[str, float]
    param_steps: Mapping[str, float]


#: Strategy registry. Parameters are the PRE-REGISTERED, frozen (blind) values -- they
#: must equal the values in the study's docs/pre_registration/ commit.
STRATEGIES: dict[str, StrategyEntry] = {
    "reference_momentum": StrategyEntry(
        factory=lambda _params: ReferenceMomentumSpec(),
        base_params={},
        param_steps={},
    ),
    "vwap_mean_reversion": StrategyEntry(
        factory=vwap_mean_reversion_spec,
        base_params={"entry_threshold": 0.004, "exit_threshold": 0.001},
        param_steps={"entry_threshold": 0.001, "exit_threshold": 0.0005},
    ),
    "vwap_cross": StrategyEntry(  # P3.1 V2 -- owed cross variant; blind params, 5-min primary
        factory=vwap_cross_spec,
        base_params={"cross_threshold": 0.002},
        param_steps={"cross_threshold": 0.001},
    ),
    "breakout": StrategyEntry(
        factory=breakout_spec,
        base_params={"breakout_lookback": 20.0, "volume_mult": 1.5},
        param_steps={"breakout_lookback": 5.0, "volume_mult": 0.25},
    ),
    "mean_reversion": StrategyEntry(  # P3.3 -- intraday-reset z-score fade; blind params
        factory=mean_reversion_spec,
        base_params={"entry_z": 2.0, "exit_z": 0.5, "lookback": 20.0},
        param_steps={"entry_z": 0.5, "exit_z": 0.25},
    ),
    "reversal": StrategyEntry(  # P3.4 -- swing-failure (failed-breakout) fade; blind params
        factory=reversal_spec,
        base_params={"swing_lookback": 20.0, "break_buffer": 0.001},
        param_steps={"swing_lookback": 5.0, "break_buffer": 0.0005},
    ),
    "pivot_reversion": StrategyEntry(  # P3.5 -- classic pivot S/R reversion; blind params
        factory=pivot_reversion_spec,
        base_params={"entry_band": 0.001, "exit_band": 0.001},
        param_steps={"entry_band": 0.0005, "exit_band": 0.0005},
    ),
    "donchian_breakout": StrategyEntry(  # P3.6 -- global multi-session channel breakout
        factory=donchian_breakout_spec,
        base_params={"channel_lookback": 55.0},
        param_steps={"channel_lookback": 10.0},
    ),
    "adaptive_ma_cross": StrategyEntry(  # P3.7 V1 -- fast/slow KAMA cross; blind
        factory=adaptive_ma_cross_spec,
        base_params={"fast_period": 10.0, "slow_period": 30.0},
        param_steps={"fast_period": 5.0, "slow_period": 10.0},
    ),
    "adaptive_ma_slope": StrategyEntry(  # P3.7 V2 -- KAMA slope (trend-state signal); blind
        factory=adaptive_ma_slope_spec,
        base_params={"kama_period": 10.0},
        param_steps={"kama_period": 5.0},
    ),
}
