"""The seven-point kill-gate (Phase 2, P2.6).

Evaluates a study's measured statistics against the single pre-committed
thresholds in ``config/killgate.yaml`` (never a range, never tuned to pass —
Inviolable Rule 1). Fail any one criterion and the verdict is KILL. NaN metrics
fail their criterion (you cannot pass on a missing measurement).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from lab.core.types import Verdict


@dataclass(frozen=True, slots=True)
class KillGateThresholds:
    """The pinned kill-gate thresholds (loaded from config/killgate.yaml)."""

    cpcv_median_path_sharpe_min: float
    dsr_min: float
    pbo_max: float
    cpcv_positive_paths_fraction_min: float
    cpcv_10th_percentile_sharpe_min: float
    profit_factor_min: float
    top5_winners_fraction_max: float
    require_expectancy_over_cost: bool
    param_sensitivity_net_sharpe_min: float
    mc_shuffle_beat_fraction_min: float
    cross_symbol_positive_fraction_min: float
    two_engine_tolerance: float
    regime_median_min_every_bucket: float
    regime_median_majority_min: float
    regime_require_positive_without_best: bool

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> KillGateThresholds:
        """Build thresholds from the parsed ``killgate.yaml`` mapping."""
        robustness = mapping["robustness"]
        regime = mapping["regime"]
        return cls(
            cpcv_median_path_sharpe_min=float(mapping["cpcv_median_path_sharpe_min"]),
            dsr_min=float(mapping["dsr_min"]),
            pbo_max=float(mapping["pbo_max"]),
            cpcv_positive_paths_fraction_min=float(mapping["cpcv_positive_paths_fraction_min"]),
            cpcv_10th_percentile_sharpe_min=float(mapping["cpcv_10th_percentile_sharpe_min"]),
            profit_factor_min=float(mapping["profit_factor_min"]),
            top5_winners_fraction_max=float(mapping["top5_winners_fraction_max"]),
            require_expectancy_over_cost=bool(mapping["require_expectancy_over_cost"]),
            param_sensitivity_net_sharpe_min=float(robustness["param_sensitivity_net_sharpe_min"]),
            mc_shuffle_beat_fraction_min=float(robustness["mc_shuffle_beat_fraction_min"]),
            cross_symbol_positive_fraction_min=float(
                robustness["cross_symbol_positive_fraction_min"]
            ),
            two_engine_tolerance=float(robustness["two_engine_tolerance"]),
            regime_median_min_every_bucket=float(regime["median_path_sharpe_min_every_bucket"]),
            regime_median_majority_min=float(regime["median_path_sharpe_majority_min"]),
            regime_require_positive_without_best=bool(
                regime["require_positive_without_best_bucket"]
            ),
        )


def load_kill_gate_thresholds(config_dir: Path) -> KillGateThresholds:
    """Load the kill-gate thresholds from ``config_dir/killgate.yaml``."""
    data: Any = yaml.safe_load((config_dir / "killgate.yaml").read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("killgate.yaml must contain a mapping")
    return KillGateThresholds.from_mapping(data)


@dataclass(frozen=True, slots=True)
class KillGateInputs:
    """The measured statistics of a study, fed to the kill-gate."""

    cpcv_median_path_sharpe: float
    cpcv_positive_fraction: float
    cpcv_tenth_percentile: float
    dsr: float
    pbo: float
    profit_factor: float
    top5_winners_fraction: float
    expectancy: float
    round_trip_cost: float
    param_sensitivity_min_net_sharpe: float
    mc_shuffle_beat_fraction: float
    cross_symbol_positive_fraction: float
    two_engines_reconcile: bool
    noise_survives: bool
    regime_bucket_medians: tuple[float, ...]
    regime_positive_without_best: bool


@dataclass(frozen=True, slots=True)
class Criterion:
    """One kill-gate criterion's pass/fail with a human-readable detail."""

    number: int
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class KillGateResult:
    """The seven criteria and the overall verdict."""

    criteria: tuple[Criterion, ...]
    verdict: Verdict

    @property
    def passed(self) -> bool:
        """Whether every criterion passed (verdict == PASS)."""
        return self.verdict is Verdict.PASS


def evaluate_kill_gate(inputs: KillGateInputs, thresholds: KillGateThresholds) -> KillGateResult:
    """Evaluate the seven-point kill-gate; KILL if any criterion fails."""
    t = thresholds
    i = inputs

    regime_all = bool(np.all(np.array(i.regime_bucket_medians) > t.regime_median_min_every_bucket))
    regime_majority = (
        float(np.mean(np.array(i.regime_bucket_medians) > t.regime_median_majority_min)) > 0.5
        if i.regime_bucket_medians
        else False
    )

    criteria = (
        Criterion(
            1,
            "CPCV median path-Sharpe",
            i.cpcv_median_path_sharpe > t.cpcv_median_path_sharpe_min,
            f"{i.cpcv_median_path_sharpe:.3f} > {t.cpcv_median_path_sharpe_min}",
        ),
        Criterion(
            2,
            "Deflated Sharpe Ratio",
            i.dsr >= t.dsr_min,
            f"{i.dsr:.3f} >= {t.dsr_min}",
        ),
        Criterion(
            3,
            "PBO",
            i.pbo < t.pbo_max,
            f"{i.pbo:.3f} < {t.pbo_max}",
        ),
        Criterion(
            4,
            "CPCV distribution positive & narrow",
            i.cpcv_positive_fraction >= t.cpcv_positive_paths_fraction_min
            and i.cpcv_tenth_percentile >= t.cpcv_10th_percentile_sharpe_min,
            f"positive={i.cpcv_positive_fraction:.2f}>={t.cpcv_positive_paths_fraction_min}, "
            f"p10={i.cpcv_tenth_percentile:.3f}>={t.cpcv_10th_percentile_sharpe_min}",
        ),
        Criterion(
            5,
            "P&L not concentrated",
            i.profit_factor >= t.profit_factor_min
            and i.top5_winners_fraction < t.top5_winners_fraction_max
            and (not t.require_expectancy_over_cost or i.expectancy > i.round_trip_cost),
            f"PF={i.profit_factor:.2f}>={t.profit_factor_min}, "
            f"top5={i.top5_winners_fraction:.2f}<{t.top5_winners_fraction_max}, "
            f"expectancy={i.expectancy:.5f} vs cost={i.round_trip_cost:.5f}",
        ),
        Criterion(
            6,
            "Robustness battery",
            i.param_sensitivity_min_net_sharpe > t.param_sensitivity_net_sharpe_min
            and i.mc_shuffle_beat_fraction >= t.mc_shuffle_beat_fraction_min
            and i.cross_symbol_positive_fraction >= t.cross_symbol_positive_fraction_min
            and i.two_engines_reconcile
            and i.noise_survives,
            f"param={i.param_sensitivity_min_net_sharpe:.2f}, mc={i.mc_shuffle_beat_fraction:.2f}, "
            f"cross_sym={i.cross_symbol_positive_fraction:.2f}, "
            f"two_engine={i.two_engines_reconcile}, noise={i.noise_survives}",
        ),
        Criterion(
            7,
            "Regime stability",
            regime_all
            and regime_majority
            and (not t.regime_require_positive_without_best or i.regime_positive_without_best),
            f"all_buckets>{t.regime_median_min_every_bucket}={regime_all}, "
            f"majority>{t.regime_median_majority_min}={regime_majority}, "
            f"drop_best_positive={i.regime_positive_without_best}",
        ),
    )
    verdict = Verdict.PASS if all(c.passed for c in criteria) else Verdict.KILL
    return KillGateResult(criteria=criteria, verdict=verdict)
