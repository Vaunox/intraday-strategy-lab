"""The seven-point kill-gate (Phase 2, P2.6).

Evaluates a study's measured statistics against the single pre-committed
thresholds in ``config/killgate.yaml`` (never a range, never tuned to pass —
Inviolable Rule 1). Fail any one criterion and the verdict is KILL. NaN metrics
fail their criterion (you cannot pass on a missing measurement).

**Stub guard (Phase-3 readiness B-1).** A NaN fails closed, but a *plausible
looking* number does not — so the gate would otherwise grade a hand-passed stub
(e.g. a single fabricated regime bucket ``(1.0,)``) as if it were real. To make
the gate refuse a stub, the criteria whose inputs are collections — 6a parameter
sweep, 6d cross-symbol holdout, 7 regime partition — arrive as the **keyed
evidence the machinery emits**, and the gate verifies their SHAPE (distinct-key
cardinality and identity, never magnitude) before grading. Under-shaped evidence
yields :attr:`~lab.core.types.Verdict.INSUFFICIENT` — a recorded "could not
certify" that is neither PASS nor KILL. The graded summaries (worst sweep-Sharpe,
cross-symbol positive fraction, regime medians) are DERIVED from those
collections here, so the number the gate grades is provably the machinery's, not
a caller's scalar. Because the check is on shape not value, a legitimately
extreme-but-real result (a Sharpe of 1.0, a strategy strong in every bucket) is
never rejected.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from lab.core.types import Verdict
from lab.research.validation.cpcv import cpcv_distribution_summary


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
    # Evidence-provenance floors (B-1 stub guard): STRUCTURAL minimums — distinct
    # buckets/symbols, never value thresholds. Under-shaped evidence -> INSUFFICIENT.
    min_regime_buckets: int
    min_cross_symbols: int
    min_cpcv_paths: int

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> KillGateThresholds:
        """Build thresholds from the parsed ``killgate.yaml`` mapping."""
        robustness = mapping["robustness"]
        regime = mapping["regime"]
        evidence = mapping["evidence"]
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
            min_regime_buckets=int(evidence["min_regime_buckets"]),
            min_cross_symbols=int(evidence["min_cross_symbols"]),
            min_cpcv_paths=int(evidence["min_cpcv_paths"]),
        )


def load_kill_gate_thresholds(config_dir: Path) -> KillGateThresholds:
    """Load the kill-gate thresholds from ``config_dir/killgate.yaml``."""
    data: Any = yaml.safe_load((config_dir / "killgate.yaml").read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("killgate.yaml must contain a mapping")
    return KillGateThresholds.from_mapping(data)


@dataclass(frozen=True, slots=True)
class KillGateInputs:
    """The measured statistics of a study, fed to the kill-gate.

    Criteria whose inputs are collections — 6a parameter sweep, 6d cross-symbol
    holdout, 7 regime partition — carry the machinery's **keyed evidence**, not a
    summary scalar, so the gate can verify their shape (B-1) and derive the graded
    summary from them. The CPCV-fed criteria (1, 4) and the single-value robustness
    legs (6b MC-shuffle, 6c noise, 6e two-engine) remain summary scalars/flags.
    """

    # criteria 1 & 4 — CPCV distribution as keyed evidence: the (post-purge) path
    # Sharpes themselves. The gate derives the median / positive-fraction / 10th-pct
    # from them and checks the finite-path count, so no CPCV summary can be a stub.
    cpcv_path_sharpes: tuple[float, ...]
    # criterion 2 / 3
    dsr: float
    pbo: float
    # criterion 5 — P&L concentration
    profit_factor: float
    top5_winners_fraction: float
    expectancy: float
    round_trip_cost: float
    # criterion 6 — robustness: 6a/6d as keyed evidence, 6b/6c/6e as computed scalars/flags
    param_config_sharpes: Mapping[str, float]  # 6a: config-label -> net Sharpe (incl. "base")
    has_tunable_params: bool  # 6a: whether the strategy declares any tunable parameter
    cross_symbol_sharpes: Mapping[str, float]  # 6d: held-out symbol -> net Sharpe
    primary_symbol: str  # 6d: the scored symbol (must not appear among the holdouts)
    mc_shuffle_beat_fraction: float
    two_engines_reconcile: bool
    noise_survives: bool
    # criterion 7 — regime stability (keyed evidence)
    regime_bucket_sharpes: Mapping[str, float]  # bucket-label -> net Sharpe
    regime_positive_without_best: bool

    @property
    def cpcv_median_path_sharpe(self) -> float:
        """Median finite CPCV path-Sharpe (criterion 1), derived from the distribution."""
        return cpcv_distribution_summary(self.cpcv_path_sharpes).median_path_sharpe

    @property
    def cpcv_positive_fraction(self) -> float:
        """Fraction of finite CPCV paths that are positive (criterion 4a)."""
        return cpcv_distribution_summary(self.cpcv_path_sharpes).positive_fraction

    @property
    def cpcv_tenth_percentile(self) -> float:
        """10th-percentile finite CPCV path-Sharpe (criterion 4b)."""
        return cpcv_distribution_summary(self.cpcv_path_sharpes).tenth_percentile

    @property
    def param_sensitivity_min_net_sharpe(self) -> float:
        """Worst (min) net Sharpe across the parameter sweep (criterion 6a)."""
        finite = [s for s in self.param_config_sharpes.values() if math.isfinite(s)]
        return min(finite) if finite else float("nan")

    @property
    def cross_symbol_positive_fraction(self) -> float:
        """Fraction of held-out symbols with a positive net Sharpe (criterion 6d)."""
        finite = [s for s in self.cross_symbol_sharpes.values() if math.isfinite(s)]
        return (sum(s > 0.0 for s in finite) / len(finite)) if finite else float("nan")

    @property
    def regime_bucket_medians(self) -> tuple[float, ...]:
        """Per-bucket net Sharpes, in bucket order (criterion 7)."""
        return tuple(self.regime_bucket_sharpes.values())


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
        """Whether every criterion passed (verdict == PASS).

        A non-PASS verdict — KILL *or* INSUFFICIENT — is not passed, so any caller
        gating on ``passed`` treats an un-certifiable (stub) study as not green.
        """
        return self.verdict is Verdict.PASS


def _evidence_failures(inputs: KillGateInputs, thresholds: KillGateThresholds) -> list[Criterion]:
    """Return the criteria whose evidence is under-shaped (a stub), else ``[]``.

    Structural only — distinct-key cardinality and identity, never a value — so a
    legitimately extreme-but-real result is never flagged; only evidence that
    could not have come from the machinery is. A parameter-free strategy
    (``has_tunable_params`` false) legitimately has a single ``base`` config and
    is not a stub.
    """
    failures: list[Criterion] = []

    cpcv = cpcv_distribution_summary(inputs.cpcv_path_sharpes)
    if cpcv.n_finite_paths < thresholds.min_cpcv_paths:
        failures.append(
            Criterion(
                1,
                "CPCV distribution evidence",
                False,
                f"got {cpcv.n_finite_paths} finite post-purge paths of "
                f"{len(inputs.cpcv_path_sharpes)} (need >= {thresholds.min_cpcv_paths}) — "
                "this is a stub, not a path-Sharpe distribution",
            )
        )

    params = inputs.param_config_sharpes
    if "base" not in params or (inputs.has_tunable_params and len(params) < 3):
        failures.append(
            Criterion(
                6,
                "parameter-sensitivity evidence",
                False,
                f"got {len(params)} configs {sorted(params)}; a tunable strategy needs 'base' "
                "plus a +/- step per parameter (>= 3) — this is a stub, not a sweep",
            )
        )

    cross = inputs.cross_symbol_sharpes
    if len(cross) < thresholds.min_cross_symbols or inputs.primary_symbol in cross:
        failures.append(
            Criterion(
                6,
                "cross-symbol evidence",
                False,
                f"got {len(cross)} held-out symbols {sorted(cross)} (need "
                f">= {thresholds.min_cross_symbols} distinct, none == primary "
                f"{inputs.primary_symbol!r}) — this is a stub, not a holdout run",
            )
        )

    regime = inputs.regime_bucket_sharpes
    if len(regime) < thresholds.min_regime_buckets:
        failures.append(
            Criterion(
                7,
                "regime evidence",
                False,
                f"got {len(regime)} regime buckets {sorted(regime)} (need "
                f">= {thresholds.min_regime_buckets}) — this is a stub, not a partition",
            )
        )
    return failures


def evaluate_kill_gate(inputs: KillGateInputs, thresholds: KillGateThresholds) -> KillGateResult:
    """Evaluate the seven-point kill-gate; KILL if any criterion fails.

    First verifies each collection-valued criterion's evidence is genuinely shaped
    (B-1). If any is a stub, returns ``INSUFFICIENT`` — a recorded "could not
    certify", neither PASS nor KILL — rather than grading a placeholder as real.
    """
    insufficient = _evidence_failures(inputs, thresholds)
    if insufficient:
        return KillGateResult(criteria=tuple(insufficient), verdict=Verdict.INSUFFICIENT)

    t = thresholds
    i = inputs

    regime_medians = i.regime_bucket_medians
    regime_all = bool(np.all(np.array(regime_medians) > t.regime_median_min_every_bucket))
    regime_majority = float(np.mean(np.array(regime_medians) > t.regime_median_majority_min)) > 0.5

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
