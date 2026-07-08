"""Study orchestration: the composition root that runs one strategy end to end.

The validation engine (``validation/``) is deliberately strategy-agnostic — it
never imports strategy code. This module sits *above* it and wires a
:class:`~lab.core.interfaces.StrategySpec` through the whole apparatus:

    backtest -> purged CPCV -> DSR (effective-N via the ledger) -> PBO
    -> trade stats -> robustness battery -> regime buckets -> kill-gate -> report

It also runs the automated **robustness battery** (P2.5's primitives composed into
the kill-gate's criterion-6 inputs) and builds the walk-forward equity curve. One
call, one honest verdict — no metric is hand-assembled per study any more.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from zoneinfo import ZoneInfo

import numpy as np
import numpy.typing as npt

from lab.core.constants import INDIA_TZ
from lab.core.interfaces import StrategySpec
from lab.core.types import Candle, Verdict
from lab.research.reports.killgate import (
    KillGateInputs,
    KillGateResult,
    KillGateThresholds,
    evaluate_kill_gate,
)
from lab.research.reports.report import StudyReport, equity_curve, trade_statistics
from lab.research.strategies.adapter import run_strategy, signals_to_targets
from lab.research.trials.ledger import TrialLedger
from lab.research.validation.backtester import BacktestResult, Trade, run_backtest
from lab.research.validation.costs import CostModel
from lab.research.validation.cpcv import combinatorial_purged_cv
from lab.research.validation.pbo import probability_of_backtest_overfitting
from lab.research.validation.robustness import (
    fraction_positive,
    inject_ohlc_noise,
    monte_carlo_sign_flip,
    two_engines_agree,
    vectorized_backtest,
)
from lab.research.validation.sharpe import annualized_sharpe, return_stats
from lab.research.validation.splitter import ONE_TRADING_DAY

FloatArray = npt.NDArray[np.float64]

#: A factory turning a parameter mapping into a concrete spec (studies with tunable
#: parameters supply one; a parameter-free spec uses the identity default).
SpecFactory = Callable[[Mapping[str, float]], StrategySpec]
DEFAULT_NOTIONAL = 100_000.0
#: A PASS whose tightest survivorship-sensitive criterion clears by less than this
#: relative margin is stamped provisional (survivor-biased data is an upper bound).
DEFAULT_SURVIVORSHIP_BAND = 0.10


def gate_pass_margin(inputs: KillGateInputs, thresholds: KillGateThresholds) -> tuple[float, str]:
    """Tightest relative margin across the survivorship-SENSITIVE return criteria.

    Survivorship inflates realized *returns*, so the sensitive gates are the
    return/breadth ones: CPCV median path-Sharpe, profit factor, cross-symbol
    positive fraction, and the worst-case parameter-sensitivity Sharpe. (DSR and
    PBO are bounded overfitting probabilities — DSR maxes at 1.0 vs a 0.95 bar, so
    a relative margin there is always tiny and misleading — and are excluded, as
    are the boolean criteria.) Relative margin is ``(obs - thr)/thr``. Returns
    ``(margin, criterion)``; only meaningful for a PASS. ``inf`` if none apply.
    """
    candidates: list[tuple[str, float, float]] = [
        # name, observed, threshold (all higher-is-better, positive threshold)
        ("cpcv_median", inputs.cpcv_median_path_sharpe, thresholds.cpcv_median_path_sharpe_min),
        ("profit_factor", inputs.profit_factor, thresholds.profit_factor_min),
        (
            "cross_symbol",
            inputs.cross_symbol_positive_fraction,
            thresholds.cross_symbol_positive_fraction_min,
        ),
        (
            "param_sensitivity",
            inputs.param_sensitivity_min_net_sharpe,
            thresholds.param_sensitivity_net_sharpe_min,
        ),
    ]
    tightest = float("inf")
    tightest_name = ""
    for name, observed, threshold in candidates:
        if not np.isfinite(observed) or threshold <= 0.0:
            continue
        margin = (observed - threshold) / threshold
        if margin < tightest:
            tightest, tightest_name = margin, name
    return tightest, tightest_name


def survivorship_stamp(
    verdict: Verdict,
    inputs: KillGateInputs,
    thresholds: KillGateThresholds,
    band: float = DEFAULT_SURVIVORSHIP_BAND,
) -> tuple[bool, str]:
    """Return ``(provisional, note)`` for a study on survivor-biased data.

    Only a PASS can be provisional, and only when its tightest survivorship-sensitive
    criterion clears by less than ``band`` (relative) — the sole place a small upward
    bias could flip the verdict. Wide-margin passes and KILLs are never stamped.
    """
    if verdict is not Verdict.PASS:
        return False, ""
    margin, criterion = gate_pass_margin(inputs, thresholds)
    if np.isfinite(margin) and margin < band:
        return True, (
            f"narrow pass ({criterion} clears its gate by {margin:.0%}); on survivor-only "
            f"data this is an upper bound (see RESEARCH_FINDINGS 2.1)"
        )
    return False, ""


# --- parameter-variant runs (shared by PBO, the ledger, and param sensitivity) - #
def enumerate_param_configs(
    base_params: Mapping[str, float], param_steps: Mapping[str, float]
) -> list[tuple[str, dict[str, float]]]:
    """The base config plus a +/-one-step neighbour for each tunable parameter."""
    configs: list[tuple[str, dict[str, float]]] = [("base", dict(base_params))]
    for name, step in param_steps.items():
        base_value = float(base_params[name])
        for sign, tag in ((1.0, "+"), (-1.0, "-")):
            variant = dict(base_params)
            variant[name] = base_value + sign * step
            configs.append((f"{name}{tag}", variant))
    return configs


def run_param_configs(
    spec_factory: SpecFactory,
    base_params: Mapping[str, float],
    param_steps: Mapping[str, float],
    candles: Sequence[Candle],
    cost_model: CostModel,
    *,
    notional_per_trade: float = DEFAULT_NOTIONAL,
    timezone: str = INDIA_TZ,
) -> dict[str, FloatArray]:
    """Run each parameter config once; return ``label -> net-return stream``."""
    streams: dict[str, FloatArray] = {}
    for label, params in enumerate_param_configs(base_params, param_steps):
        result = run_strategy(
            spec_factory(params),
            candles,
            cost_model,
            notional_per_trade=notional_per_trade,
            timezone=timezone,
        )
        streams[label] = result.net_returns
    return streams


# --- robustness battery (kill-gate criterion 6) ----------------------------- #
@dataclass(frozen=True, slots=True)
class RobustnessReport:
    """The measured robustness inputs the kill-gate's criterion 6 consumes."""

    param_sensitivity_min_net_sharpe: float
    mc_shuffle_beat_fraction: float
    cross_symbol_positive_fraction: float
    two_engines_reconcile: bool
    noise_survives: bool
    param_net_sharpes: tuple[float, ...]
    cross_symbol_net_sharpes: tuple[float, ...]
    noise_net_sharpes: tuple[float, ...]


def run_robustness_battery(
    spec_factory: SpecFactory,
    base_params: Mapping[str, float],
    param_steps: Mapping[str, float],
    candles: Sequence[Candle],
    cost_model: CostModel,
    *,
    periods_per_year: float,
    cross_symbol_candles: Mapping[str, Sequence[Candle]] | None = None,
    config_streams: Mapping[str, FloatArray] | None = None,
    notional_per_trade: float = DEFAULT_NOTIONAL,
    noise_seeds: int = 8,
    noise_scale: float = 0.0005,
    mc_shuffles: int = 1000,
    mc_seed: int = 0,
    timezone: str = INDIA_TZ,
) -> RobustnessReport:
    """Compose the P2.5 primitives into the kill-gate's criterion-6 inputs.

    Parameter sensitivity re-runs the strategy at +/-one step of every parameter
    and takes the WORST net Sharpe; noise survival re-runs on OHLC-perturbed bars
    and asks the median edge to stay positive; cross-symbol asks a majority of
    held-out symbols to be net-positive; plus the MC sign-flip and the two-engine
    reconciliation. ``config_streams`` may be passed in to avoid re-running the
    parameter variants (the orchestrator shares them with PBO and the ledger).
    """
    streams = (
        dict(config_streams)
        if config_streams is not None
        else run_param_configs(
            spec_factory,
            base_params,
            param_steps,
            candles,
            cost_model,
            notional_per_trade=notional_per_trade,
            timezone=timezone,
        )
    )
    param_sharpes = tuple(annualized_sharpe(s, periods_per_year) for s in streams.values())
    param_min = min((s for s in param_sharpes if np.isfinite(s)), default=float("nan"))

    base_net = streams["base"]
    mc_beat = monte_carlo_sign_flip(base_net, n_shuffles=mc_shuffles, seed=mc_seed)

    # Two-engine reconciliation on the base config's identical target series.
    base_spec = spec_factory(base_params)
    targets = signals_to_targets(candles, base_spec.generate_signals(candles), timezone=timezone)
    event_driven = run_backtest(
        candles, targets, cost_model, notional_per_trade=notional_per_trade, timezone=timezone
    )
    vectorized = vectorized_backtest(
        candles, targets, cost_model, notional_per_trade=notional_per_trade, timezone=timezone
    )
    reconcile = two_engines_agree(event_driven, vectorized)

    # Noise survival: the edge must persist through realistic OHLC perturbation.
    noise_sharpes = tuple(
        annualized_sharpe(
            run_strategy(
                base_spec,
                inject_ohlc_noise(candles, relative_scale=noise_scale, seed=seed),
                cost_model,
                notional_per_trade=notional_per_trade,
                timezone=timezone,
            ).net_returns,
            periods_per_year,
        )
        for seed in range(noise_seeds)
    )
    finite_noise = [s for s in noise_sharpes if np.isfinite(s)]
    noise_survives = bool(finite_noise) and float(np.median(finite_noise)) > 0.0

    # Cross-symbol: net-positive on a majority of held-out symbols.
    cross_sharpes = tuple(
        annualized_sharpe(
            run_strategy(
                base_spec,
                symbol_candles,
                cost_model,
                notional_per_trade=notional_per_trade,
                timezone=timezone,
            ).net_returns,
            periods_per_year,
        )
        for symbol_candles in (cross_symbol_candles or {}).values()
    )
    cross_fraction = fraction_positive(cross_sharpes) if cross_sharpes else float("nan")

    return RobustnessReport(
        param_sensitivity_min_net_sharpe=param_min,
        mc_shuffle_beat_fraction=mc_beat,
        cross_symbol_positive_fraction=cross_fraction,
        two_engines_reconcile=reconcile,
        noise_survives=noise_survives,
        param_net_sharpes=param_sharpes,
        cross_symbol_net_sharpes=cross_sharpes,
        noise_net_sharpes=noise_sharpes,
    )


# --- regime stability (kill-gate criterion 7) ------------------------------- #
def _time_of_day_bucket(trade: Trade, timezone: str) -> str:
    """Fixed, pre-defined session buckets (morning / midday / afternoon, IST)."""
    hour = trade.entry_time.astimezone(ZoneInfo(timezone)).hour
    if hour < 11:
        return "morning"
    if hour < 13:
        return "midday"
    return "afternoon"


def regime_bucket_stats(
    trades: Sequence[Trade],
    periods_per_year: float,
    *,
    labeler: Callable[[Trade], str] | None = None,
    timezone: str = INDIA_TZ,
) -> tuple[tuple[float, ...], bool]:
    """Per-bucket annualized net Sharpe, and whether the edge survives dropping the best bucket.

    Buckets are fixed before the run (default: session thirds). Returns the Sharpe
    of every occupied bucket (a thin bucket scores NaN, which fails the gate — you
    cannot certify a regime you barely traded) and ``positive_without_best``.
    """
    label_of = labeler or (lambda t: _time_of_day_bucket(t, timezone))
    buckets: dict[str, list[Trade]] = {}
    for trade in trades:
        buckets.setdefault(label_of(trade), []).append(trade)
    if not buckets:
        return (), False

    medians = tuple(
        annualized_sharpe([t.net_return for t in bucket_trades], periods_per_year)
        for bucket_trades in buckets.values()
    )
    best_label = max(buckets, key=lambda k: sum(t.net_return for t in buckets[k]))
    without_best = [
        t.net_return for label, ts in buckets.items() if label != best_label for t in ts
    ]
    positive_without_best = (
        len(without_best) >= 2 and annualized_sharpe(without_best, periods_per_year) > 0.0
    )
    return medians, positive_without_best


# --- the orchestrator ------------------------------------------------------- #
def run_study(
    spec: StrategySpec,
    candles: Sequence[Candle],
    cost_model: CostModel,
    thresholds: KillGateThresholds,
    ledger: TrialLedger,
    *,
    periods_per_year: float,
    n_groups: int = 6,
    k_test_groups: int = 2,
    cpcv_embargo: timedelta = ONE_TRADING_DAY,
    pbo_splits: int = 8,
    notional_per_trade: float = DEFAULT_NOTIONAL,
    spec_factory: SpecFactory | None = None,
    base_params: Mapping[str, float] | None = None,
    param_steps: Mapping[str, float] | None = None,
    cross_symbol_candles: Mapping[str, Sequence[Candle]] | None = None,
    regime_labeler: Callable[[Trade], str] | None = None,
    log_trials: bool = True,
    survivorship_band: float = DEFAULT_SURVIVORSHIP_BAND,
    timezone: str = INDIA_TZ,
) -> StudyReport:
    """Run one strategy through the whole harness and return its :class:`StudyReport`.

    A parameter-free strategy may omit ``spec_factory``/``base_params``/``param_steps``
    (the spec is used directly and parameter sensitivity is vacuous). Strategies
    with tunables supply a factory and a +/-step per parameter, which drives the
    parameter-sensitivity leg, the PBO configuration matrix, and the ledger's
    effective-trial deflation of the DSR.
    """
    factory: SpecFactory = spec_factory or (lambda _params: spec)
    params: Mapping[str, float] = base_params or {}
    steps: Mapping[str, float] = param_steps or {}

    # Base backtest -> trades, returns, statistics, walk-forward equity.
    result: BacktestResult = run_strategy(
        factory(params),
        candles,
        cost_model,
        notional_per_trade=notional_per_trade,
        timezone=timezone,
    )
    trades = result.trades
    net = result.net_returns
    stats = trade_statistics(trades)
    equity = equity_curve(trades)

    # Purged, embargoed CPCV path-Sharpe distribution.
    cpcv = combinatorial_purged_cv(
        net,
        result.entry_times,
        result.exit_times,
        n_groups=n_groups,
        k_test_groups=k_test_groups,
        periods_per_year=periods_per_year,
        embargo=cpcv_embargo,
    )

    # Parameter-variant streams: shared by PBO, the ledger, and param sensitivity.
    config_streams = run_param_configs(
        factory,
        params,
        steps,
        candles,
        cost_model,
        notional_per_trade=notional_per_trade,
        timezone=timezone,
    )

    # Ledger: log every config as a trial so the DSR deflates by the EFFECTIVE
    # (cluster-adjusted) trial count — a one-parameter sweep is ~1 effective trial.
    if log_trials:
        for label, stream in config_streams.items():
            ledger.log_trial(spec.name, {"config": label}, stream.tolist())
    moments = return_stats(net)
    dsr = ledger.deflated_sharpe(moments.sharpe, moments.n, moments.skew, moments.kurtosis)

    # PBO across the parameter configs (needs >= 2 configs and enough periods).
    pbo = _pbo_across_configs(config_streams, n_splits=pbo_splits)

    # Robustness battery + regime stability.
    robustness = run_robustness_battery(
        factory,
        params,
        steps,
        candles,
        cost_model,
        periods_per_year=periods_per_year,
        cross_symbol_candles=cross_symbol_candles,
        config_streams=config_streams,
        notional_per_trade=notional_per_trade,
        timezone=timezone,
    )
    regime_medians, regime_without_best = regime_bucket_stats(
        trades, periods_per_year, labeler=regime_labeler, timezone=timezone
    )

    inputs = KillGateInputs(
        cpcv_median_path_sharpe=cpcv.median_path_sharpe,
        cpcv_positive_fraction=cpcv.positive_fraction,
        cpcv_tenth_percentile=cpcv.tenth_percentile,
        dsr=dsr,
        pbo=pbo,
        profit_factor=stats.profit_factor,
        top5_winners_fraction=stats.top5_winners_fraction,
        expectancy=stats.expectancy,
        round_trip_cost=cost_model.round_trip_cost_fraction(notional_per_trade),
        param_sensitivity_min_net_sharpe=robustness.param_sensitivity_min_net_sharpe,
        mc_shuffle_beat_fraction=robustness.mc_shuffle_beat_fraction,
        cross_symbol_positive_fraction=robustness.cross_symbol_positive_fraction,
        two_engines_reconcile=robustness.two_engines_reconcile,
        noise_survives=robustness.noise_survives,
        regime_bucket_medians=regime_medians,
        regime_positive_without_best=regime_without_best,
    )
    kill_gate: KillGateResult = evaluate_kill_gate(inputs, thresholds)
    provisional, provisional_note = survivorship_stamp(
        kill_gate.verdict, inputs, thresholds, survivorship_band
    )

    return StudyReport(
        strategy=spec.name,
        cpcv=cpcv,
        dsr=dsr,
        pbo=pbo,
        effective_trials=ledger.effective_trials(),
        trades=stats,
        kill_gate=kill_gate,
        equity=equity,
        provisional=provisional,
        provisional_note=provisional_note,
    )


def _pbo_across_configs(config_streams: Mapping[str, FloatArray], *, n_splits: int) -> float:
    """PBO over the parameter-config performance matrix; NaN if it cannot be formed."""
    streams = [s for s in config_streams.values() if s.size > 0]
    if len(streams) < 2:
        return float("nan")  # a single-config strategy has no cross-config overfitting to measure
    length = min(s.size for s in streams)
    if length < n_splits:
        return float("nan")
    matrix = np.column_stack([s[:length] for s in streams])
    return probability_of_backtest_overfitting(matrix, n_splits=n_splits).pbo
