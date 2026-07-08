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
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np

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
) -> dict[str, BacktestResult]:
    """Run each parameter config once; return ``label -> BacktestResult``.

    The full result (not just the return array) is the single source shared by the
    ledger (per-trade returns), parameter sensitivity (net Sharpe), and PBO (which
    needs each trade's entry time to align configs on a common daily grid, B-2).
    """
    results: dict[str, BacktestResult] = {}
    for label, params in enumerate_param_configs(base_params, param_steps):
        results[label] = run_strategy(
            spec_factory(params),
            candles,
            cost_model,
            notional_per_trade=notional_per_trade,
            timezone=timezone,
        )
    return results


# --- robustness battery (kill-gate criterion 6) ----------------------------- #
@dataclass(frozen=True, slots=True)
class RobustnessReport:
    """The measured robustness inputs the kill-gate's criterion 6 consumes.

    ``param_config_sharpes`` (6a) and ``cross_symbol_sharpes`` (6d) are the KEYED
    evidence the kill-gate's stub guard checks (B-1): the gate derives the worst
    sweep-Sharpe and the cross-symbol positive fraction from them and rejects an
    under-shaped stub. Keys are the config label and the held-out symbol.
    """

    param_sensitivity_min_net_sharpe: float
    mc_shuffle_beat_fraction: float
    cross_symbol_positive_fraction: float
    two_engines_reconcile: bool
    noise_survives: bool
    param_config_sharpes: Mapping[str, float]
    cross_symbol_sharpes: Mapping[str, float]
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
    config_results: Mapping[str, BacktestResult] | None = None,
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
    reconciliation. ``config_results`` may be passed in to avoid re-running the
    parameter variants (the orchestrator shares them with PBO and the ledger).
    """
    results = (
        dict(config_results)
        if config_results is not None
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
    param_config_sharpes = {
        label: annualized_sharpe(r.net_returns, periods_per_year) for label, r in results.items()
    }
    param_min = min(
        (s for s in param_config_sharpes.values() if np.isfinite(s)), default=float("nan")
    )

    base_net = results["base"].net_returns
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

    # Cross-symbol: net-positive on a majority of held-out symbols (keyed by symbol
    # so the kill-gate can verify distinct held-out identities, not just a count).
    cross_symbol_sharpes = {
        symbol: annualized_sharpe(
            run_strategy(
                base_spec,
                symbol_candles,
                cost_model,
                notional_per_trade=notional_per_trade,
                timezone=timezone,
            ).net_returns,
            periods_per_year,
        )
        for symbol, symbol_candles in (cross_symbol_candles or {}).items()
    }
    cross_fraction = (
        fraction_positive(tuple(cross_symbol_sharpes.values()))
        if cross_symbol_sharpes
        else float("nan")
    )

    return RobustnessReport(
        param_sensitivity_min_net_sharpe=param_min,
        mc_shuffle_beat_fraction=mc_beat,
        cross_symbol_positive_fraction=cross_fraction,
        two_engines_reconcile=reconcile,
        noise_survives=noise_survives,
        param_config_sharpes=param_config_sharpes,
        cross_symbol_sharpes=cross_symbol_sharpes,
        noise_net_sharpes=noise_sharpes,
    )


# --- regime stability (kill-gate criterion 7) ------------------------------- #
def _year_bucket(trade: Trade, timezone: str) -> str:
    """Self-contained fallback bucket (calendar year of entry).

    Used only when :func:`regime_bucket_stats` is called without a labeler; the
    orchestrator (:func:`run_study`) supplies the full year x vol/trend labeler
    from :func:`build_regime_labeler` by default (B-4).
    """
    return str(trade.entry_time.astimezone(ZoneInfo(timezone)).year)


def build_regime_labeler(
    candles: Sequence[Candle], *, window: int = 20, timezone: str = INDIA_TZ
) -> Callable[[Trade], str]:
    """Build criterion 7's default labeler: ``year | vol-regime | trend-regime`` (B-4).

    Blueprint criterion 7 partitions by year AND by volatility/trend regime (not
    time-of-day). Each bar is tagged from CAUSAL context — trailing realized vol
    over ``window`` bars and the sign of the trailing ``window``-bar return — with
    the high/low vol split at the sample median of that trailing vol (an
    analysis-time partition of realized results, fixed before grading, never a
    signal input). A trade inherits its entry bar's regime; an unmatched entry
    falls back to the year alone.
    """
    tz = ZoneInfo(timezone)
    timestamps = [c.timestamp for c in candles]
    close = np.array([c.close for c in candles], dtype=np.float64)
    log_ret = np.diff(np.log(close), prepend=np.log(close[:1])) if close.size else close
    trailing_vol = np.full(close.size, np.nan, dtype=np.float64)
    for k in range(close.size):
        lo = max(0, k - window + 1)
        if k - lo >= 1:
            trailing_vol[k] = float(np.std(log_ret[lo : k + 1]))
    median_vol = (
        float(np.nanmedian(trailing_vol))
        if bool(np.any(np.isfinite(trailing_vol)))
        else float("nan")
    )

    regime_by_ts: dict[datetime, str] = {}
    for k, ts in enumerate(timestamps):
        year = ts.astimezone(tz).year
        vol = trailing_vol[k]
        high_vol = bool(np.isfinite(vol) and np.isfinite(median_vol) and vol >= median_vol)
        base = k - window
        up = base >= 0 and close[k] >= close[base]
        regime_by_ts[ts] = f"{year}|{'hivol' if high_vol else 'lovol'}|{'up' if up else 'down'}"

    def label(trade: Trade) -> str:
        return regime_by_ts.get(trade.entry_time, f"{trade.entry_time.astimezone(tz).year}|unknown")

    return label


def regime_bucket_stats(
    trades: Sequence[Trade],
    periods_per_year: float,
    *,
    labeler: Callable[[Trade], str] | None = None,
    timezone: str = INDIA_TZ,
) -> tuple[dict[str, float], bool]:
    """Per-bucket annualized net Sharpe (keyed by bucket), and drop-the-best survival.

    Returns ``{bucket_label: net Sharpe}`` for every occupied bucket (a thin bucket
    scores NaN, which fails the gate — you cannot certify a regime you barely
    traded) and ``positive_without_best``. The keyed shape is criterion 7's
    evidence: the kill-gate's stub guard rejects a partition with too few distinct
    buckets (B-1). ``labeler`` defaults to the calendar year; the orchestrator
    passes the full year x vol/trend labeler (:func:`build_regime_labeler`).
    """
    label_of = labeler or (lambda t: _year_bucket(t, timezone))
    buckets: dict[str, list[Trade]] = {}
    for trade in trades:
        buckets.setdefault(label_of(trade), []).append(trade)
    if not buckets:
        return {}, False

    bucket_sharpes = {
        label: annualized_sharpe([t.net_return for t in bucket_trades], periods_per_year)
        for label, bucket_trades in buckets.items()
    }
    best_label = max(buckets, key=lambda k: sum(t.net_return for t in buckets[k]))
    without_best = [
        t.net_return for label, ts in buckets.items() if label != best_label for t in ts
    ]
    positive_without_best = (
        len(without_best) >= 2 and annualized_sharpe(without_best, periods_per_year) > 0.0
    )
    return bucket_sharpes, positive_without_best


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

    # Parameter-variant runs: shared by PBO, the ledger, and param sensitivity.
    config_results = run_param_configs(
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
        for label, cfg_result in config_results.items():
            ledger.log_trial(spec.name, {"config": label}, cfg_result.net_returns.tolist())
    moments = return_stats(net)
    dsr = ledger.deflated_sharpe(moments.sharpe, moments.n, moments.skew, moments.kurtosis)

    # PBO across the parameter configs, time-aligned by trading day (needs >= 2
    # configs and enough days, else NaN -> criterion 3 fails closed).
    pbo = _pbo_across_configs(config_results, n_splits=pbo_splits, timezone=timezone)

    # Robustness battery + regime stability.
    robustness = run_robustness_battery(
        factory,
        params,
        steps,
        candles,
        cost_model,
        periods_per_year=periods_per_year,
        cross_symbol_candles=cross_symbol_candles,
        config_results=config_results,
        notional_per_trade=notional_per_trade,
        timezone=timezone,
    )
    # Criterion 7 defaults to the year x vol/trend partition (B-4), built from the
    # scored candles; a caller may override with an explicit labeler.
    labeler = regime_labeler or build_regime_labeler(candles, timezone=timezone)
    regime_bucket_sharpes, regime_without_best = regime_bucket_stats(
        trades, periods_per_year, labeler=labeler, timezone=timezone
    )
    primary_symbol = candles[0].symbol if candles else ""

    inputs = KillGateInputs(
        cpcv_path_sharpes=cpcv.path_sharpes,
        dsr=dsr,
        pbo=pbo,
        profit_factor=stats.profit_factor,
        top5_winners_fraction=stats.top5_winners_fraction,
        expectancy=stats.expectancy,
        round_trip_cost=cost_model.round_trip_cost_fraction(notional_per_trade),
        param_config_sharpes=robustness.param_config_sharpes,
        has_tunable_params=bool(steps),
        cross_symbol_sharpes=robustness.cross_symbol_sharpes,
        primary_symbol=primary_symbol,
        mc_shuffle_beat_fraction=robustness.mc_shuffle_beat_fraction,
        two_engines_reconcile=robustness.two_engines_reconcile,
        noise_survives=robustness.noise_survives,
        regime_bucket_sharpes=regime_bucket_sharpes,
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


def _daily_net_pnl(trades: Sequence[Trade], timezone: str) -> dict[date, float]:
    """Sum each trade's net return into its IST entry trading day."""
    tz = ZoneInfo(timezone)
    daily: dict[date, float] = {}
    for trade in trades:
        day = trade.entry_time.astimezone(tz).date()
        daily[day] = daily.get(day, 0.0) + trade.net_return
    return daily


def _pbo_across_configs(
    config_results: Mapping[str, BacktestResult], *, n_splits: int, timezone: str
) -> float:
    """PBO over the parameter configs, time-aligned by trading day (B-2).

    Each config's trades are aggregated to per-day net P&L and reindexed onto the
    UNION of trading days (a no-trade day contributes 0.0), so a matrix row is one
    trading day — the SAME period across every config — and CSCV's IS/OOS blocks
    are coherent time partitions, not positionally-stacked j-th trades landing at
    different timestamps across configs. A trading day is the natural unit of
    independence here: intraday square-off means no position crosses the close.
    Returns NaN — which the kill-gate reads as a FAILED criterion 3, never a pass —
    when the matrix cannot be formed (fewer than 2 configs, or fewer trading days
    than blocks).
    """
    per_day = {
        label: _daily_net_pnl(result.trades, timezone)
        for label, result in config_results.items()
        if result.trades
    }
    if len(per_day) < 2:
        return float("nan")  # a single-config strategy has no cross-config overfitting to measure
    all_days = sorted({day for daily in per_day.values() for day in daily})
    if len(all_days) < n_splits:
        return float("nan")
    matrix = np.array(
        [[daily.get(day, 0.0) for daily in per_day.values()] for day in all_days],
        dtype=np.float64,
    )
    # Per-day label windows (a point at each day's start); a 1-trading-day embargo
    # then purges adjacent days at IS/OOS block boundaries via the shared primitive.
    tz = ZoneInfo(timezone)
    entry_times = [datetime(d.year, d.month, d.day, tzinfo=tz) for d in all_days]
    exit_times = list(entry_times)
    return probability_of_backtest_overfitting(
        matrix, entry_times, exit_times, n_splits=n_splits
    ).pbo
