"""Panel-scope study orchestration (Phase 3): per-symbol-then-aggregate scoring.

Single-symbol scoring is insufficient (RELIANCE-alone was blind to the lower-liquidity
tail). A panel study scores a strategy across a FROZEN multi-symbol panel and returns a
TWO-PART verdict, both parts required:

  Part 1 (aggregate): an equal-weight panel-PORTFOLIO daily return stream (contribute-zero
    on a symbol's no-trade days) must clear the same seven-point kill-gate.
  Part 2 (breadth): the median across panel symbols of the per-symbol CPCV median
    path-Sharpe must exceed the pinned bar, AND a pinned majority of symbols must be
    individually positive.

This wrapper is ADDITIVE: it composes the UNCHANGED validation primitives (CPCV, the DSR
via the ledger, PBO/CSCV, robustness, regime, the seven-criterion ``evaluate_kill_gate``)
on the aggregate stream and on per-symbol streams. It never modifies the seven-criterion
logic.

Pooling the symbols into one trade stream would be INCORRECT -- same-timestamp
cross-symbol trades leak across CPCV/CSCV purge boundaries -- so we aggregate to a per-DAY
portfolio stream (intraday square-off means a trading day is the natural unit of
independence) and score that. The trial ledger receives the K aggregate-portfolio streams
(one per param config), NOT N*K per-symbol streams: the panel is SCOPE, not extra trials,
so it must not inflate the effective-N that deflates every future study's DSR.

All Sharpes use the realized-frequency convention (:mod:`lab.research.validation.sharpe`):
the daily aggregate annualizes at its realized ~252/yr, each per-symbol stream at its own
trade rate.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import yaml

from lab.core.constants import INDIA_TZ
from lab.core.interfaces import StrategySpec
from lab.core.types import BarInterval, Candle, Side, Verdict
from lab.research.reports.killgate import (
    Criterion,
    KillGateInputs,
    KillGateResult,
    KillGateThresholds,
    evaluate_kill_gate,
)
from lab.research.reports.report import StudyReport, equity_curve, trade_statistics
from lab.research.strategies.adapter import run_strategy, signals_to_targets
from lab.research.study import (
    DEFAULT_NOTIONAL,
    SpecFactory,
    build_regime_labeler,
    enumerate_param_configs,
    regime_bucket_stats,
    survivorship_stamp,
)
from lab.research.trials.ledger import TrialLedger
from lab.research.validation.backtester import BacktestResult, Trade, run_backtest
from lab.research.validation.costs import CostModel
from lab.research.validation.cpcv import CPCVResult, combinatorial_purged_cv
from lab.research.validation.pbo import probability_of_backtest_overfitting
from lab.research.validation.robustness import (
    inject_ohlc_noise,
    monte_carlo_sign_flip,
    two_engines_agree,
    vectorized_backtest,
)
from lab.research.validation.sharpe import (
    annualized_sharpe,
    realized_periods_per_year,
    return_stats,
)
from lab.research.validation.splitter import ONE_TRADING_DAY


# --- frozen panel config (Lock A) ------------------------------------------- #
@dataclass(frozen=True, slots=True)
class StudyPanel:
    """The FROZEN exploration panel + criterion-6d held-out set (Lock A).

    Pre-committed in ``config/universe/study_panel.yaml`` and never tuned to rescue a
    study. The held-out set must be DISJOINT from the panel (criterion 6d asks "works on
    names never scored").
    """

    panel: tuple[str, ...]
    holdout_6d: tuple[str, ...]
    scope_caveat: str

    def __post_init__(self) -> None:
        """Fail loudly on an empty or overlapping frozen set (Lock A integrity)."""
        if not self.panel:
            raise ValueError("study panel has no panel symbols")
        if not self.holdout_6d:
            raise ValueError("study panel has no criterion-6d held-out symbols")
        overlap = set(self.panel) & set(self.holdout_6d)
        if overlap:
            raise ValueError(
                f"criterion-6d held-out symbols must be OUTSIDE the panel; overlap: {sorted(overlap)}"
            )


def load_study_panel(config_dir: Path) -> StudyPanel:
    """Load the frozen study panel from ``config_dir/universe/study_panel.yaml``."""
    path = config_dir / "universe" / "study_panel.yaml"
    data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    for key in ("panel", "holdout_6d", "scope_caveat"):
        if key not in data:
            raise ValueError(f"study panel artifact missing required key: {key!r}")
    return StudyPanel(
        panel=tuple(str(symbol) for symbol in data["panel"]),
        holdout_6d=tuple(str(symbol) for symbol in data["holdout_6d"]),
        scope_caveat=" ".join(str(data["scope_caveat"]).split()),
    )


# --- pinned panel thresholds ------------------------------------------------ #
@dataclass(frozen=True, slots=True)
class PanelThresholds:
    """The pinned panel-scope thresholds (loaded from the ``panel:`` block)."""

    breadth_median_path_sharpe_min: float
    breadth_positive_fraction_min: float
    noise_survives_fraction_min: float
    two_engine_reconcile_fraction_min: float
    min_panel_symbols: int
    min_portfolio_days: int

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> PanelThresholds:
        """Build panel thresholds from the parsed ``killgate.yaml`` mapping."""
        panel = mapping["panel"]
        return cls(
            breadth_median_path_sharpe_min=float(panel["breadth_median_path_sharpe_min"]),
            breadth_positive_fraction_min=float(panel["breadth_positive_fraction_min"]),
            noise_survives_fraction_min=float(panel["noise_survives_fraction_min"]),
            two_engine_reconcile_fraction_min=float(panel["two_engine_reconcile_fraction_min"]),
            min_panel_symbols=int(panel["min_panel_symbols"]),
            min_portfolio_days=int(panel["min_portfolio_days"]),
        )


def load_panel_thresholds(config_dir: Path) -> PanelThresholds:
    """Load the pinned panel thresholds from ``config_dir/killgate.yaml``."""
    data: Any = yaml.safe_load((config_dir / "killgate.yaml").read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("killgate.yaml must contain a mapping")
    return PanelThresholds.from_mapping(data)


# --- equal-weight portfolio aggregation (contribute-zero) ------------------- #
@dataclass(frozen=True, slots=True)
class PortfolioStream:
    """An equal-weight panel-portfolio DAILY return stream for one param config."""

    days: tuple[date, ...]
    returns: tuple[float, ...]  # equal-weight (contribute-zero) daily portfolio return
    entry_times: tuple[datetime, ...]  # day-start labels (coherent CPCV/PBO time blocks)


def _daily_net_pnl(trades: Sequence[Trade], tz: ZoneInfo) -> dict[date, float]:
    """Sum each trade's net return into its IST entry trading day."""
    daily: dict[date, float] = {}
    for trade in trades:
        day = trade.entry_time.astimezone(tz).date()
        daily[day] = daily.get(day, 0.0) + trade.net_return
    return daily


def equal_weight_portfolio_stream(
    per_symbol_trades: Mapping[str, Sequence[Trade]],
    n_panel_symbols: int,
    *,
    timezone: str = INDIA_TZ,
) -> PortfolioStream:
    """Aggregate per-symbol trades into an equal-weight daily portfolio stream.

    CONTRIBUTE-ZERO: on a day a symbol did not trade it contributes 0.0 to the mean, and
    the divisor is the FIXED panel size ``n_panel_symbols`` -- weight is NOT redistributed
    to the active names. A day with few active symbols is diluted toward zero: an honest
    thin-participation penalty, not a survivorship-flattering re-weight. A trading day is
    the unit of aggregation (intraday square-off means no position crosses the close), so
    CPCV/PBO over this stream partition coherent time blocks rather than leaking
    same-timestamp cross-symbol trades across a purge boundary (which is why pooling into
    one trade series is incorrect).
    """
    tz = ZoneInfo(timezone)
    per_symbol_daily = {
        symbol: _daily_net_pnl(trades, tz) for symbol, trades in per_symbol_trades.items()
    }
    all_days = sorted({day for daily in per_symbol_daily.values() for day in daily})
    returns = tuple(
        sum(daily.get(day, 0.0) for daily in per_symbol_daily.values()) / float(n_panel_symbols)
        for day in all_days
    )
    entry_times = tuple(datetime(day.year, day.month, day.day, tzinfo=tz) for day in all_days)
    return PortfolioStream(days=tuple(all_days), returns=returns, entry_times=entry_times)


def _operating_span_years(candles: Sequence[Candle]) -> float:
    """Operating span (first to last candle) in years; NaN for a degenerate window."""
    if len(candles) < 2:
        return float("nan")
    seconds = (candles[-1].timestamp - candles[0].timestamp).total_seconds()
    return seconds / (365.25 * 24.0 * 3600.0) if seconds > 0.0 else float("nan")


# --- criterion-7 regime from an equal-weight panel PRICE index (non-circular) --- #
def _symbol_daily_close(candles: Sequence[Candle], tz: ZoneInfo) -> dict[date, float]:
    """The last close per IST trading day for one symbol (a daily price series)."""
    daily: dict[date, float] = {}
    for candle in candles:
        daily[candle.timestamp.astimezone(tz).date()] = candle.close
    return daily


def build_panel_index_regime_labeler(
    panel_candles: Mapping[str, Sequence[Candle]],
    *,
    window: int = 20,
    timezone: str = INDIA_TZ,
) -> Callable[[Trade], str]:
    """Criterion-7 regime labeler from an equal-weight PANEL PRICE INDEX (Lock-A ruling).

    The regime (year x hi/lo-vol x up/down-trend) is derived from the panel symbols'
    PRICES -- the market itself, exogenous to the strategy's P&L -- exactly as the
    single-symbol labeler uses price, not trades, so it is NON-CIRCULAR. Each symbol is
    normalized to 1.0 at its first close; the equal-weight average of those levels per day
    is a synthetic index price path, and the shared ``build_regime_labeler`` tags each day
    off it. The frozen ``regime_method`` in ``study_panel.yaml`` pins this against drift to
    a year-only or P&L-derived version.
    """
    tz = ZoneInfo(timezone)
    normalized: dict[str, dict[date, float]] = {}
    for symbol, candles in panel_candles.items():
        daily = _symbol_daily_close(candles, tz)
        if daily and daily[min(daily)] > 0.0:
            first_close = daily[min(daily)]
            normalized[symbol] = {day: close / first_close for day, close in daily.items()}
    all_days = sorted({day for daily in normalized.values() for day in daily})
    index_candles: list[Candle] = []
    for day in all_days:
        levels = [daily[day] for daily in normalized.values() if day in daily]
        if levels:
            level = float(np.mean(levels))
            ts = datetime(day.year, day.month, day.day, tzinfo=tz)
            index_candles.append(
                Candle("PANEL_INDEX", BarInterval.MIN_5, ts, level, level, level, level, 0)
            )
    return build_regime_labeler(index_candles, window=window, timezone=timezone)


# --- reuse the Trade-based primitives on the aggregate stream ---------------- #
def _portfolio_daily_trades(stream: PortfolioStream) -> list[Trade]:
    """Synthetic one-per-day Trades carrying the portfolio's daily net return.

    Lets the UNCHANGED Trade-based primitives (``trade_statistics``,
    ``regime_bucket_stats``, ``equity_curve``) score the aggregate stream directly:
    ``gross_return`` = the day's portfolio return, ``cost_fraction`` = 0 (the return is
    already net of each symbol's costs), so the ``net_return`` property is the day's
    portfolio return. Concentration (criterion 5) is thus measured per DAY -- the daily
    analog of per-trade; the aggregate is inherently smoother (it averages the panel each
    day), so "top-5 winning days < 40% of gross" is a genuine, marginally-lenient
    concentration check, consistent with scoring the tradeable aggregate.
    """
    return [
        Trade(Side.LONG, entry, entry, 0.0, 0.0, ret, 0.0)
        for entry, ret in zip(stream.entry_times, stream.returns, strict=True)
    ]


def _panel_pbo(streams: Mapping[str, PortfolioStream], *, n_splits: int, timezone: str) -> float:
    """PBO across the K config portfolio streams, day-aligned (panel analog of B-2)."""
    per_config = {
        label: dict(zip(stream.days, stream.returns, strict=True))
        for label, stream in streams.items()
    }
    if len(per_config) < 2:
        return float("nan")
    all_days = sorted({day for daily in per_config.values() for day in daily})
    if len(all_days) < n_splits:
        return float("nan")
    matrix = np.array(
        [[daily.get(day, 0.0) for daily in per_config.values()] for day in all_days],
        dtype=np.float64,
    )
    tz = ZoneInfo(timezone)
    entry_times = [datetime(day.year, day.month, day.day, tzinfo=tz) for day in all_days]
    return probability_of_backtest_overfitting(
        matrix, entry_times, list(entry_times), n_splits=n_splits
    ).pbo


# --- per-symbol robustness legs (criterion 6, aggregated across the panel) --- #
def _symbol_noise_survives(
    base_spec: StrategySpec,
    candles: Sequence[Candle],
    cost_model: CostModel,
    *,
    notional: float,
    seeds: int,
    scale: float,
    timezone: str,
    square_off: time | None,
) -> bool:
    """One symbol survives if the median annualized net Sharpe over noise seeds > 0."""
    span_years = _operating_span_years(candles)
    sharpes: list[float] = []
    for seed in range(seeds):
        noisy = inject_ohlc_noise(candles, relative_scale=scale, seed=seed)
        result = run_strategy(
            base_spec,
            noisy,
            cost_model,
            notional_per_trade=notional,
            timezone=timezone,
            square_off=square_off,
        )
        ppy = realized_periods_per_year(len(result.net_returns), span_years)
        sharpes.append(annualized_sharpe(result.net_returns, ppy))
    finite = [s for s in sharpes if np.isfinite(s)]
    return bool(finite) and float(np.median(finite)) > 0.0


def _symbol_two_engines_reconcile(
    base_spec: StrategySpec,
    candles: Sequence[Candle],
    cost_model: CostModel,
    *,
    notional: float,
    timezone: str,
    square_off: time | None,
) -> bool:
    """One symbol reconciles if the event-driven and vectorized engines agree."""
    targets = signals_to_targets(candles, base_spec.generate_signals(candles), timezone=timezone)
    event_driven = run_backtest(
        candles,
        targets,
        cost_model,
        notional_per_trade=notional,
        timezone=timezone,
        square_off=square_off,
    )
    vectorized = vectorized_backtest(
        candles,
        targets,
        cost_model,
        notional_per_trade=notional,
        timezone=timezone,
        square_off=square_off,
    )
    return two_engines_agree(event_driven, vectorized)


def _symbol_cpcv_median(
    result: BacktestResult,
    span_years: float,
    *,
    n_groups: int,
    k_test_groups: int,
    embargo: timedelta,
) -> float:
    """One symbol's CPCV median path-Sharpe (base config), realized-frequency annualized."""
    net = result.net_returns
    if len(net) < n_groups:
        return float("nan")
    ppy = realized_periods_per_year(len(net), span_years)
    cpcv = combinatorial_purged_cv(
        net,
        result.entry_times,
        result.exit_times,
        n_groups=n_groups,
        k_test_groups=k_test_groups,
        periods_per_year=ppy,
        embargo=embargo,
    )
    return cpcv.median_path_sharpe


# --- the two-part panel result ---------------------------------------------- #
@dataclass(frozen=True, slots=True)
class BreadthResult:
    """Part-2 breadth across the panel symbols (BOTH legs must hold)."""

    median_path_sharpe: float
    positive_fraction: float
    per_symbol_path_sharpe: Mapping[str, float]
    passed: bool


@dataclass(frozen=True, slots=True)
class PanelStudyReport:
    """The two-part panel verdict: the aggregate seven-criterion gate AND breadth."""

    strategy: str
    verdict: Verdict
    aggregate: StudyReport  # Part 1: the seven criteria on the equal-weight portfolio
    breadth: BreadthResult  # Part 2: breadth across the panel symbols
    scope_caveat: str  # stamped on EVERY panel result (the large-cap scope caveat)
    n_panel_symbols: int
    n_portfolio_days: int
    effective_trials: float


def _insufficient_panel(
    strategy: str,
    reason: str,
    scope_caveat: str,
    n_panel: int,
    n_days: int,
    effective_trials: float,
) -> PanelStudyReport:
    """A fail-closed 'cannot certify this panel' result (a structural floor was not met)."""
    aggregate = StudyReport(
        strategy=strategy,
        cpcv=CPCVResult((), 0, 0, 0.0),
        dsr=float("nan"),
        pbo=float("nan"),
        effective_trials=effective_trials,
        trades=trade_statistics([]),
        kill_gate=KillGateResult(
            criteria=(Criterion(0, "panel evidence", False, reason),),
            verdict=Verdict.INSUFFICIENT,
        ),
        equity=None,
    )
    return PanelStudyReport(
        strategy=strategy,
        verdict=Verdict.INSUFFICIENT,
        aggregate=aggregate,
        breadth=BreadthResult(float("nan"), float("nan"), {}, passed=False),
        scope_caveat=scope_caveat,
        n_panel_symbols=n_panel,
        n_portfolio_days=n_days,
        effective_trials=effective_trials,
    )


def run_panel_study(
    spec_factory: SpecFactory,
    base_params: Mapping[str, float],
    param_steps: Mapping[str, float],
    panel_candles: Mapping[str, Sequence[Candle]],
    holdout_candles: Mapping[str, Sequence[Candle]],
    cost_model: CostModel,
    thresholds: KillGateThresholds,
    panel_thresholds: PanelThresholds,
    ledger: TrialLedger,
    scope_caveat: str,
    *,
    n_groups: int = 6,
    k_test_groups: int = 2,
    cpcv_embargo: timedelta = ONE_TRADING_DAY,
    pbo_splits: int = 8,
    notional_per_trade: float = DEFAULT_NOTIONAL,
    noise_seeds: int = 8,
    noise_scale: float = 0.0005,
    mc_shuffles: int = 1000,
    mc_seed: int = 0,
    log_trials: bool = True,
    timezone: str = INDIA_TZ,
    square_off: time | None = None,
) -> PanelStudyReport:
    """Score a strategy across the FROZEN panel (per-symbol-then-aggregate); two-part verdict.

    Part 1 scores the equal-weight panel-PORTFOLIO daily stream (contribute-zero) through
    the UNCHANGED seven-point kill-gate; criterion 6 is aggregated across the panel (noise
    and two-engine as pinned fractions, 6d REPURPOSED onto the held-out set) and criterion
    7 uses the exogenous panel-index regime. Part 2 requires breadth: the median per-symbol
    CPCV path-Sharpe over the pinned bar AND a pinned majority of symbols individually
    positive. The panel PASSES only if BOTH hold. The ledger receives the K aggregate
    streams (not N*K). Every result carries ``scope_caveat``.
    """
    strategy_name = spec_factory(base_params).name
    effective = ledger.effective_trials()
    n_panel = len(panel_candles)
    if n_panel < panel_thresholds.min_panel_symbols:
        return _insufficient_panel(
            strategy_name,
            f"got {n_panel} panel symbols (need >= {panel_thresholds.min_panel_symbols})",
            scope_caveat,
            n_panel,
            0,
            effective,
        )

    # Per-config, per-symbol backtests -> equal-weight contribute-zero portfolio streams.
    per_config_symbol_results: dict[str, dict[str, BacktestResult]] = {}
    portfolio_streams: dict[str, PortfolioStream] = {}
    for label, params in enumerate_param_configs(base_params, param_steps):
        spec = spec_factory(params)
        symbol_results = {
            symbol: run_strategy(
                spec,
                candles,
                cost_model,
                notional_per_trade=notional_per_trade,
                timezone=timezone,
                square_off=square_off,
            )
            for symbol, candles in panel_candles.items()
        }
        per_config_symbol_results[label] = symbol_results
        portfolio_streams[label] = equal_weight_portfolio_stream(
            {symbol: result.trades for symbol, result in symbol_results.items()},
            n_panel,
            timezone=timezone,
        )

    base_stream = portfolio_streams["base"]
    n_days = len(base_stream.days)
    if n_days < panel_thresholds.min_portfolio_days:
        return _insufficient_panel(
            strategy_name,
            f"got {n_days} portfolio days (need >= {panel_thresholds.min_portfolio_days})",
            scope_caveat,
            n_panel,
            n_days,
            effective,
        )

    # --- Part 1: the seven criteria on the equal-weight portfolio DAILY stream ---------
    span_years = (base_stream.days[-1] - base_stream.days[0]).days / 365.25
    ppy = realized_periods_per_year(n_days, span_years)
    base_returns = np.asarray(base_stream.returns, dtype=np.float64)
    daily_trades = _portfolio_daily_trades(base_stream)

    cpcv = combinatorial_purged_cv(
        base_returns,
        list(base_stream.entry_times),
        list(base_stream.entry_times),
        n_groups=n_groups,
        k_test_groups=k_test_groups,
        periods_per_year=ppy,
        embargo=cpcv_embargo,
    )

    # Ledger: K aggregate-portfolio streams (NOT N*K per-symbol) -- panel is scope, not trials.
    if log_trials:
        for label, stream in portfolio_streams.items():
            ledger.log_trial(
                strategy_name, {"config": label, "scope": "panel"}, list(stream.returns)
            )
    moments = return_stats(base_returns)
    dsr = ledger.deflated_sharpe(moments.sharpe, moments.n, moments.skew, moments.kurtosis)
    pbo = _panel_pbo(portfolio_streams, n_splits=pbo_splits, timezone=timezone)
    stats = trade_statistics(daily_trades)  # per-DAY concentration (criterion 5)

    # criterion 6 -- aggregated across the panel
    param_config_sharpes = {
        label: annualized_sharpe(
            np.asarray(stream.returns, dtype=np.float64),
            realized_periods_per_year(len(stream.days), span_years),
        )
        for label, stream in portfolio_streams.items()
    }
    mc_beat = monte_carlo_sign_flip(base_returns, n_shuffles=mc_shuffles, seed=mc_seed)
    base_spec = spec_factory(base_params)
    noise_survivors = sum(
        1
        for candles in panel_candles.values()
        if _symbol_noise_survives(
            base_spec,
            candles,
            cost_model,
            notional=notional_per_trade,
            seeds=noise_seeds,
            scale=noise_scale,
            timezone=timezone,
            square_off=square_off,
        )
    )
    noise_survives = noise_survivors / n_panel >= panel_thresholds.noise_survives_fraction_min
    reconcilers = sum(
        1
        for candles in panel_candles.values()
        if _symbol_two_engines_reconcile(
            base_spec,
            candles,
            cost_model,
            notional=notional_per_trade,
            timezone=timezone,
            square_off=square_off,
        )
    )
    two_engines = reconcilers / n_panel >= panel_thresholds.two_engine_reconcile_fraction_min

    # criterion 6d REPURPOSED: net Sharpe on the genuinely held-out names.
    cross_symbol_sharpes: dict[str, float] = {}
    for symbol, candles in holdout_candles.items():
        result = run_strategy(
            base_spec,
            candles,
            cost_model,
            notional_per_trade=notional_per_trade,
            timezone=timezone,
            square_off=square_off,
        )
        cross_symbol_sharpes[symbol] = annualized_sharpe(
            result.net_returns,
            realized_periods_per_year(len(result.net_returns), _operating_span_years(candles)),
        )

    # criterion 7: regime from the exogenous panel PRICE index (non-circular).
    labeler = build_panel_index_regime_labeler(panel_candles, timezone=timezone)
    regime_bucket_sharpes, regime_without_best = regime_bucket_stats(
        daily_trades, ppy, labeler=labeler, timezone=timezone
    )

    inputs = KillGateInputs(
        cpcv_path_sharpes=cpcv.path_sharpes,
        dsr=dsr,
        pbo=pbo,
        profit_factor=stats.profit_factor,
        top5_winners_fraction=stats.top5_winners_fraction,
        expectancy=stats.expectancy,
        round_trip_cost=cost_model.round_trip_cost_fraction(notional_per_trade),
        param_config_sharpes=param_config_sharpes,
        has_tunable_params=bool(param_steps),
        cross_symbol_sharpes=cross_symbol_sharpes,
        primary_symbol="PANEL-PORTFOLIO",
        mc_shuffle_beat_fraction=mc_beat,
        two_engines_reconcile=two_engines,
        noise_survives=noise_survives,
        regime_bucket_sharpes=regime_bucket_sharpes,
        regime_positive_without_best=regime_without_best,
    )
    aggregate_gate = evaluate_kill_gate(inputs, thresholds)
    provisional, provisional_note = survivorship_stamp(aggregate_gate.verdict, inputs, thresholds)
    aggregate = StudyReport(
        strategy=strategy_name,
        cpcv=cpcv,
        dsr=dsr,
        pbo=pbo,
        effective_trials=ledger.effective_trials(),
        trades=stats,
        kill_gate=aggregate_gate,
        equity=equity_curve(daily_trades),
        provisional=provisional,
        provisional_note=provisional_note,
    )

    # --- Part 2: breadth across the panel symbols -------------------------------------
    per_symbol_path_sharpe = {
        symbol: _symbol_cpcv_median(
            per_config_symbol_results["base"][symbol],
            _operating_span_years(candles),
            n_groups=n_groups,
            k_test_groups=k_test_groups,
            embargo=cpcv_embargo,
        )
        for symbol, candles in panel_candles.items()
    }
    finite = [s for s in per_symbol_path_sharpe.values() if np.isfinite(s)]
    breadth_median = float(np.median(finite)) if finite else float("nan")
    positive_fraction = sum(1 for s in finite if s > 0.0) / n_panel
    breadth_passed = bool(
        np.isfinite(breadth_median)
        and breadth_median > panel_thresholds.breadth_median_path_sharpe_min
        and positive_fraction >= panel_thresholds.breadth_positive_fraction_min
    )
    breadth = BreadthResult(
        breadth_median, positive_fraction, per_symbol_path_sharpe, breadth_passed
    )

    # --- two-part verdict: PASS iff the aggregate gate PASSES AND breadth holds --------
    if aggregate_gate.verdict is Verdict.INSUFFICIENT:
        verdict = Verdict.INSUFFICIENT
    elif aggregate_gate.verdict is Verdict.PASS and breadth_passed:
        verdict = Verdict.PASS
    else:
        verdict = Verdict.KILL

    return PanelStudyReport(
        strategy=strategy_name,
        verdict=verdict,
        aggregate=aggregate,
        breadth=breadth,
        scope_caveat=scope_caveat,
        n_panel_symbols=n_panel,
        n_portfolio_days=n_days,
        effective_trials=ledger.effective_trials(),
    )


def render_panel_report(report: PanelStudyReport) -> str:
    """Render the panel study report as a markdown section, scope caveat stamped."""
    kg = report.aggregate.kill_gate
    breadth = report.breadth
    lines = [
        f"### {report.strategy} — PANEL ({report.n_panel_symbols} symbols, "
        f"{report.n_portfolio_days} days)",
        "",
        f"- **Panel verdict:** {report.verdict.value.upper()} — Part 1 (aggregate gate) "
        f"{kg.verdict.value.upper()}, Part 2 (breadth) {'PASS' if breadth.passed else 'FAIL'} "
        f"(BOTH required for a panel PASS)",
        f"- **Breadth:** median per-symbol CPCV path-Sharpe {breadth.median_path_sharpe:.3f}, "
        f"positive fraction {breadth.positive_fraction:.2f}",
        f"- **Aggregate:** CPCV median path-Sharpe {report.aggregate.cpcv.median_path_sharpe:.3f} · "
        f"DSR {report.aggregate.dsr:.3f} · PBO {report.aggregate.pbo:.3f} · "
        f"effective trials {report.effective_trials:.2f}",
    ]
    if report.aggregate.provisional:
        lines.append(f"- **PROVISIONAL — upper bound:** {report.aggregate.provisional_note}")
    lines.append("- **Part 1 — seven-point kill-gate (equal-weight portfolio, per-day stream):**")
    for criterion in kg.criteria:
        mark = "PASS" if criterion.passed else "FAIL"
        lines.append(f"  {criterion.number}. {criterion.name}: **{mark}** — {criterion.detail}")
    lines.append(f"- **SCOPE (stamped on every panel result):** {report.scope_caveat}")
    lines.append("")
    return "\n".join(lines)
