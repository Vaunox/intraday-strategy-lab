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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

from lab.core.constants import INDIA_TZ
from lab.research.validation.backtester import Trade


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
