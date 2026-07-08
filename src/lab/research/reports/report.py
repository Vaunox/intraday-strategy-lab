"""Per-study report assembly and rendering (Phase 2, P2.6).

Turns a study's trades and validation outputs into trade statistics (profit
factor, P&L concentration, expectancy) and a rendered markdown section carrying
the CPCV distribution, DSR (with the effective trial count), PBO, and the
seven-point kill-gate verdict.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from lab.research.reports.killgate import KillGateResult
from lab.research.validation.backtester import Trade
from lab.research.validation.cpcv import CPCVResult


@dataclass(frozen=True, slots=True)
class TradeStatistics:
    """Concentration and expectancy statistics over a study's trades."""

    n_trades: int
    profit_factor: float
    top5_winners_fraction: float
    expectancy: float
    win_rate: float
    total_net_return: float


@dataclass(frozen=True, slots=True)
class EquityCurve:
    """Walk-forward realized equity: cumulative net return trade-by-trade in time order."""

    equity: tuple[float, ...]  # cumulative net return AFTER each trade (chronological)
    total_return: float  # final cumulative net return
    max_drawdown: float  # largest peak-to-trough decline in the curve (>= 0)
    n_trades: int


def equity_curve(trades: Sequence[Trade]) -> EquityCurve:
    """Build the walk-forward equity curve (cumulative net return) over ``trades``.

    Trades are taken in the order produced by the backtester (chronological). The
    curve is additive in per-trade net returns — the honest realized path an
    operator would have walked, drawdown included.
    """
    if not trades:
        return EquityCurve(equity=(), total_return=0.0, max_drawdown=0.0, n_trades=0)
    nets = np.array([t.net_return for t in trades], dtype=np.float64)
    curve = np.cumsum(nets)
    running_peak = np.maximum.accumulate(curve)
    max_drawdown = float(np.max(running_peak - curve))
    return EquityCurve(
        equity=tuple(float(x) for x in curve),
        total_return=float(curve[-1]),
        max_drawdown=max_drawdown,
        n_trades=int(nets.size),
    )


def trade_statistics(trades: Sequence[Trade]) -> TradeStatistics:
    """Compute profit factor, top-5 concentration, and expectancy over trades."""
    if not trades:
        return TradeStatistics(0, float("nan"), float("nan"), float("nan"), float("nan"), 0.0)
    nets = np.array([t.net_return for t in trades], dtype=np.float64)
    wins = nets[nets > 0]
    losses = nets[nets < 0]
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = float("nan")

    top5 = float(np.sort(wins)[-5:].sum()) if wins.size else 0.0
    top5_fraction = top5 / gross_profit if gross_profit > 0 else float("nan")

    return TradeStatistics(
        n_trades=int(nets.size),
        profit_factor=profit_factor,
        top5_winners_fraction=top5_fraction,
        expectancy=float(nets.mean()),
        win_rate=float(wins.size / nets.size),
        total_net_return=float(nets.sum()),
    )


@dataclass(frozen=True, slots=True)
class StudyReport:
    """The full result of a single strategy study."""

    strategy: str
    cpcv: CPCVResult
    dsr: float
    pbo: float
    effective_trials: float
    trades: TradeStatistics
    kill_gate: KillGateResult
    equity: EquityCurve | None = None
    provisional: bool = False  # narrow gate pass on survivor-biased data (upper bound)
    provisional_note: str = ""


#: Unicode blocks for a compact inline equity sparkline (low -> high).
_SPARK_BLOCKS = "▁▂▃▄▅▆▇█"


def _sparkline(values: Sequence[float], width: int = 40) -> str:
    """Render ``values`` as a fixed-width unicode sparkline (empty if <2 points)."""
    series = np.asarray(values, dtype=np.float64)
    if series.size < 2:
        return ""
    if series.size > width:  # downsample by averaging into ``width`` buckets
        buckets = np.array_split(series, width)
        series = np.array([b.mean() for b in buckets], dtype=np.float64)
    low, high = float(series.min()), float(series.max())
    span = high - low
    if span <= 0.0:
        return _SPARK_BLOCKS[0] * series.size
    scaled = (series - low) / span * (len(_SPARK_BLOCKS) - 1)
    indices = np.clip(np.round(scaled), 0, len(_SPARK_BLOCKS) - 1).astype(np.int64)
    return "".join(_SPARK_BLOCKS[i] for i in indices)


def render_report(report: StudyReport) -> str:
    """Render the study report as a markdown section for the research paper."""
    kg = report.kill_gate
    lines = [
        f"### {report.strategy}",
        "",
        f"- **Verdict:** {kg.verdict.value.upper()}",
    ]
    if report.provisional:
        lines.append(f"- **PROVISIONAL — upper bound:** {report.provisional_note}")
    lines += [
        f"- **CPCV median path-Sharpe (net):** {report.cpcv.median_path_sharpe:.3f} "
        f"(positive paths {report.cpcv.positive_fraction:.2f}, "
        f"10th pct {report.cpcv.tenth_percentile:.3f}, {report.cpcv.n_paths:.0f} paths)",
        f"- **DSR:** {report.dsr:.3f} · **PBO:** {report.pbo:.3f} · "
        f"**effective trials:** {report.effective_trials:.2f}",
        f"- **Trades:** {report.trades.n_trades}; profit factor "
        f"{report.trades.profit_factor:.2f}; expectancy {report.trades.expectancy:.5f}; "
        f"win rate {report.trades.win_rate:.2f}",
    ]
    if report.equity is not None:
        eq = report.equity
        spark = _sparkline(eq.equity)
        spark_suffix = f" `{spark}`" if spark else ""
        lines.append(
            f"- **Walk-forward equity (net):** total {eq.total_return:+.4f}, "
            f"max drawdown {eq.max_drawdown:.4f} over {eq.n_trades} trades{spark_suffix}"
        )
    lines.append("- **Seven-point kill-gate:**")
    for c in kg.criteria:
        mark = "PASS" if c.passed else "FAIL"
        lines.append(f"  {c.number}. {c.name}: **{mark}** — {c.detail}")
    lines.append("")
    return "\n".join(lines)
