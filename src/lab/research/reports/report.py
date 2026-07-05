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


def render_report(report: StudyReport) -> str:
    """Render the study report as a markdown section for the research paper."""
    kg = report.kill_gate
    lines = [
        f"### {report.strategy}",
        "",
        f"- **Verdict:** {kg.verdict.value.upper()}",
        f"- **CPCV median path-Sharpe (net):** {report.cpcv.median_path_sharpe:.3f} "
        f"(positive paths {report.cpcv.positive_fraction:.2f}, "
        f"10th pct {report.cpcv.tenth_percentile:.3f}, {report.cpcv.n_paths:.0f} paths)",
        f"- **DSR:** {report.dsr:.3f} · **PBO:** {report.pbo:.3f} · "
        f"**effective trials:** {report.effective_trials:.2f}",
        f"- **Trades:** {report.trades.n_trades}; profit factor "
        f"{report.trades.profit_factor:.2f}; expectancy {report.trades.expectancy:.5f}; "
        f"win rate {report.trades.win_rate:.2f}",
        "- **Seven-point kill-gate:**",
    ]
    for c in kg.criteria:
        mark = "PASS" if c.passed else "FAIL"
        lines.append(f"  {c.number}. {c.name}: **{mark}** — {c.detail}")
    lines.append("")
    return "\n".join(lines)
