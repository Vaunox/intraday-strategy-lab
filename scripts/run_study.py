r"""Operator entry point: run one strategy study end to end (Phase 3 driver).

Thin by design — the validation harness, kill-gate, and survivorship stamp all
live in the library (``lab.research.study.run_study``). This script only marshals
inputs: it loads config, reads a strategy's candles from the raw Parquet archive,
runs the study, prints the rendered report, and (optionally) appends it to the
research paper. It logs every trial to the program-wide ledger, so the Deflated
Sharpe deflates by the honest effective-trial count across sessions.

The strategy is chosen from a registry; Phase 3 adds one thin ``StrategySpec``
per study there. The primary ``--symbol`` is scored; ``--cross-symbols`` (if any)
feed the cross-symbol robustness leg.

Example:
    uv run python scripts/run_study.py --strategy reference_momentum \
        --symbol RELIANCE --interval 5minute \
        --start 2016-01-01 --end 2024-12-31 \
        --cross-symbols TCS,INFY --data-root data \
        --paper docs/RESEARCH_FINDINGS.md
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from lab.core.config import configure_logging_from_settings, load_settings
from lab.core.constants import INDIA_TZ
from lab.core.interfaces import StrategySpec
from lab.core.logging import get_logger
from lab.core.types import BarInterval, Candle
from lab.data.store.parquet_archive import ParquetArchive
from lab.research.reports.killgate import load_kill_gate_thresholds
from lab.research.reports.paper import append_study_section
from lab.research.reports.report import render_report
from lab.research.strategies.reference import ReferenceMomentumSpec
from lab.research.study import run_study
from lab.research.trials.ledger import TrialLedger
from lab.research.validation.costs import load_cost_model
from lab.research.validation.sharpe import SharpeConvention

#: Strategy registry — Phase 3 adds one thin StrategySpec factory per study here.
STRATEGIES: dict[str, Callable[[], StrategySpec]] = {
    "reference_momentum": lambda: ReferenceMomentumSpec(),
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for a study run."""
    parser = argparse.ArgumentParser(description="Run one strategy study through the harness.")
    parser.add_argument("--strategy", required=True, choices=sorted(STRATEGIES))
    parser.add_argument("--symbol", required=True, help="primary symbol to score")
    parser.add_argument("--interval", default="5minute", help="candle interval (default: 5minute)")
    parser.add_argument("--start", required=True, help="inclusive start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="inclusive end date YYYY-MM-DD")
    parser.add_argument(
        "--cross-symbols", default=None, help="comma-separated held-out symbols (robustness leg)"
    )
    parser.add_argument("--data-root", default="data", help="raw archive root (default: data)")
    parser.add_argument(
        "--ledger-dir", default="data/ledger", help="trial ledger dir (default: data/ledger)"
    )
    parser.add_argument("--config-dir", default="config", help="config dir (default: config)")
    parser.add_argument("--env", default=None, help="config environment (default: $LAB_ENV or dev)")
    parser.add_argument(
        "--paper", default=None, help="append the rendered section to this research paper"
    )
    return parser.parse_args(argv)


def _read(
    archive: ParquetArchive, symbol: str, interval: BarInterval, start: datetime, end: datetime
) -> list[Candle]:
    return list(archive.read_candles(symbol, interval, start, end))


def main(argv: list[str] | None = None) -> None:
    """Run the study described by the command-line arguments."""
    args = _parse_args(argv)
    config_dir = Path(args.config_dir)
    settings = load_settings(args.env, config_dir=config_dir)
    configure_logging_from_settings(settings)
    log = get_logger("scripts.run_study")

    cost_model = load_cost_model(config_dir)
    thresholds = load_kill_gate_thresholds(config_dir)
    periods_per_year = SharpeConvention.from_settings(settings).periods_per_year

    tz = ZoneInfo(INDIA_TZ)
    interval = BarInterval(args.interval)
    start_day, end_day = date.fromisoformat(args.start), date.fromisoformat(args.end)
    start = datetime(start_day.year, start_day.month, start_day.day, 0, 0, tzinfo=tz)
    end = datetime(end_day.year, end_day.month, end_day.day, 23, 59, 59, tzinfo=tz)

    archive = ParquetArchive(Path(args.data_root))
    candles = _read(archive, args.symbol, interval, start, end)
    if not candles:
        raise SystemExit(
            f"no candles for {args.symbol} {interval.value} in [{args.start}, {args.end}] "
            f"under {args.data_root} — backfill first"
        )

    cross_symbols = [s.strip() for s in (args.cross_symbols or "").split(",") if s.strip()]
    cross_candles: dict[str, Sequence[Candle]] = {
        symbol: _read(archive, symbol, interval, start, end) for symbol in cross_symbols
    }
    cross_candles = {symbol: series for symbol, series in cross_candles.items() if series}

    spec = STRATEGIES[args.strategy]()
    ledger = TrialLedger(Path(args.ledger_dir))
    log.info(
        "study_start",
        strategy=spec.name,
        symbol=args.symbol,
        candles=len(candles),
        cross_symbols=len(cross_candles),
    )

    report = run_study(
        spec,
        candles,
        cost_model,
        thresholds,
        ledger,
        periods_per_year=periods_per_year,
        cross_symbol_candles=cross_candles or None,
        square_off=settings.calendar.session.square_off,
    )

    print(render_report(report))
    if args.paper:
        append_study_section(report, findings_path=Path(args.paper))
        log.info("study_appended_to_paper", path=args.paper)
    log.info(
        "study_done",
        strategy=spec.name,
        verdict=report.kill_gate.verdict.value,
    )


if __name__ == "__main__":
    main()
