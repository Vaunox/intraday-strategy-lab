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
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from lab.core.config import configure_logging_from_settings, load_settings
from lab.core.constants import INDIA_TZ
from lab.core.logging import get_logger
from lab.core.nse_calendar import NseCalendar
from lab.core.types import BarInterval, Candle
from lab.data.hygiene.session import regular_session_candles
from lab.data.store.parquet_archive import ParquetArchive
from lab.research.reports.killgate import load_kill_gate_thresholds
from lab.research.reports.paper import append_study_section
from lab.research.reports.report import render_report
from lab.research.strategies.reference import ReferenceMomentumSpec
from lab.research.strategies.vwap import vwap_mean_reversion_spec
from lab.research.study import SpecFactory, run_study
from lab.research.trials.ledger import TrialLedger
from lab.research.validation.costs import load_cost_model


@dataclass(frozen=True)
class StrategyEntry:
    """A registered study: its spec factory plus the PRE-COMMITTED parameters.

    ``base_params`` is the frozen pre-registered configuration; ``param_steps`` is
    the +/- one-step neighbour per tunable parameter that drives criterion-6a
    parameter sensitivity and the PBO configuration matrix. A parameter-free
    strategy leaves both empty (its factory ignores the argument).
    """

    factory: SpecFactory
    base_params: Mapping[str, float]
    param_steps: Mapping[str, float]


#: Strategy registry — Phase 3 adds one entry per study. Parameters are the
#: PRE-REGISTERED, frozen values (see docs/pre_registration/).
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
    # Force UTF-8 stdout so the rendered report's non-ASCII glyphs (equity sparkline
    # blocks, middle dot, phi) print under a redirected pipe on Windows (cp1252
    # default) instead of crashing the run with UnicodeEncodeError after the whole
    # pipeline has already completed.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = _parse_args(argv)
    config_dir = Path(args.config_dir)
    settings = load_settings(args.env, config_dir=config_dir)
    configure_logging_from_settings(settings)
    log = get_logger("scripts.run_study")
    calendar = NseCalendar.from_settings(settings)

    cost_model = load_cost_model(config_dir)
    thresholds = load_kill_gate_thresholds(config_dir)

    tz = ZoneInfo(INDIA_TZ)
    interval = BarInterval(args.interval)
    start_day, end_day = date.fromisoformat(args.start), date.fromisoformat(args.end)
    start = datetime(start_day.year, start_day.month, start_day.day, 0, 0, tzinfo=tz)
    end = datetime(end_day.year, end_day.month, end_day.day, 23, 59, 59, tzinfo=tz)

    archive = ParquetArchive(Path(args.data_root))
    raw_candles = _read(archive, args.symbol, interval, start, end)
    # Filter to the regular session (drops Muhurat evening bars etc.) at the ingest
    # boundary, before anything computes features or backtests; the raw store is
    # untouched.
    candles = regular_session_candles(raw_candles, calendar)
    if not candles:
        raise SystemExit(
            f"no candles for {args.symbol} {interval.value} in [{args.start}, {args.end}] "
            f"under {args.data_root} — backfill first"
        )
    if len(candles) != len(raw_candles):
        log.info(
            "session_filter",
            symbol=args.symbol,
            kept=len(candles),
            dropped=len(raw_candles) - len(candles),
        )

    cross_symbols = [s.strip() for s in (args.cross_symbols or "").split(",") if s.strip()]
    cross_candles: dict[str, Sequence[Candle]] = {
        symbol: regular_session_candles(_read(archive, symbol, interval, start, end), calendar)
        for symbol in cross_symbols
    }
    cross_candles = {symbol: series for symbol, series in cross_candles.items() if series}

    entry = STRATEGIES[args.strategy]
    spec = entry.factory(entry.base_params)
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
        spec_factory=entry.factory,
        base_params=entry.base_params,
        param_steps=entry.param_steps,
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
