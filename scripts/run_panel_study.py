r"""Operator entry point: run one strategy across the FROZEN panel (Phase 3 panel driver).

Thin by design over ``lab.research.panel.run_panel_study`` (the two-part panel verdict:
the equal-weight portfolio through the seven-point gate AND breadth). It loads config and
the FROZEN study panel, reads the panel + criterion-6d held-out symbols' candles from the
raw Parquet archive, runs the panel study, and prints the rendered report. The panel and
held-out sets are NOT command-line arguments -- they are the Lock-A frozen sets in
``config/universe/study_panel.yaml`` (so a study cannot be rescued by picking names). Every
result carries the large-cap scope caveat automatically. Trials logged are the K
aggregate-portfolio streams (panel is scope, not extra trials).

Example:
    uv run python scripts/run_panel_study.py --strategy breakout \
        --interval 5minute --start 2015-02-02 --end 2026-07-03 \
        --data-root data --ledger-dir data/ledger --config-dir config
"""

from __future__ import annotations

import argparse
import sys
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
from lab.research.panel import (
    load_panel_thresholds,
    load_study_panel,
    render_panel_report,
    run_panel_study,
)
from lab.research.reports.killgate import load_kill_gate_thresholds
from lab.research.strategies.registry import STRATEGIES
from lab.research.trials.ledger import TrialLedger
from lab.research.validation.costs import load_cost_model


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for a panel study run."""
    parser = argparse.ArgumentParser(description="Run one strategy across the frozen study panel.")
    parser.add_argument("--strategy", required=True, choices=sorted(STRATEGIES))
    parser.add_argument("--interval", default="5minute", help="candle interval (default: 5minute)")
    parser.add_argument("--start", required=True, help="inclusive start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="inclusive end date YYYY-MM-DD")
    parser.add_argument("--data-root", default="data", help="raw archive root (default: data)")
    parser.add_argument(
        "--ledger-dir", default="data/ledger", help="trial ledger dir (default: data/ledger)"
    )
    parser.add_argument("--config-dir", default="config", help="config dir (default: config)")
    parser.add_argument("--env", default=None, help="config environment (default: $LAB_ENV or dev)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the panel study described by the command-line arguments."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # UTF-8 report glyphs under a redirected pipe
    args = _parse_args(argv)
    config_dir = Path(args.config_dir)
    settings = load_settings(args.env, config_dir=config_dir)
    configure_logging_from_settings(settings)
    log = get_logger("scripts.run_panel_study")
    calendar = NseCalendar.from_settings(settings)

    cost_model = load_cost_model(config_dir)
    thresholds = load_kill_gate_thresholds(config_dir)
    panel_thresholds = load_panel_thresholds(config_dir)
    study_panel = load_study_panel(config_dir)  # the Lock-A frozen panel + held-out sets

    tz = ZoneInfo(INDIA_TZ)
    interval = BarInterval(args.interval)
    start_day, end_day = date.fromisoformat(args.start), date.fromisoformat(args.end)
    start = datetime(start_day.year, start_day.month, start_day.day, 0, 0, tzinfo=tz)
    end = datetime(end_day.year, end_day.month, end_day.day, 23, 59, 59, tzinfo=tz)
    archive = ParquetArchive(Path(args.data_root))

    def read_all(symbols: tuple[str, ...]) -> dict[str, list[Candle]]:
        out: dict[str, list[Candle]] = {}
        for symbol in symbols:
            raw = list(archive.read_candles(symbol, interval, start, end))
            candles = regular_session_candles(raw, calendar)  # drops Muhurat/out-of-session bars
            if not candles:
                raise SystemExit(
                    f"no candles for {symbol} {interval.value} in [{args.start}, {args.end}] "
                    f"under {args.data_root} — backfill first"
                )
            out[symbol] = candles
        return out

    panel_candles = read_all(study_panel.panel)
    holdout_candles = read_all(study_panel.holdout_6d)

    entry = STRATEGIES[args.strategy]
    spec_name = entry.factory(entry.base_params).name
    ledger = TrialLedger(Path(args.ledger_dir))
    log.info(
        "panel_study_start",
        strategy=spec_name,
        panel_symbols=len(panel_candles),
        holdout_symbols=len(holdout_candles),
    )

    report = run_panel_study(
        entry.factory,
        entry.base_params,
        entry.param_steps,
        panel_candles,
        holdout_candles,
        cost_model,
        thresholds,
        panel_thresholds,
        ledger,
        study_panel.scope_caveat,
        square_off=settings.calendar.session.square_off,
    )

    print(render_panel_report(report))
    log.info("panel_study_done", strategy=spec_name, verdict=report.verdict.value)


if __name__ == "__main__":
    main()
