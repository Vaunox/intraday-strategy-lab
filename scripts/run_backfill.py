r"""Operator entry point: run a historical backfill into the raw Parquet archive.

Wires configuration, secrets, the Kite adapter, the archive, and the calendar,
then runs a :class:`~lab.data.ingest.backfill.BackfillPlan`. Instrument tokens are
read from a JSON ``{symbol: token}`` file (produced from the Kite instruments
dump). Requires valid Kite credentials in the environment / ``.env`` and a
current daily access token.

Example:
    uv run python scripts/run_backfill.py \
        --symbols RELIANCE,TCS --interval 5minute \
        --start 2024-01-01 --end 2024-03-31 \
        --instruments secrets/instrument_tokens.json --data-root data
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from lab.core.config import configure_logging_from_settings, load_settings
from lab.core.logging import get_logger
from lab.core.nse_calendar import NseCalendar
from lab.core.secrets import EnvSecretsProvider
from lab.core.types import BarInterval
from lab.data.brokers.kite_adapter import KiteAdapter
from lab.data.ingest.backfill import Backfiller, BackfillPlan
from lab.data.store.parquet_archive import ParquetArchive


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the backfill."""
    parser = argparse.ArgumentParser(description="Backfill Kite historical candles into Parquet.")
    parser.add_argument("--env", default=None, help="config environment (default: $LAB_ENV or dev)")
    parser.add_argument("--symbols", required=True, help="comma-separated trading symbols")
    parser.add_argument("--interval", required=True, help="candle interval, e.g. 5minute")
    parser.add_argument("--start", required=True, help="inclusive start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="inclusive end date YYYY-MM-DD")
    parser.add_argument("--instruments", required=True, help="path to {symbol: token} JSON")
    parser.add_argument("--data-root", default="data", help="archive root directory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the backfill described by the command-line arguments."""
    args = _parse_args(argv)
    settings = load_settings(args.env)
    configure_logging_from_settings(settings)
    log = get_logger("scripts.run_backfill")

    secrets = EnvSecretsProvider.from_environment(dotenv_path=Path(".env"))
    instrument_tokens: dict[str, int] = json.loads(Path(args.instruments).read_text("utf-8"))
    adapter = KiteAdapter.from_secrets(secrets, instrument_tokens)
    archive = ParquetArchive(Path(args.data_root))
    calendar = NseCalendar.from_settings(settings)

    plan = BackfillPlan(
        symbols=tuple(s.strip() for s in args.symbols.split(",") if s.strip()),
        interval=BarInterval(args.interval),
        start=date.fromisoformat(args.start),
        end=date.fromisoformat(args.end),
    )
    log.info("backfill_start", symbols=len(plan.symbols), interval=plan.interval.value)
    Backfiller(adapter, archive, calendar).run(plan)


if __name__ == "__main__":
    main()
