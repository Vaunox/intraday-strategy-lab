"""Generate the NSE trading-holiday list for config/default.yaml (P1.4 maintenance).

NSE and BSE share the same equity trading holidays, so this derives them from the
maintained ``exchange_calendars`` XBOM calendar (weekday non-sessions) rather than
hand-typing dates — which is more accurate and complete (it picks up special/ad-hoc
holidays like election days). The runtime calendar stays config-driven; this only
regenerates the data that lives in config.

Requires the optional 'tooling' dependency group::

    uv sync --group tooling
    uv run python scripts/generate_nse_holidays.py --start 2018 --end 2026

Paste the printed block into config/default.yaml under ``calendar.holidays`` and
re-run the calendar tests.
"""

from __future__ import annotations

import argparse
from datetime import date

import exchange_calendars as xc
import pandas as pd


def generate_holidays(start_year: int, end_year: int) -> list[date]:
    """Return NSE (XBOM) weekday trading holidays across the inclusive year range."""
    calendar = xc.get_calendar("XBOM")
    start = pd.Timestamp(f"{start_year}-01-01")
    end = pd.Timestamp(f"{end_year}-12-31")
    sessions = set(calendar.sessions_in_range(start, end).date)
    business_days = pd.bdate_range(start, end)
    return sorted(day.date() for day in business_days if day.date() not in sessions)


def to_yaml_block(holidays: list[date]) -> str:
    """Format holidays as an indented YAML list, grouped by year with comments."""
    lines: list[str] = []
    current_year: int | None = None
    for day in holidays:
        if day.year != current_year:
            current_year = day.year
            lines.append(f"    # --- {day.year} ---")
        lines.append(f'    - "{day.isoformat()}"')
    return "\n".join(lines)


def main() -> None:
    """Print the config YAML block for the requested year range."""
    parser = argparse.ArgumentParser(description="Generate NSE holidays for config.")
    parser.add_argument("--start", type=int, default=2018, help="first year (inclusive)")
    parser.add_argument("--end", type=int, default=2026, help="last year (inclusive)")
    args = parser.parse_args()
    print(to_yaml_block(generate_holidays(args.start, args.end)))


if __name__ == "__main__":
    main()
