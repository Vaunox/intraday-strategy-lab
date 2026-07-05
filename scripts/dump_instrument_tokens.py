"""Operator entry point: dump the {symbol: token} instrument map to JSON (P1.1).

Fetches the Kite instruments dump and writes the ``{tradingsymbol: instrument_token}``
mapping the backfill needs (``--instruments``). Requires a current daily access
token (run ``scripts/kite_login.py`` first) and ``KITE_API_KEY`` in the
environment / ``.env``. Restricted to cash-equity (EQ) instruments; pass a
``--universe`` to narrow it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lab.core.secrets import EnvSecretsProvider
from lab.data.brokers.kite_adapter import (
    API_KEY_SECRET,
    MissingAccessTokenError,
    fetch_instrument_tokens,
)
from lab.data.brokers.kite_auth import KiteTokenStore


def _load_universe(spec: str | None) -> list[str] | None:
    """Parse a universe from a comma list or a file of one symbol per line."""
    if not spec:
        return None
    path = Path(spec)
    if path.exists():
        return [line.strip() for line in path.read_text("utf-8").splitlines() if line.strip()]
    return [symbol.strip() for symbol in spec.split(",") if symbol.strip()]


def main(argv: list[str] | None = None) -> None:
    """Fetch and write the instrument-token map."""
    parser = argparse.ArgumentParser(description="Dump NSE instrument tokens to JSON.")
    parser.add_argument(
        "--output", default="secrets/instrument_tokens.json", help="output JSON path"
    )
    parser.add_argument("--exchange", default="NSE", help="exchange (default: NSE)")
    parser.add_argument(
        "--universe", default=None, help="comma-separated symbols, or a path to a symbols file"
    )
    args = parser.parse_args(argv)

    secrets = EnvSecretsProvider.from_environment(dotenv_path=Path(".env"))
    api_key = secrets.get(API_KEY_SECRET)
    stored = KiteTokenStore().load()
    if stored is None:
        raise MissingAccessTokenError("no access token; run scripts/kite_login.py first")

    tokens = fetch_instrument_tokens(
        api_key, stored.access_token, exchange=args.exchange, universe=_load_universe(args.universe)
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(tokens, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {len(tokens)} instrument tokens to {output}")


if __name__ == "__main__":
    main()
