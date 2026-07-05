"""Operator entry point: mint the daily Kite access token (P1.1).

Kite access tokens expire daily and require a manual 2FA/TOTP login. Run once with
no arguments to get the login URL; log in, copy the ``request_token`` from the
redirect URL, then run again with ``--request-token`` to mint and save the day's
token. The token is saved to a git-ignored location and is never printed.

Requires ``KITE_API_KEY`` and ``KITE_API_SECRET`` in the environment / ``.env``.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from lab.core.constants import INDIA_TZ
from lab.core.secrets import EnvSecretsProvider
from lab.data.brokers.kite_adapter import API_KEY_SECRET, API_SECRET_SECRET
from lab.data.brokers.kite_auth import (
    KiteTokenStore,
    StoredToken,
    build_login_url,
    mint_access_token,
)


def main(argv: list[str] | None = None) -> None:
    """Mint (or start minting) the day's Kite access token."""
    parser = argparse.ArgumentParser(description="Mint the daily Kite access token.")
    parser.add_argument(
        "--request-token", default=None, help="one-time token from the login redirect URL"
    )
    args = parser.parse_args(argv)

    secrets = EnvSecretsProvider.from_environment(dotenv_path=Path(".env"))
    api_key = secrets.get(API_KEY_SECRET)

    if args.request_token is None:
        print("1) Open this URL and log in (2FA/TOTP):")
        print(f"   {build_login_url(api_key)}")
        print("2) After login you land on a redirect URL containing request_token=...")
        print("3) Re-run: uv run python scripts/kite_login.py --request-token <REQUEST_TOKEN>")
        return

    api_secret = secrets.get(API_SECRET_SECRET)
    access_token = mint_access_token(api_key, api_secret, args.request_token)
    issued_on = datetime.now(ZoneInfo(INDIA_TZ)).date()
    KiteTokenStore().save(StoredToken(access_token=access_token, issued_on=issued_on))
    print(f"Access token minted and saved for {issued_on} (git-ignored). The backfill can now run.")


if __name__ == "__main__":
    main()
