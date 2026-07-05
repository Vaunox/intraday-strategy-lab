"""Kite Connect daily authentication flow (Phase 1, P1.1).

Kite access tokens expire daily and are minted through a SEBI-mandated manual
login: the operator opens the login URL, authenticates (with 2FA/TOTP), and Kite
redirects back with a one-time ``request_token`` that is exchanged for the day's
``access_token``. That token is a secret — it is persisted only to a
git-ignored location, never logged, and registered for redaction on load.

This module (with ``kite_adapter``) is one of the only two that touch the
``kiteconnect`` SDK. The exchange is driven through the :class:`KiteSession`
Protocol so it is testable with a fake.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Protocol

from lab.core.logging import get_logger, register_secret_value

_log = get_logger("data.brokers.kite_auth")

#: Default git-ignored location for the persisted daily token (see .gitignore).
DEFAULT_TOKEN_PATH = Path("secrets") / "kite_access_token.json"


class KiteSession(Protocol):
    """The auth surface of ``kiteconnect.KiteConnect`` used to mint a token."""

    def login_url(self) -> str:
        """Return the Kite login URL the operator opens to authenticate."""

    def generate_session(self, request_token: str, api_secret: str) -> dict[str, Any]:
        """Exchange a request token for a session dict containing the access token."""


@dataclass(frozen=True, slots=True)
class StoredToken:
    """A persisted daily access token and the IST date it was issued for."""

    access_token: str
    issued_on: date

    def is_valid_for(self, day: date) -> bool:
        """Return whether this token was issued for ``day`` (tokens are daily)."""
        return self.issued_on == day


def exchange_request_token(session: KiteSession, request_token: str, api_secret: str) -> str:
    """Exchange a one-time ``request_token`` for the day's access token.

    Args:
        session: A live or fake Kite session client.
        request_token: The one-time token from the login redirect.
        api_secret: The Kite API secret (from the secrets interface).

    Returns:
        The access token (also registered for log redaction).
    """
    data = session.generate_session(request_token, api_secret)
    access_token = str(data["access_token"])
    register_secret_value(access_token)
    _log.info("kite_session_generated")  # never logs the token itself
    return access_token


def build_login_url(api_key: str) -> str:
    """Return the Kite login URL for ``api_key`` (the operator opens this to log in).

    The ``kiteconnect`` SDK is imported lazily so this module never requires it at
    import time.
    """
    from kiteconnect import KiteConnect

    client = KiteConnect(api_key=api_key)
    return str(client.login_url())


def mint_access_token(api_key: str, api_secret: str, request_token: str) -> str:
    """Mint the day's access token from a one-time ``request_token`` (live SDK).

    Thin operator-facing wrapper over :func:`exchange_request_token`; the SDK is
    imported lazily. Never logs the token.
    """
    from kiteconnect import KiteConnect

    session: KiteSession = KiteConnect(api_key=api_key)
    return exchange_request_token(session, request_token, api_secret)


class KiteTokenStore:
    """Persists the daily Kite access token to a git-ignored JSON file."""

    def __init__(self, path: Path = DEFAULT_TOKEN_PATH) -> None:
        """Bind the store to ``path`` (created on first save)."""
        self._path = path

    def save(self, token: StoredToken) -> None:
        """Write ``token`` to disk, creating the parent directory if needed."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"access_token": token.access_token, "issued_on": token.issued_on.isoformat()}
        self._path.write_text(json.dumps(payload), encoding="utf-8")
        _log.info("kite_token_saved", path=str(self._path), issued_on=token.issued_on.isoformat())

    def load(self) -> StoredToken | None:
        """Return the persisted token (registered for redaction), or ``None``."""
        if not self._path.exists():
            return None
        payload = json.loads(self._path.read_text(encoding="utf-8"))
        access_token = str(payload["access_token"])
        register_secret_value(access_token)
        return StoredToken(
            access_token=access_token, issued_on=date.fromisoformat(payload["issued_on"])
        )
