"""Kite Connect historical-candle adapter (Phase 1, P1.1).

The concrete ``BrokerAdapter`` for Zerodha Kite Connect. This module and
``kite_auth`` are the ONLY places the ``kiteconnect`` SDK is imported (Part I §1,
Part II data policy) — everything else depends on the ``BrokerAdapter`` Protocol.

Only the historical-candle surface is used: no order placement, no live market
depth (out of scope for this research program). The SDK client is injected behind
the :class:`KiteClient` Protocol, so tests drive the adapter with a fake and
recorded fixtures instead of the network.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from lab.core.logging import get_logger
from lab.core.secrets import SecretsProvider
from lab.core.types import BarInterval, Candle

_log = get_logger("data.brokers.kite")

#: Secret names the adapter resolves through the secrets interface (never literals).
API_KEY_SECRET = "KITE_API_KEY"  # noqa: S105 — secret NAME (env-var key), not a value
ACCESS_TOKEN_SECRET = "KITE_ACCESS_TOKEN"  # noqa: S105 — secret NAME, not a value


@runtime_checkable
class KiteClient(Protocol):
    """The minimal surface of ``kiteconnect.KiteConnect`` this program uses.

    Typing against this Protocol (rather than the untyped SDK) keeps the adapter
    statically checked and lets tests substitute a fake client.
    """

    def set_access_token(self, access_token: str) -> None:
        """Attach the day's access token to the client."""

    def historical_data(
        self,
        instrument_token: int,
        from_date: datetime,
        to_date: datetime,
        interval: str,
        continuous: bool = False,
        oi: bool = False,
    ) -> list[dict[str, Any]]:
        """Return raw candle rows for an instrument over a date range."""


class UnknownSymbolError(KeyError):
    """Raised when a symbol cannot be resolved to a Kite instrument token."""


def candle_from_kite_row(symbol: str, interval: BarInterval, row: Mapping[str, Any]) -> Candle:
    """Convert one Kite ``historical_data`` row into a :class:`Candle`.

    Kite returns rows keyed ``date`` (timezone-aware IST), ``open``, ``high``,
    ``low``, ``close``, ``volume`` and, when requested, ``oi``. Validation of the
    OHLC invariants and timezone-awareness happens in :class:`Candle` itself.
    """
    oi_raw = row.get("oi")
    return Candle(
        symbol=symbol,
        interval=interval,
        timestamp=row["date"],
        open=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        close=float(row["close"]),
        volume=int(row["volume"]),
        open_interest=int(oi_raw) if oi_raw is not None else None,
    )


class KiteAdapter:
    """Fetches historical candles from Kite Connect behind ``BrokerAdapter``.

    Args:
        client: A live or fake :class:`KiteClient`.
        instrument_tokens: Mapping of trading symbol to Kite instrument token
            (sourced from the instruments dump; injected so the adapter stays
            offline-testable).
        fetch_oi: Whether to request open interest (only meaningful for
            derivatives; off for cash equities).
    """

    def __init__(
        self,
        client: KiteClient,
        instrument_tokens: Mapping[str, int],
        *,
        fetch_oi: bool = False,
    ) -> None:
        self._client = client
        self._tokens = dict(instrument_tokens)
        self._fetch_oi = fetch_oi

    @classmethod
    def from_secrets(
        cls,
        secrets: SecretsProvider,
        instrument_tokens: Mapping[str, int],
        *,
        fetch_oi: bool = False,
    ) -> KiteAdapter:
        """Build a live adapter, resolving credentials through the secrets interface.

        The ``kiteconnect`` SDK is imported lazily here so importing this module
        never requires the SDK to be installed at type-check time on paths that
        only use a fake client.
        """
        from kiteconnect import KiteConnect

        api_key = secrets.get(API_KEY_SECRET)
        access_token = secrets.get(ACCESS_TOKEN_SECRET)
        client: KiteClient = KiteConnect(api_key=api_key)
        client.set_access_token(access_token)
        return cls(client, instrument_tokens, fetch_oi=fetch_oi)

    def _resolve_token(self, symbol: str) -> int:
        try:
            return self._tokens[symbol]
        except KeyError as exc:
            raise UnknownSymbolError(
                f"no Kite instrument token known for symbol {symbol!r}"
            ) from exc

    def fetch_historical_candles(
        self,
        symbol: str,
        interval: BarInterval,
        start: datetime,
        end: datetime,
    ) -> Sequence[Candle]:
        """Return candles for ``symbol`` at ``interval`` within ``[start, end]``.

        Rows are returned by Kite in ascending time order; that order is
        preserved. See :class:`lab.core.interfaces.BrokerAdapter`.
        """
        token = self._resolve_token(symbol)
        _log.debug(
            "fetch_historical_candles",
            symbol=symbol,
            interval=interval.value,
            start=start.isoformat(),
            end=end.isoformat(),
        )
        rows = self._client.historical_data(
            token, start, end, interval.value, continuous=False, oi=self._fetch_oi
        )
        return [candle_from_kite_row(symbol, interval, row) for row in rows]
