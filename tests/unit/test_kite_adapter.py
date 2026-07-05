"""Tests for the Kite historical adapter and daily-auth flow (P1.1).

The adapter is driven with a fake KiteClient over recorded fixtures — no network,
no SDK import in the test path.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from lab.core.interfaces import BrokerAdapter
from lab.core.types import BarInterval, Candle
from lab.data.brokers.kite_adapter import (
    KiteAdapter,
    UnknownSymbolError,
    candle_from_kite_row,
)
from lab.data.brokers.kite_auth import KiteTokenStore, StoredToken, exchange_request_token

IST = ZoneInfo("Asia/Kolkata")
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "kite_historical_reliance_5min.json"
RELIANCE_TOKEN = 738561


def _load_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = json.loads(FIXTURE.read_text(encoding="utf-8"))
    # The real SDK returns 'date' as a tz-aware datetime; mimic that here.
    for row in rows:
        row["date"] = datetime.fromisoformat(row["date"])
    return rows


class FakeKiteClient:
    """In-memory KiteClient returning fixture rows within the requested range."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self.access_token: str | None = None
        self.calls: list[tuple[int, str]] = []

    def set_access_token(self, access_token: str) -> None:
        self.access_token = access_token

    def historical_data(
        self,
        instrument_token: int,
        from_date: datetime,
        to_date: datetime,
        interval: str,
        continuous: bool = False,
        oi: bool = False,
    ) -> list[dict[str, Any]]:
        self.calls.append((instrument_token, interval))
        return [row for row in self._rows if from_date <= row["date"] <= to_date]


class FakeKiteSession:
    """Fake auth session client for token exchange."""

    def login_url(self) -> str:
        return "https://kite.zerodha.com/connect/login?api_key=fake"

    def generate_session(self, request_token: str, api_secret: str) -> dict[str, Any]:
        assert request_token and api_secret
        return {"access_token": "resolved-access-token", "user_id": "AB1234"}


def _adapter() -> KiteAdapter:
    return KiteAdapter(FakeKiteClient(_load_rows()), {"RELIANCE": RELIANCE_TOKEN})


def test_candle_from_kite_row() -> None:
    row = _load_rows()[0]
    candle = candle_from_kite_row("RELIANCE", BarInterval.MIN_5, row)
    assert candle.symbol == "RELIANCE"
    assert candle.close == 2903.2
    assert candle.timestamp == datetime(2024, 7, 15, 9, 15, tzinfo=IST)
    assert candle.open_interest is None


def test_fetch_historical_candles_returns_ordered_candles() -> None:
    adapter = _adapter()
    start = datetime(2024, 7, 15, 9, 15, tzinfo=IST)
    end = datetime(2024, 7, 15, 15, 30, tzinfo=IST)
    candles = adapter.fetch_historical_candles("RELIANCE", BarInterval.MIN_5, start, end)
    assert len(candles) == 5
    assert all(isinstance(c, Candle) for c in candles)
    timestamps = [c.timestamp for c in candles]
    assert timestamps == sorted(timestamps)


def test_fetch_respects_time_range() -> None:
    adapter = _adapter()
    start = datetime(2024, 7, 15, 9, 20, tzinfo=IST)
    end = datetime(2024, 7, 15, 9, 30, tzinfo=IST)
    candles = adapter.fetch_historical_candles("RELIANCE", BarInterval.MIN_5, start, end)
    assert [c.timestamp.minute for c in candles] == [20, 25, 30]


def test_unknown_symbol_raises() -> None:
    adapter = KiteAdapter(FakeKiteClient(_load_rows()), {})
    with pytest.raises(UnknownSymbolError, match="RELIANCE"):
        adapter.fetch_historical_candles(
            "RELIANCE",
            BarInterval.MIN_5,
            datetime(2024, 7, 15, 9, 15, tzinfo=IST),
            datetime(2024, 7, 15, 15, 30, tzinfo=IST),
        )


def test_adapter_satisfies_broker_protocol() -> None:
    adapter: BrokerAdapter = _adapter()  # structural conformance (mypy)
    assert isinstance(adapter, BrokerAdapter)


def _use_adapter(broker: BrokerAdapter) -> Sequence[Candle]:
    return broker.fetch_historical_candles(
        "RELIANCE",
        BarInterval.MIN_5,
        datetime(2024, 7, 15, 9, 15, tzinfo=IST),
        datetime(2024, 7, 15, 9, 15, tzinfo=IST),
    )


def test_adapter_usable_through_protocol() -> None:
    assert len(_use_adapter(_adapter())) == 1


def test_exchange_request_token() -> None:
    token = exchange_request_token(FakeKiteSession(), "one-time-request-token", "api-secret")
    assert token == "resolved-access-token"


def test_token_store_round_trip(tmp_path: Path) -> None:
    store = KiteTokenStore(tmp_path / "secrets" / "kite_access_token.json")
    assert store.load() is None
    store.save(StoredToken(access_token="tok-abc", issued_on=date(2024, 7, 15)))
    loaded = store.load()
    assert loaded is not None
    assert loaded.access_token == "tok-abc"
    assert loaded.is_valid_for(date(2024, 7, 15))
    assert not loaded.is_valid_for(date(2024, 7, 16))
