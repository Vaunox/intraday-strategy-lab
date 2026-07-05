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
from lab.core.secrets import EnvSecretsProvider
from lab.core.types import BarInterval, Candle
from lab.data.brokers.kite_adapter import (
    KiteAdapter,
    MissingAccessTokenError,
    UnknownSymbolError,
    _parse_instrument_tokens,
    _resolve_access_token,
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
    # request_interval=0 so tests never sleep on the throttle.
    return KiteAdapter(
        FakeKiteClient(_load_rows()), {"RELIANCE": RELIANCE_TOKEN}, request_interval=0.0
    )


class _RateLimitedClient(FakeKiteClient):
    """A fake client that raises 'Too many requests' the first ``fail_times`` calls."""

    def __init__(self, rows: list[dict[str, Any]], *, fail_times: int) -> None:
        super().__init__(rows)
        self._fail_times = fail_times
        self.attempts = 0

    def historical_data(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        self.attempts += 1
        if self.attempts <= self._fail_times:
            raise RuntimeError("Too many requests")  # message matched by _is_rate_limit
        return super().historical_data(*args, **kwargs)


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


def test_fetch_skips_corrupt_bars_without_crashing() -> None:
    # Kite can emit a zero-price bar; it must be dropped, not abort the whole fetch.
    rows = _load_rows()
    bad = dict(rows[0])
    bad["date"] = datetime(2024, 7, 15, 9, 40, tzinfo=IST)
    bad["open"] = 0.0  # corrupt vendor bar
    adapter = KiteAdapter(
        FakeKiteClient([*rows, bad]), {"RELIANCE": RELIANCE_TOKEN}, request_interval=0.0
    )
    candles = adapter.fetch_historical_candles(
        "RELIANCE",
        BarInterval.MIN_5,
        datetime(2024, 7, 15, 9, 15, tzinfo=IST),
        datetime(2024, 7, 15, 15, 30, tzinfo=IST),
    )
    assert len(candles) == 5  # five good bars kept, the zero-price bar dropped
    assert all(c.open > 0.0 for c in candles)


def test_fetch_retries_on_rate_limit_then_succeeds() -> None:
    client = _RateLimitedClient(_load_rows(), fail_times=2)
    waits: list[float] = []
    adapter = KiteAdapter(
        client,
        {"RELIANCE": RELIANCE_TOKEN},
        request_interval=0.0,
        backoff_base=0.01,
        sleep=waits.append,
    )
    candles = adapter.fetch_historical_candles(
        "RELIANCE",
        BarInterval.MIN_5,
        datetime(2024, 7, 15, 9, 15, tzinfo=IST),
        datetime(2024, 7, 15, 15, 30, tzinfo=IST),
    )
    assert client.attempts == 3  # two 429s, then success
    assert len(candles) == 5
    assert waits == [0.01, 0.02]  # exponential backoff between retries


def test_fetch_raises_after_max_retries() -> None:
    client = _RateLimitedClient(_load_rows(), fail_times=99)
    adapter = KiteAdapter(
        client,
        {"RELIANCE": RELIANCE_TOKEN},
        request_interval=0.0,
        max_retries=2,
        backoff_base=0.0,
        sleep=lambda _s: None,
    )
    with pytest.raises(RuntimeError, match="Too many requests"):
        adapter.fetch_historical_candles(
            "RELIANCE",
            BarInterval.MIN_5,
            datetime(2024, 7, 15, 9, 15, tzinfo=IST),
            datetime(2024, 7, 15, 15, 30, tzinfo=IST),
        )
    assert client.attempts == 3  # initial try + 2 retries


def test_fetch_does_not_retry_non_rate_limit_errors() -> None:
    class _Boom(FakeKiteClient):
        def historical_data(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
            self.calls.append((0, "x"))
            raise ValueError("boom")

    adapter = KiteAdapter(
        _Boom(_load_rows()),
        {"RELIANCE": RELIANCE_TOKEN},
        request_interval=0.0,
        sleep=lambda _s: None,
    )
    with pytest.raises(ValueError, match="boom"):
        adapter.fetch_historical_candles(
            "RELIANCE",
            BarInterval.MIN_5,
            datetime(2024, 7, 15, 9, 15, tzinfo=IST),
            datetime(2024, 7, 15, 15, 30, tzinfo=IST),
        )


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


def test_resolve_access_token_prefers_token_store(tmp_path: Path) -> None:
    store = KiteTokenStore(tmp_path / "kite_access_token.json")
    store.save(StoredToken(access_token="from-store", issued_on=date(2024, 7, 15)))
    secrets = EnvSecretsProvider({"KITE_ACCESS_TOKEN": "from-env"})
    assert _resolve_access_token(secrets, store) == "from-store"


def test_resolve_access_token_falls_back_to_secret(tmp_path: Path) -> None:
    empty_store = KiteTokenStore(tmp_path / "missing.json")  # nothing saved
    secrets = EnvSecretsProvider({"KITE_ACCESS_TOKEN": "from-env"})
    assert _resolve_access_token(secrets, empty_store) == "from-env"
    assert _resolve_access_token(secrets, None) == "from-env"


def test_resolve_access_token_raises_when_absent() -> None:
    with pytest.raises(MissingAccessTokenError, match="kite_login"):
        _resolve_access_token(EnvSecretsProvider({}), None)


def test_parse_instrument_tokens_filters_equity_and_universe() -> None:
    rows = [
        {"tradingsymbol": "RELIANCE", "instrument_token": 738561, "instrument_type": "EQ"},
        {"tradingsymbol": "TCS", "instrument_token": 2953217, "instrument_type": "EQ"},
        {"tradingsymbol": "NIFTY24DECFUT", "instrument_token": 999, "instrument_type": "FUT"},
    ]
    assert _parse_instrument_tokens(rows) == {"RELIANCE": 738561, "TCS": 2953217}
    assert _parse_instrument_tokens(rows, universe=["RELIANCE"]) == {"RELIANCE": 738561}
