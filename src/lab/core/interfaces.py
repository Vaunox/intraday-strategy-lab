"""Swappable interface Protocols for anything with more than one implementation.

Programming to these — not to concrete classes — is what keeps the broker SDK
out of everything but ``data/brokers/`` and the storage client out of everything
but ``data/store/`` (Part I §1). They are structural (``typing.Protocol``): any
object with the right shape satisfies them, so trivial fakes work in tests
without inheritance.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
from typing import Protocol, runtime_checkable

from lab.core.types import BarInterval, Candle, StrategySignal


@runtime_checkable
class BrokerAdapter(Protocol):
    """Read-only access to a broker's HISTORICAL candle data.

    This research program uses Kite historical candles only — there is no order
    placement and no live market-depth surface here (Part II data policy). The
    concrete ``KiteAdapter`` (Phase 1, P1.1) is the only code that imports the
    Kite SDK; everything else depends on this Protocol.
    """

    def fetch_historical_candles(
        self,
        symbol: str,
        interval: BarInterval,
        start: datetime,
        end: datetime,
    ) -> Sequence[Candle]:
        """Return candles for ``symbol`` at ``interval`` within ``[start, end]``.

        Args:
            symbol: Trading symbol (exchange-qualified as the adapter requires).
            interval: Candle interval to fetch.
            start: Inclusive start of the range (timezone-aware, IST).
            end: Inclusive end of the range (timezone-aware, IST).

        Returns:
            Candles ordered by ascending timestamp.
        """
        ...


@runtime_checkable
class Repository(Protocol):
    """Versioned, append-only storage for candle data.

    The raw archive is immutable: corrections are written as new versions, never
    silent in-place mutations (Part III Layer 1). The concrete Parquet
    implementation (Phase 1, P1.2) is the only code that imports the storage
    client; everything else depends on this Protocol.
    """

    def write_candles(
        self,
        symbol: str,
        interval: BarInterval,
        candles: Sequence[Candle],
    ) -> None:
        """Persist ``candles`` for ``symbol``/``interval`` (append-only)."""
        ...

    def read_candles(
        self,
        symbol: str,
        interval: BarInterval,
        start: datetime,
        end: datetime,
    ) -> Sequence[Candle]:
        """Return stored candles for ``symbol``/``interval`` within ``[start, end]``."""
        ...

    def stored_dates(self, symbol: str, interval: BarInterval) -> Sequence[date]:
        """Return the trading dates already stored for ``symbol``/``interval``.

        Lets a resumable backfill (P1.3) skip completed days without re-fetching
        or duplicating; a store with no notion of partitioning may return empty.
        """
        ...


@runtime_checkable
class StrategySpec(Protocol):
    """A rule-based intraday strategy: event → entry → exit/holding → position.

    Each of the 20 strategies in the slate (Part V) is one *thin* spec that emits
    point-in-time signals from candles and never touches the validation engine.
    The adapter turning a spec's signals into the per-period position/return
    series the CPCV engine and kill-gate consume is built once in Phase 2 (P2.4);
    the validation engine is written once and reused unchanged for all specs
    (Part III Layer 2).
    """

    @property
    def name(self) -> str:
        """Stable, unique identifier for the strategy (used in reports/ledger)."""
        ...

    @property
    def interval(self) -> BarInterval:
        """The decision-bar interval this strategy operates on."""
        ...

    def generate_signals(self, candles: Sequence[Candle]) -> Sequence[StrategySignal]:
        """Emit point-in-time signals from an ascending series of ``candles``.

        A signal at bar *t*'s close is executed at bar *t+1*'s open by the
        backtester (Inviolable Rule 2); the spec itself must not assume same-bar
        fills or use any data after each signal's ``asof``.
        """
        ...
