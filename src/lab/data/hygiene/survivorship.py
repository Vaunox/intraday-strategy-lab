"""Point-in-time constituents / survivorship control (Phase 1, P1.4).

Survivorship bias — testing only names that exist *today* — is designed out
structurally (Part III Layer 1): the universe is defined by point-in-time
membership records that include delisted and renamed names. Asking which symbols
were constituents on a past date returns exactly the names tradable *then*, not a
today-survivors filter.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True, slots=True)
class ConstituentRecord:
    """A symbol's membership window in the universe.

    ``to_date`` is ``None`` while the name is still listed; a set ``to_date``
    marks a delisting. ``renamed_to`` links an old symbol to its successor.
    """

    symbol: str
    from_date: date
    to_date: date | None = None
    renamed_to: str | None = None

    def is_active_on(self, day: date) -> bool:
        """Return whether the symbol was a constituent on ``day``."""
        return self.from_date <= day and (self.to_date is None or day <= self.to_date)


class PointInTimeUniverse:
    """A survivorship-correct universe backed by membership records."""

    def __init__(self, records: Iterable[ConstituentRecord]) -> None:
        """Build the universe from constituent membership records."""
        self._records: tuple[ConstituentRecord, ...] = tuple(records)

    def constituents_on(self, day: date) -> set[str]:
        """Return the symbols tradable on ``day`` (delisted names included historically)."""
        return {record.symbol for record in self._records if record.is_active_on(day)}

    def all_symbols(self) -> set[str]:
        """Return every symbol ever a constituent (so backfill covers delisted names)."""
        return {record.symbol for record in self._records}

    def resolve_current_symbol(self, symbol: str) -> str:
        """Follow any rename chain from ``symbol`` to its most recent symbol."""
        by_symbol = {record.symbol: record for record in self._records}
        seen: set[str] = set()
        current = symbol
        while current in by_symbol:
            successor = by_symbol[current].renamed_to
            if successor is None or current in seen:  # end of chain, or cyclic guard
                break
            seen.add(current)
            current = successor
        return current
