"""The trading universe as a provenance-stamped artifact (Phase 3 prep).

A study's universe is never an ad-hoc list — it is a versioned file carrying its
``as_of`` date and ``source``, so every result can be traced to exactly which
constituent snapshot it ran on. Loading fails loudly if the provenance markers
are missing (that is the whole point of the artifact).

The current backfill is survivor-only (today's members over past data); the
``backfill:`` block of the artifact records the discrepancy, and
``docs/RESEARCH_FINDINGS.md`` documents the bias. See also the provisional /
upper-bound stamp applied to narrow-margin study passes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml


def _require_date(value: Any, field: str) -> date:
    """Coerce an ISO string or ``date`` to a ``date``; raise on anything else."""
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"universe {field!r} is not an ISO date: {value!r}") from exc
    raise ValueError(f"universe {field!r} must be an ISO date; got {value!r}")


@dataclass(frozen=True, slots=True)
class Universe:
    """A dated, sourced constituent set (the provenance-stamped universe)."""

    index: str
    as_of: date  # effective date of this constituent set
    verified_on: date  # when it was read from the source
    source: str  # authoritative provenance
    constituents: tuple[str, ...]

    def __post_init__(self) -> None:
        """Fail loudly if the artifact is empty (a universe must have members)."""
        if not self.constituents:
            raise ValueError("universe artifact has no constituents")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> Universe:
        """Build a universe from a parsed artifact; require the provenance markers."""
        for required in ("as_of", "source", "constituents"):
            if required not in mapping:
                raise ValueError(f"universe artifact missing required key: {required!r}")
        as_of = _require_date(mapping["as_of"], "as_of")
        verified_on = _require_date(mapping.get("verified_on", mapping["as_of"]), "verified_on")
        return cls(
            index=str(mapping.get("index", "")),
            as_of=as_of,
            verified_on=verified_on,
            source=str(mapping["source"]),
            constituents=tuple(str(symbol) for symbol in mapping["constituents"]),
        )


def load_universe(path: Path) -> Universe:
    """Load a provenance-stamped universe artifact from ``path``."""
    data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return Universe.from_mapping(data)
