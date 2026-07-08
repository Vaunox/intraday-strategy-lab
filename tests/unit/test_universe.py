"""Tests for the provenance-stamped universe artifact and loader."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from lab.data.universe import Universe, load_universe

REPO_UNIVERSE = Path(__file__).resolve().parents[2] / "config" / "universe" / "nifty50.yaml"


def test_repo_nifty50_artifact_loads_with_provenance() -> None:
    universe = load_universe(REPO_UNIVERSE)
    assert universe.index == "NIFTY 50"
    assert len(universe.constituents) == 50
    assert isinstance(universe.as_of, date)  # the whole point: a dated snapshot
    assert universe.as_of >= date(2025, 9, 30)  # post the 2025 reconstitution
    assert universe.source  # non-empty provenance


def test_artifact_reflects_corrected_constituents() -> None:
    members = set(load_universe(REPO_UNIVERSE).constituents)
    # New members that the earlier stale list was missing.
    assert {"ETERNAL", "JIOFIN", "INDIGO", "MAXHEALTH", "TMPV"} <= members
    # Names the stale list wrongly carried and must NOT be in the current set.
    assert members.isdisjoint(
        {"LTIM", "INDUSINDBK", "HEROMOTOCO", "BPCL", "BRITANNIA", "TATAMOTORS"}
    )


def test_from_mapping_requires_provenance_markers() -> None:
    base = {"as_of": "2025-09-30", "source": "x", "constituents": ["RELIANCE"]}
    for missing in ("as_of", "source", "constituents"):
        bad = {k: v for k, v in base.items() if k != missing}
        with pytest.raises(ValueError, match=missing):
            Universe.from_mapping(bad)


def test_from_mapping_rejects_empty_and_bad_date() -> None:
    with pytest.raises(ValueError, match="no constituents"):
        Universe.from_mapping({"as_of": "2025-09-30", "source": "x", "constituents": []})
    with pytest.raises(ValueError, match="ISO date"):
        Universe.from_mapping({"as_of": "not-a-date", "source": "x", "constituents": ["A"]})


def test_from_mapping_accepts_native_date() -> None:
    universe = Universe.from_mapping(
        {"as_of": date(2025, 9, 30), "source": "x", "constituents": ["RELIANCE", "TCS"]}
    )
    assert universe.as_of == date(2025, 9, 30)
    assert universe.verified_on == date(2025, 9, 30)  # defaults to as_of
