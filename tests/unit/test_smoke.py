"""Smoke test: the whole package tree imports cleanly.

Satisfies the Gate 0 "CI green on a test" bar and guards against import-time
errors (circular imports, missing deps, broken package layout) across the
``lab`` namespace.
"""

from __future__ import annotations

import importlib


def test_package_tree_imports() -> None:
    for module in (
        "lab",
        "lab.core",
        "lab.core.constants",
        "lab.data",
        "lab.data.brokers",
        "lab.data.ingest",
        "lab.data.store",
        "lab.data.hygiene",
        "lab.data.features",
        "lab.research",
        "lab.research.validation",
        "lab.research.strategies",
        "lab.research.trials",
        "lab.research.reports",
    ):
        assert importlib.import_module(module) is not None
