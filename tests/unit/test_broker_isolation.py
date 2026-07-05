"""Architecture guard: the Kite SDK is imported ONLY inside data/brokers/.

Enforces Part I §1 / Part II — nothing outside ``lab.data.brokers`` may depend on
``kiteconnect``; everything else programs to the ``BrokerAdapter`` Protocol.
"""

from __future__ import annotations

from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "lab"
BROKERS = SRC / "data" / "brokers"


def test_kiteconnect_imported_only_in_brokers() -> None:
    offenders: list[str] = []
    for path in SRC.rglob("*.py"):
        if BROKERS in path.parents:
            continue
        if "kiteconnect" in path.read_text(encoding="utf-8"):
            offenders.append(str(path.relative_to(SRC)))
    assert not offenders, f"kiteconnect referenced outside data/brokers/: {offenders}"
