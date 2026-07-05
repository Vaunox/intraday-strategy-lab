"""Architecture guard: pyarrow/pandas are imported ONLY inside data/store/.

The storage-client analogue of ``test_broker_isolation``: nothing outside
``lab.data.store`` may depend on ``pyarrow``/``pandas`` — everything else programs
to the ``Repository`` Protocol (Part I §1). Guards the isolation against
regression, since it was previously true only by convention.
"""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "lab"
STORE = SRC / "data" / "store"
_IMPORT = re.compile(r"^\s*(?:import|from)\s+(?:pyarrow|pandas)\b", re.MULTILINE)


def test_pyarrow_and_pandas_imported_only_in_store() -> None:
    offenders: list[str] = []
    for path in SRC.rglob("*.py"):
        if STORE in path.parents:
            continue
        if _IMPORT.search(path.read_text(encoding="utf-8")):
            offenders.append(str(path.relative_to(SRC)))
    assert not offenders, f"pyarrow/pandas imported outside data/store/: {offenders}"
