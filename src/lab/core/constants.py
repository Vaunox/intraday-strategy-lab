"""Invariant application constants.

These are structural facts about the application itself, not tunable parameters.
Anything a run might legitimately vary — universe, lookbacks, thresholds, rates,
session times, paths — is versioned configuration under ``config/`` (Part I §2),
never here.
"""

from __future__ import annotations

from typing import Final

#: Logger namespace root and package slug.
PACKAGE_NAME: Final = "lab"

#: Environment variable that selects the active configuration environment
#: (e.g. ``dev``, ``research``). See :mod:`lab.core.config`.
CONFIG_ENV_VAR: Final = "LAB_ENV"

#: Default configuration environment when ``LAB_ENV`` is unset.
DEFAULT_ENVIRONMENT: Final = "dev"

#: Prefix marking environment variables that override configuration keys.
#: ``LAB__LOGGING__LEVEL=DEBUG`` overrides ``logging.level`` (Part I §2 —
#: default.yaml ← env file ← environment variables).
ENV_OVERRIDE_PREFIX: Final = "LAB__"

#: Delimiter separating nested keys within an override variable name.
ENV_OVERRIDE_DELIMITER: Final = "__"

#: IANA timezone the exchange (and therefore every timestamp and log) operates in.
#: The authoritative value lives in ``config`` (``calendar.timezone``); this
#: constant is the fallback/label used where config is not yet available.
INDIA_TZ: Final = "Asia/Kolkata"
