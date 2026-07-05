"""Structured logging, configured once for the whole program (Part I §8).

Uses ``structlog`` to emit structured events (never bare prints) with:

* **levels** — DEBUG/INFO/WARNING/ERROR/CRITICAL, filtered efficiently;
* **correlation IDs** — bound via context vars so every event in a run is
  traceable (:func:`correlation_context`, :func:`bind_correlation_id`);
* **IST timestamps** — every event is stamped in ``Asia/Kolkata``;
* **secret redaction** — values under sensitive keys, and any value registered
  through :func:`register_secret_value`, are masked before rendering. Secrets are
  never logged (Part I §2/§8).

Call :func:`configure_logging` once at process start, then obtain loggers with
:func:`get_logger`.
"""

from __future__ import annotations

import logging as stdlib_logging
import uuid
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import TextIO, cast
from zoneinfo import ZoneInfo

import structlog
from structlog.typing import EventDict, FilteringBoundLogger, WrappedLogger

from lab.core.constants import INDIA_TZ, PACKAGE_NAME

#: Substrings (case-insensitive) whose keys are always masked in log output.
DEFAULT_REDACT_KEYS: frozenset[str] = frozenset(
    {
        "token",
        "secret",
        "password",
        "passwd",
        "api_key",
        "apikey",
        "access_token",
        "request_token",
        "authorization",
        "credential",
        "private_key",
    }
)

#: Mask substituted for redacted content.
REDACTION_MASK = "***REDACTED***"

# Process-global registry of literal secret VALUES to scrub from any event.
# This is deliberately module-level: redaction is a cross-cutting security
# guarantee (like the logging configuration itself, which structlog also holds
# globally) and must catch a secret wherever it surfaces, not just where a key
# name looks sensitive. It holds no business state.
_REGISTERED_SECRET_VALUES: set[str] = set()


def register_secret_value(value: str) -> None:
    """Register a literal secret value to be redacted from all future log events.

    Called by the secrets layer whenever a secret is resolved, so that if the
    value ever reaches a log record — under any key — it is masked.

    Args:
        value: The secret string to scrub. Empty strings are ignored.
    """
    if value:
        _REGISTERED_SECRET_VALUES.add(value)


def _make_ist_timestamper(tz_name: str) -> object:
    """Build a processor that stamps each event with an ISO-8601 IST timestamp."""
    tz = ZoneInfo(tz_name)

    def add_timestamp(_logger: WrappedLogger, _name: str, event_dict: EventDict) -> EventDict:
        event_dict["timestamp"] = datetime.now(tz).isoformat()
        return event_dict

    return add_timestamp


def _make_redactor(redact_keys: Iterable[str]) -> object:
    """Build a processor that masks sensitive keys and registered secret values."""
    key_patterns = tuple(k.lower() for k in redact_keys)

    def redact(_logger: WrappedLogger, _name: str, event_dict: EventDict) -> EventDict:
        for key in list(event_dict.keys()):
            lowered = key.lower()
            if any(pattern in lowered for pattern in key_patterns):
                event_dict[key] = REDACTION_MASK
        if _REGISTERED_SECRET_VALUES:
            for key, value in list(event_dict.items()):
                if isinstance(value, str):
                    masked = value
                    for secret in _REGISTERED_SECRET_VALUES:
                        if secret in masked:
                            masked = masked.replace(secret, REDACTION_MASK)
                    event_dict[key] = masked
        return event_dict

    return redact


def _resolve_level(level: str) -> int:
    """Translate a level name (e.g. ``"INFO"``) to its numeric value, or raise."""
    numeric = stdlib_logging.getLevelNamesMapping().get(level.upper())
    if numeric is None:
        raise ValueError(f"unknown log level: {level!r}")
    return numeric


def configure_logging(
    *,
    level: str = "INFO",
    renderer: str = "console",
    redact_keys: Iterable[str] = DEFAULT_REDACT_KEYS,
    timezone: str = INDIA_TZ,
    stream: TextIO | None = None,
) -> None:
    """Configure structlog once for the process.

    Idempotent — safe to call again (e.g. to switch renderer in tests); the last
    call wins.

    Args:
        level: Minimum level to emit (``DEBUG``..``CRITICAL``).
        renderer: ``"console"`` for human-readable dev output, ``"json"`` for
            machine-parseable output.
        redact_keys: Case-insensitive key substrings to mask.
        timezone: IANA timezone for timestamps (IST by default).
        stream: Optional output stream (used in tests); defaults to stdout.

    Raises:
        ValueError: If ``level`` or ``renderer`` is not recognized.
    """
    final_renderer: structlog.types.Processor
    if renderer == "json":
        final_renderer = structlog.processors.JSONRenderer()
    elif renderer == "console":
        final_renderer = structlog.dev.ConsoleRenderer(colors=False)
    else:
        raise ValueError(f"unknown renderer: {renderer!r} (expected 'console' or 'json')")

    processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        cast("structlog.types.Processor", _make_ist_timestamper(timezone)),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        cast("structlog.types.Processor", _make_redactor(redact_keys)),
        final_renderer,
    ]

    logger_factory = (
        structlog.WriteLoggerFactory(file=stream)
        if stream is not None
        else structlog.PrintLoggerFactory()
    )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(_resolve_level(level)),
        context_class=dict,
        logger_factory=logger_factory,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> FilteringBoundLogger:
    """Return a bound logger under the ``lab`` namespace.

    Args:
        name: Optional sub-name (e.g. a module); defaults to the package root.
    """
    return cast("FilteringBoundLogger", structlog.get_logger(name or PACKAGE_NAME))


def bind_correlation_id(correlation_id: str | None = None) -> str:
    """Bind a correlation ID into the logging context for this execution.

    Args:
        correlation_id: An explicit ID; if omitted, a random one is generated.

    Returns:
        The correlation ID that was bound (so callers can log/propagate it).
    """
    cid = correlation_id or uuid.uuid4().hex
    structlog.contextvars.bind_contextvars(correlation_id=cid)
    return cid


def clear_context() -> None:
    """Clear all context-var bindings (correlation ID and any bound fields)."""
    structlog.contextvars.clear_contextvars()


@contextmanager
def correlation_context(correlation_id: str | None = None) -> Iterator[str]:
    """Scope a correlation ID to a block, restoring prior context on exit.

    Args:
        correlation_id: An explicit ID; if omitted, a random one is generated.

    Yields:
        The correlation ID bound for the duration of the block.
    """
    cid = correlation_id or uuid.uuid4().hex
    with structlog.contextvars.bound_contextvars(correlation_id=cid):
        yield cid
