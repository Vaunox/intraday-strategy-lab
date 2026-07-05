"""Layered, typed configuration loader (Part I §2).

One source of truth, resolved in strict precedence order:

    config/default.yaml  <  config/env/<environment>.yaml  <  LAB__ environment vars

The result is a frozen, typed :class:`Settings` tree. Every run is reproducible
from its configuration; business logic reads typed fields here instead of
carrying literals. Missing or malformed keys raise :class:`ConfigError` loudly at
load time rather than failing deep in a run.

Environment-variable overrides use a ``LAB__`` prefix and ``__`` to separate
nested keys, e.g. ``LAB__LOGGING__LEVEL=DEBUG`` overrides ``logging.level``.
Values are parsed as YAML scalars, so numbers and booleans convert naturally.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

import yaml

from lab.core.constants import (
    CONFIG_ENV_VAR,
    DEFAULT_ENVIRONMENT,
    ENV_OVERRIDE_DELIMITER,
    ENV_OVERRIDE_PREFIX,
)
from lab.core.logging import DEFAULT_REDACT_KEYS, configure_logging


class ConfigError(RuntimeError):
    """Raised when configuration is missing, malformed, or of the wrong type."""


@dataclass(frozen=True, slots=True)
class LoggingSettings:
    """Resolved logging configuration."""

    level: str
    renderer: str
    redact_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SessionSettings:
    """The intraday session boundaries for the exchange (all IST)."""

    pre_open_start: time
    open: time
    close: time
    square_off: time
    post_close_start: time
    post_close_end: time


@dataclass(frozen=True, slots=True)
class CalendarSettings:
    """Trading-calendar configuration: timezone, session bounds, holidays."""

    timezone: str
    session: SessionSettings
    holidays: tuple[date, ...]


@dataclass(frozen=True, slots=True)
class Settings:
    """The fully-resolved, typed configuration for a run.

    ``raw`` exposes the merged mapping for sections not yet promoted to typed
    fields (added by later phases); prefer the typed accessors.
    """

    environment: str
    logging: LoggingSettings
    calendar: CalendarSettings
    raw: Mapping[str, Any]


# --------------------------------------------------------------------------- #
# Merge / load helpers                                                        #
# --------------------------------------------------------------------------- #
def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file into a dict, or raise :class:`ConfigError`."""
    data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    return data


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base`` (override wins; dicts merge)."""
    result: dict[str, Any] = dict(base)
    for key, value in override.items():
        existing = result.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            result[key] = _deep_merge(existing, value)
        else:
            result[key] = value
    return result


def _parse_scalar(raw: str) -> Any:
    """Parse an override value as a YAML scalar, falling back to the raw string."""
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def _set_nested(target: dict[str, Any], segments: list[str], value: Any) -> None:
    """Set ``value`` at the nested ``segments`` path, creating dicts as needed."""
    cursor = target
    for segment in segments[:-1]:
        nxt = cursor.get(segment)
        if not isinstance(nxt, dict):
            nxt = {}
            cursor[segment] = nxt
        cursor = nxt
    cursor[segments[-1]] = value


def _env_overrides(environ: Mapping[str, str]) -> dict[str, Any]:
    """Extract ``LAB__``-prefixed environment variables into a nested dict."""
    overrides: dict[str, Any] = {}
    for key, value in environ.items():
        if not key.startswith(ENV_OVERRIDE_PREFIX):
            continue
        remainder = key[len(ENV_OVERRIDE_PREFIX) :]
        segments = [s.lower() for s in remainder.split(ENV_OVERRIDE_DELIMITER) if s]
        if not segments:
            continue
        _set_nested(overrides, segments, _parse_scalar(value))
    return overrides


# --------------------------------------------------------------------------- #
# Typed extraction helpers                                                     #
# --------------------------------------------------------------------------- #
def _require(mapping: Mapping[str, Any], key: str, path: str) -> Any:
    """Return ``mapping[key]`` or raise a :class:`ConfigError` naming ``path``."""
    if key not in mapping:
        raise ConfigError(f"missing required config key: {path}")
    return mapping[key]


def _require_mapping(mapping: Mapping[str, Any], key: str, path: str) -> dict[str, Any]:
    """Return a required nested mapping at ``key``."""
    value = _require(mapping, key, path)
    if not isinstance(value, dict):
        raise ConfigError(f"config key {path} must be a mapping, got {type(value).__name__}")
    return value


def _require_str(mapping: Mapping[str, Any], key: str, path: str) -> str:
    """Return a required string at ``key``."""
    value = _require(mapping, key, path)
    if not isinstance(value, str):
        raise ConfigError(f"config key {path} must be a string, got {type(value).__name__}")
    return value


def _require_list(mapping: Mapping[str, Any], key: str, path: str) -> list[Any]:
    """Return a required list at ``key``."""
    value = _require(mapping, key, path)
    if not isinstance(value, list):
        raise ConfigError(f"config key {path} must be a list, got {type(value).__name__}")
    return value


def _coerce_time(value: Any, path: str) -> time:
    """Coerce a ``"HH:MM"`` string (or ``time``) to a :class:`~datetime.time`."""
    if isinstance(value, time):
        return value
    if isinstance(value, str):
        try:
            return time.fromisoformat(value)
        except ValueError as exc:
            raise ConfigError(f"config key {path} is not a valid HH:MM time: {value!r}") from exc
    raise ConfigError(
        f"config key {path} must be a quoted 'HH:MM' string, got {type(value).__name__} "
        f"(unquoted times are misread by YAML — quote them)"
    )


def _coerce_date(value: Any, path: str) -> date:
    """Coerce an ISO date string (or ``date``/``datetime``) to a :class:`~datetime.date`."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ConfigError(f"config key {path} is not a valid ISO date: {value!r}") from exc
    raise ConfigError(f"config key {path} must be an ISO date, got {type(value).__name__}")


def _effective_redact_keys(configured: Any, path: str) -> tuple[str, ...]:
    """Union the built-in redaction keys with any configured extras.

    The defaults are always applied — configuration may only add keys, never
    weaken redaction by omission (Part I §8).
    """
    if not isinstance(configured, list):
        raise ConfigError(f"config key {path} must be a list, got {type(configured).__name__}")
    extra = {str(item).lower() for item in configured}
    return tuple(sorted(DEFAULT_REDACT_KEYS | extra))


def _build_session(mapping: Mapping[str, Any]) -> SessionSettings:
    """Build :class:`SessionSettings` from the ``calendar.session`` mapping."""
    path = "calendar.session"
    return SessionSettings(
        pre_open_start=_coerce_time(
            _require(mapping, "pre_open_start", f"{path}.pre_open_start"), f"{path}.pre_open_start"
        ),
        open=_coerce_time(_require(mapping, "open", f"{path}.open"), f"{path}.open"),
        close=_coerce_time(_require(mapping, "close", f"{path}.close"), f"{path}.close"),
        square_off=_coerce_time(
            _require(mapping, "square_off", f"{path}.square_off"), f"{path}.square_off"
        ),
        post_close_start=_coerce_time(
            _require(mapping, "post_close_start", f"{path}.post_close_start"),
            f"{path}.post_close_start",
        ),
        post_close_end=_coerce_time(
            _require(mapping, "post_close_end", f"{path}.post_close_end"), f"{path}.post_close_end"
        ),
    )


def _build_settings(environment: str, merged: Mapping[str, Any]) -> Settings:
    """Construct the typed :class:`Settings` tree, validating as it goes."""
    logging_map = _require_mapping(merged, "logging", "logging")
    logging_settings = LoggingSettings(
        level=_require_str(logging_map, "level", "logging.level"),
        renderer=_require_str(logging_map, "renderer", "logging.renderer"),
        redact_keys=_effective_redact_keys(
            logging_map.get("redact_keys", []), "logging.redact_keys"
        ),
    )

    calendar_map = _require_mapping(merged, "calendar", "calendar")
    session_map = _require_mapping(calendar_map, "session", "calendar.session")
    holidays_raw = _require_list(calendar_map, "holidays", "calendar.holidays")
    calendar_settings = CalendarSettings(
        timezone=_require_str(calendar_map, "timezone", "calendar.timezone"),
        session=_build_session(session_map),
        holidays=tuple(
            _coerce_date(item, f"calendar.holidays[{i}]") for i, item in enumerate(holidays_raw)
        ),
    )

    return Settings(
        environment=environment,
        logging=logging_settings,
        calendar=calendar_settings,
        raw=merged,
    )


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #
def _resolve_config_dir(config_dir: Path | None, environ: Mapping[str, str]) -> Path:
    """Resolve the config directory: explicit arg, then ``LAB_CONFIG_DIR``, then cwd."""
    if config_dir is not None:
        return config_dir
    from_env = environ.get("LAB_CONFIG_DIR")
    if from_env:
        return Path(from_env)
    return Path.cwd() / "config"


def load_settings(
    environment: str | None = None,
    *,
    config_dir: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Settings:
    """Load and merge configuration into a typed :class:`Settings`.

    Args:
        environment: Environment name; defaults to ``$LAB_ENV`` or ``dev``.
        config_dir: Directory holding ``default.yaml`` and ``env/``; defaults to
            ``$LAB_CONFIG_DIR`` or ``./config``.
        environ: Environment mapping for overrides; defaults to ``os.environ``.

    Returns:
        The fully-resolved, validated configuration.

    Raises:
        ConfigError: If a config file or required key is missing or malformed.
    """
    env_map: Mapping[str, str] = os.environ if environ is None else environ
    resolved_env = environment or env_map.get(CONFIG_ENV_VAR) or DEFAULT_ENVIRONMENT
    cfg_dir = _resolve_config_dir(config_dir, env_map)

    default_path = cfg_dir / "default.yaml"
    if not default_path.exists():
        raise ConfigError(f"base config not found: {default_path}")

    env_path = cfg_dir / "env" / f"{resolved_env}.yaml"
    if not env_path.exists():
        available = sorted(p.stem for p in (cfg_dir / "env").glob("*.yaml"))
        raise ConfigError(
            f"unknown environment {resolved_env!r}: {env_path} does not exist "
            f"(available: {', '.join(available) or 'none'})"
        )

    merged = _deep_merge(_load_yaml(default_path), _load_yaml(env_path))
    merged = _deep_merge(merged, _env_overrides(env_map))
    return _build_settings(resolved_env, merged)


def configure_logging_from_settings(settings: Settings) -> None:
    """Configure structured logging from a loaded :class:`Settings`.

    This is the single wiring point between config and logging; it keeps the
    logging module free of any dependency on config.
    """
    configure_logging(
        level=settings.logging.level,
        renderer=settings.logging.renderer,
        redact_keys=settings.logging.redact_keys,
        timezone=settings.calendar.timezone,
    )
