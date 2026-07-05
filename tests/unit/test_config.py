"""Tests for the layered, typed configuration loader."""

from __future__ import annotations

from datetime import date, time
from pathlib import Path

import pytest

from lab.core.config import ConfigError, load_settings
from lab.core.logging import DEFAULT_REDACT_KEYS

REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_CONFIG = REPO_ROOT / "config"

_DEFAULT_BODY = """
environment: dev
logging:
  level: INFO
  renderer: console
  redact_keys: ["custom_key"]
calendar:
  timezone: "Asia/Kolkata"
  session:
    pre_open_start: "09:00"
    open: "09:15"
    close: "15:30"
    square_off: "15:20"
    post_close_start: "15:40"
    post_close_end: "16:00"
  holidays:
    - "2024-08-15"
"""


def _make_config_tree(root: Path, *, default: str = _DEFAULT_BODY, envs: dict[str, str]) -> Path:
    cfg = root / "config"
    (cfg / "env").mkdir(parents=True)
    (cfg / "default.yaml").write_text(default, encoding="utf-8")
    for name, body in envs.items():
        (cfg / "env" / f"{name}.yaml").write_text(body, encoding="utf-8")
    return cfg


def test_real_repo_config_loads_and_types() -> None:
    # The actual shipped config must parse and build a typed Settings tree.
    settings = load_settings("dev", config_dir=REPO_CONFIG, environ={})
    assert settings.environment == "dev"
    assert settings.calendar.timezone == "Asia/Kolkata"
    assert settings.calendar.session.open == time(9, 15)
    assert settings.calendar.session.square_off == time(15, 20)
    assert date(2024, 8, 15) in settings.calendar.holidays
    assert settings.logging.renderer == "console"  # dev overlay


def test_env_overlay_overrides_default(tmp_path: Path) -> None:
    cfg = _make_config_tree(
        tmp_path,
        envs={
            "dev": "logging:\n  level: DEBUG\n",
            "research": "logging:\n  level: INFO\n  renderer: json\n",
        },
    )
    dev = load_settings("dev", config_dir=cfg, environ={})
    research = load_settings("research", config_dir=cfg, environ={})
    assert dev.logging.level == "DEBUG"
    assert dev.logging.renderer == "console"  # inherited from default
    assert research.logging.renderer == "json"


def test_env_var_overrides_win(tmp_path: Path) -> None:
    cfg = _make_config_tree(tmp_path, envs={"dev": "logging:\n  level: DEBUG\n"})
    settings = load_settings("dev", config_dir=cfg, environ={"LAB__LOGGING__LEVEL": "WARNING"})
    assert settings.logging.level == "WARNING"


def test_env_selected_by_lab_env_var(tmp_path: Path) -> None:
    cfg = _make_config_tree(
        tmp_path,
        envs={"dev": "logging:\n  level: DEBUG\n", "research": "logging:\n  renderer: json\n"},
    )
    settings = load_settings(config_dir=cfg, environ={"LAB_ENV": "research"})
    assert settings.environment == "research"
    assert settings.logging.renderer == "json"


def test_redact_keys_union_defaults_and_config(tmp_path: Path) -> None:
    cfg = _make_config_tree(tmp_path, envs={"dev": "{}\n"})
    settings = load_settings("dev", config_dir=cfg, environ={})
    # Config extras are added; built-in security keys are always present.
    assert "custom_key" in settings.logging.redact_keys
    assert DEFAULT_REDACT_KEYS.issubset(set(settings.logging.redact_keys))


def test_unknown_environment_raises(tmp_path: Path) -> None:
    cfg = _make_config_tree(tmp_path, envs={"dev": "{}\n"})
    with pytest.raises(ConfigError, match="unknown environment"):
        load_settings("staging", config_dir=cfg, environ={})


def test_missing_base_config_raises(tmp_path: Path) -> None:
    (tmp_path / "config" / "env").mkdir(parents=True)
    with pytest.raises(ConfigError, match="base config not found"):
        load_settings("dev", config_dir=tmp_path / "config", environ={})


def test_missing_required_key_raises(tmp_path: Path) -> None:
    bad_default = """
logging:
  level: INFO
  renderer: console
calendar:
  session:
    pre_open_start: "09:00"
    open: "09:15"
    close: "15:30"
    square_off: "15:20"
    post_close_start: "15:40"
    post_close_end: "16:00"
  holidays: []
"""  # missing calendar.timezone
    cfg = _make_config_tree(tmp_path, default=bad_default, envs={"dev": "{}\n"})
    with pytest.raises(ConfigError, match=r"calendar\.timezone"):
        load_settings("dev", config_dir=cfg, environ={})


def test_unquoted_time_is_rejected_with_clear_error(tmp_path: Path) -> None:
    # An unquoted time with a leading non-zero digit (e.g. 15:30) is parsed by
    # YAML 1.1 as a base-60 integer (930), not a string. The loader must reject
    # the wrong type clearly rather than silently mis-parse it.
    bad_default = _DEFAULT_BODY.replace('close: "15:30"', "close: 15:30")
    cfg = _make_config_tree(tmp_path, default=bad_default, envs={"dev": "{}\n"})
    with pytest.raises(ConfigError, match=r"calendar\.session\.close"):
        load_settings("dev", config_dir=cfg, environ={})


def test_env_var_scalar_is_typed(tmp_path: Path) -> None:
    # LAB__ overrides are parsed as YAML scalars, so a nested value round-trips.
    cfg = _make_config_tree(tmp_path, envs={"dev": "{}\n"})
    settings = load_settings(
        "dev", config_dir=cfg, environ={"LAB__CALENDAR__TIMEZONE": "Asia/Kolkata"}
    )
    assert settings.calendar.timezone == "Asia/Kolkata"


def test_india_tz_constant_matches_config_timezone() -> None:
    # The INDIA_TZ fallback constant (used as a default in Phase-2 modules) must not
    # silently drift from the authoritative config value everything derives IST from.
    from lab.core.constants import INDIA_TZ

    settings = load_settings("dev", config_dir=REPO_CONFIG, environ={})
    assert settings.calendar.timezone == INDIA_TZ
