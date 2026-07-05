"""Tests for structured logging: JSON structure, IST timestamps, level filtering,
correlation IDs, and secret redaction.
"""

from __future__ import annotations

import io
import json
from collections.abc import Iterator
from typing import Any

import pytest
import structlog

from lab.core.logging import (
    bind_correlation_id,
    clear_context,
    configure_logging,
    correlation_context,
    get_logger,
    register_secret_value,
)


@pytest.fixture(autouse=True)
def _reset_structlog() -> Iterator[None]:
    """Isolate each test from global structlog state."""
    clear_context()
    yield
    clear_context()
    structlog.reset_defaults()


def _lines(buffer: io.StringIO) -> list[dict[str, Any]]:
    return [json.loads(line) for line in buffer.getvalue().strip().splitlines() if line]


def test_json_event_is_structured_with_ist_timestamp() -> None:
    buffer = io.StringIO()
    configure_logging(renderer="json", level="INFO", stream=buffer)
    get_logger("test.log").info("study_started", symbol="RELIANCE")

    (record,) = _lines(buffer)
    assert record["event"] == "study_started"
    assert record["level"] == "info"
    assert record["symbol"] == "RELIANCE"
    # IST offset proves the timestamp is in Asia/Kolkata.
    assert record["timestamp"].endswith("+05:30")


def test_level_filtering_suppresses_below_threshold() -> None:
    buffer = io.StringIO()
    configure_logging(renderer="json", level="WARNING", stream=buffer)
    log = get_logger("test.level")
    log.info("hidden")
    log.warning("shown")

    records = _lines(buffer)
    assert len(records) == 1
    assert records[0]["event"] == "shown"


def test_sensitive_keys_are_redacted() -> None:
    buffer = io.StringIO()
    configure_logging(renderer="json", stream=buffer)
    get_logger("test.redact").info("auth", access_token="live-token", api_key="k", user="alice")

    (record,) = _lines(buffer)
    assert record["access_token"] == "***REDACTED***"
    assert record["api_key"] == "***REDACTED***"
    assert record["user"] == "alice"


def test_registered_secret_value_is_scrubbed_anywhere() -> None:
    register_secret_value("hunter2-value")
    buffer = io.StringIO()
    configure_logging(renderer="json", stream=buffer)
    get_logger("test.scrub").info("note", detail="prefix hunter2-value suffix")

    (record,) = _lines(buffer)
    assert "hunter2-value" not in json.dumps(record)
    assert "***REDACTED***" in record["detail"]


def test_correlation_context_scopes_the_id() -> None:
    buffer = io.StringIO()
    configure_logging(renderer="json", stream=buffer)
    log = get_logger("test.corr")

    with correlation_context("run-42") as cid:
        assert cid == "run-42"
        log.info("inside")
    log.info("outside")

    inside, outside = _lines(buffer)
    assert inside["correlation_id"] == "run-42"
    assert "correlation_id" not in outside


def test_bind_correlation_id_generates_and_returns() -> None:
    buffer = io.StringIO()
    configure_logging(renderer="json", stream=buffer)
    cid = bind_correlation_id()
    assert cid  # a generated hex id
    get_logger("test.bind").info("evt")

    (record,) = _lines(buffer)
    assert record["correlation_id"] == cid


def test_unknown_level_and_renderer_raise() -> None:
    with pytest.raises(ValueError, match="log level"):
        configure_logging(level="LOUD")
    with pytest.raises(ValueError, match="renderer"):
        configure_logging(renderer="xml")
