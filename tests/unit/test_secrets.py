"""Tests for the secrets interface: env/.env resolution, missing-secret errors,
and automatic registration of resolved values for log redaction.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from lab.core.logging import configure_logging, get_logger
from lab.core.secrets import EnvSecretsProvider, MissingSecretError, SecretsProvider


def test_get_returns_value() -> None:
    provider = EnvSecretsProvider({"KITE_API_KEY": "abc123"})
    assert provider.get("KITE_API_KEY") == "abc123"


def test_missing_secret_raises_clear_error() -> None:
    provider = EnvSecretsProvider({})
    with pytest.raises(MissingSecretError) as exc_info:
        provider.get("KITE_API_SECRET")
    assert exc_info.value.name == "KITE_API_SECRET"
    assert "never be committed" in str(exc_info.value)


def test_empty_value_is_treated_as_missing() -> None:
    provider = EnvSecretsProvider({"KITE_API_KEY": ""})
    with pytest.raises(MissingSecretError):
        provider.get("KITE_API_KEY")


def test_get_optional_returns_default() -> None:
    provider = EnvSecretsProvider({})
    assert provider.get_optional("NOPE", default="fallback") == "fallback"
    assert provider.get_optional("NOPE") is None


def test_provider_satisfies_protocol() -> None:
    provider: SecretsProvider = EnvSecretsProvider({"A": "1"})
    assert isinstance(provider, SecretsProvider)
    assert provider.get("A") == "1"


def test_dotenv_loaded_and_env_wins(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "# a comment\nexport KITE_API_KEY='from_dotenv'\nKITE_API_SECRET=secret_from_dotenv\n",
        encoding="utf-8",
    )
    provider = EnvSecretsProvider.from_environment(
        environ={"KITE_API_KEY": "from_env"}, dotenv_path=dotenv
    )
    # Real environment overrides the .env file.
    assert provider.get("KITE_API_KEY") == "from_env"
    # .env-only values are still available, with quotes/export stripped.
    assert provider.get("KITE_API_SECRET") == "secret_from_dotenv"


def test_resolved_secret_is_redacted_in_logs() -> None:
    # Resolving a secret must register it so it can never surface in a log line.
    provider = EnvSecretsProvider({"KITE_API_SECRET": "zzz-super-secret"})
    resolved = provider.get("KITE_API_SECRET")

    buffer = io.StringIO()
    configure_logging(renderer="json", stream=buffer)
    get_logger("test.secrets").info("auth", note=f"token is {resolved}")

    record = json.loads(buffer.getvalue().strip())
    assert "zzz-super-secret" not in json.dumps(record)
    assert "***REDACTED***" in record["note"]
