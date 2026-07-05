"""The single interface through which secrets are read.

Secrets (Kite API key/secret, daily tokens) never live in code or in versioned
config — only in environment variables or an untracked ``.env`` file (Part I §2).
Business logic depends on the :class:`SecretsProvider` Protocol, so the source
can be swapped without touching callers. A missing secret raises
:class:`MissingSecretError` loudly rather than defaulting silently. Every
resolved value is registered with the logging layer for redaction, so a secret
can never leak into a log line.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, runtime_checkable

from lab.core.logging import register_secret_value


class MissingSecretError(RuntimeError):
    """Raised when a required secret is absent from every configured source."""

    def __init__(self, name: str) -> None:
        """Build the error with an actionable message naming the missing secret."""
        super().__init__(
            f"required secret {name!r} is not set; provide it via an environment "
            f"variable or an untracked .env file (it must never be committed)"
        )
        self.name = name


@runtime_checkable
class SecretsProvider(Protocol):
    """Source of runtime secrets. Implementations must never persist or log values."""

    def get(self, name: str) -> str:
        """Return the secret ``name`` or raise :class:`MissingSecretError`."""
        ...

    def get_optional(self, name: str, default: str | None = None) -> str | None:
        """Return the secret ``name`` if present, otherwise ``default``."""
        ...


def _parse_dotenv(path: Path) -> dict[str, str]:
    """Parse a simple ``KEY=VALUE`` ``.env`` file into a dict.

    Blank lines and ``#`` comments are ignored; a leading ``export`` is stripped;
    matching single or double quotes around a value are removed. Deliberately
    minimal — no interpolation — to avoid surprising behavior around secrets.
    """
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key:
            values[key] = value
    return values


class EnvSecretsProvider:
    """Secrets provider backed by environment variables and an optional ``.env``.

    Real environment variables take precedence over ``.env`` file entries. An
    empty value is treated as absent, so a blank variable does not masquerade as
    a provided secret.
    """

    def __init__(self, values: Mapping[str, str]) -> None:
        """Wrap an already-resolved mapping of secret names to values."""
        self._values: dict[str, str] = dict(values)

    @classmethod
    def from_environment(
        cls,
        *,
        environ: Mapping[str, str] | None = None,
        dotenv_path: Path | None = None,
    ) -> EnvSecretsProvider:
        """Build a provider from the process environment and an optional ``.env``.

        Args:
            environ: Environment mapping to use; defaults to ``os.environ``.
            dotenv_path: Optional path to a ``.env`` file layered *beneath* the
                environment (environment variables win on conflict).
        """
        merged: dict[str, str] = {}
        if dotenv_path is not None and dotenv_path.exists():
            merged.update(_parse_dotenv(dotenv_path))
        merged.update(os.environ if environ is None else environ)
        return cls(merged)

    def get(self, name: str) -> str:
        """Return the secret ``name`` or raise :class:`MissingSecretError`."""
        value = self._values.get(name)
        if not value:
            raise MissingSecretError(name)
        register_secret_value(value)
        return value

    def get_optional(self, name: str, default: str | None = None) -> str | None:
        """Return the secret ``name`` if present and non-empty, else ``default``."""
        value = self._values.get(name)
        if not value:
            return default
        register_secret_value(value)
        return value
