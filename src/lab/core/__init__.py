"""Core domain layer: types, interfaces, and cross-cutting infrastructure.

Everything here is dependency-light and imported by every other layer: domain
types (``types``), the swappable interface Protocols (``interfaces``), the
layered config loader (``config``), the secrets interface (``secrets``),
structured logging (``logging``), and the NSE trading calendar (``nse_calendar``).

Nothing in this package imports a broker SDK, a storage client, or any strategy
code — the dependency arrows point inward, toward ``core``.
"""
