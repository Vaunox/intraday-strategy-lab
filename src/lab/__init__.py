"""Intraday Strategy Research Lab.

An honest, cost-inclusive research apparatus for testing a fixed slate of classic
intraday trading strategies on liquid NSE cash equities, using Zerodha Kite
Connect historical candle data as the only data source.

The build order is deliberate: the research apparatus (foundation, data/features,
validation harness — Phases 0-2) is built and gated before any strategy is tested
(Phases 3-4). See ``MASTER_BLUEPRINT.md`` for the full specification.
"""
