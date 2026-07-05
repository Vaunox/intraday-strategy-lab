"""Data layer (Phase 1): Kite historical ingestion, storage, hygiene, features.

Built in Phase 1. Subpackages: ``brokers`` (the only place the Kite SDK is
imported), ``ingest`` (backfill), ``store`` (Parquet repository), ``hygiene``
(corp-actions, survivorship, bad-tick, gaps, liquidity), and ``features``
(point-in-time pure feature/indicator functions).
"""
