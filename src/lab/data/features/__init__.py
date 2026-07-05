"""Feature & indicator library (Phase 1): point-in-time pure functions.

All OHLCV-derived and point-in-time: a feature at ``asof`` may only use data
available at or before ``asof``. Exposes ``compute_features(symbol, asof)`` used
identically in backfill and serving, with a vectorized == incremental skew
tripwire and CI leakage tests (Part III Layer 1).
"""
