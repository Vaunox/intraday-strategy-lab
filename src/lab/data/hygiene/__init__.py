"""Data hygiene jobs (Phase 1), each idempotent, tested, and logged.

Corp-action adjustment (raw + adjusted stored), point-in-time constituents /
survivorship control, bad-tick filtering (logged, never silently mutated), gap
detection, liquidity screening, and ESM/T2T exclusion (Part III Layer 1).
"""
