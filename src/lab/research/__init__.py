"""Research layer (Phase 2+): validation harness, strategies, trials, reports.

Built after the data layer. Subpackages: ``validation`` (purged CV, embargo,
CPCV, DSR, PBO, cost backtester, robustness), ``strategies`` (the ``StrategySpec``
Protocol and one thin module per study), ``trials`` (the cumulative honest
trial-count ledger), and ``reports`` (per-study report, tearsheet, kill-gate
emitter, paper updater). The validation engine is written once and reused
unchanged for every strategy (Part III Layer 2).
"""
