"""Strategy specs (Phase 3+): the ``StrategySpec`` Protocol and one module per study.

Each of the 20 strategies (Part V) is one thin ``StrategySpec`` — event → entry →
exit/holding → position/weight — that never touches the validation engine
(Part III Layer 2).
"""
