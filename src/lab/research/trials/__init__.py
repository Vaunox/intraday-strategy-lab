"""Honest effective-trial ledger (Phase 2, P2.3).

Program-wide, machine-maintained store of every strategy variant's realized
return stream (including discarded ones), persisted across all sessions and
phases. The Deflated Sharpe is deflated by the EFFECTIVE number of independent
trials — correlated variants clustered by P&L correlation, not the raw variant
count, so a one-at-a-time parameter sweep is not mistaken for many independent
bets. No caller ever passes a literal N (Inviolable Rule 4).
"""
