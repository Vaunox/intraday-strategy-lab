# Deep Dive 02 — Research & Validation Layer (Layer 2)

> **Status: authored outline — expand on demand.** Part III of `MASTER_BLUEPRINT.md` is the self-contained, authoritative distillation of this layer and is sufficient to build Phases 2–4 from. This file exists so the reference resolves and so deeper detail can be added where a task needs it. Part III governs on any divergence; log divergences in `docs/PROGRESS.md`.

## Scope
From a rule-based `StrategySpec` to an honest, cost-inclusive, overfitting-resistant PASS/KILL verdict and a written paper section. **The validation engine is written once and reused unchanged for all 20 strategies.**

## Strategy specification
`StrategySpec` = event → entry → exit/holding → position/weight, producing the per-period position/return series the validation engine consumes. Each strategy is one *thin* spec; strategy code never touches the validation engine.

## Backtester realism (non-negotiable)
- **Next-bar-open fills:** decide on bar *t* close, fill at bar *t+1* open. Never same-bar.
- **Full Indian cost model** (rates in `config/costs.yaml`): brokerage (lower of ~0.03% / ₹20), STT ~0.025% sell-side, exchange txn ~0.003%/side NSE, SEBI turnover ₹10/crore, GST 18% on (brokerage + exchange + SEBI), stamp ~0.003% buy-side. **No DP charge** (MIS intraday, not delivery). ≈ 0.12–0.20% round trip.
- **Slippage:** 0.05–0.20%, size/liquidity-aware; widen in stress.
- **Intraday square-off** at configured session end; no overnight carry.

## Sharpe convention (fixed before Phase 2)
Annualized by √(`sharpe.periods_per_year`); scaled on **in-market periods, not calendar**; identical across every study. A bare "Sharpe" is not comparable across intraday strategies — the convention is pinned so P3.1 and P3.13 are.

## Validation — two questions, two tools
- **Is the edge real?** Purged k-fold + embargo (**1-trading-day embargo** ≥ the intraday holding horizon); CPCV (N groups, k test → C(N,k) splits → φ = C(N,k)·k/N paths; judge the *distribution* of path-Sharpes); Deflated Sharpe Ratio (corrects for the **effective number of independent trials**, skew, kurtosis, length; fed by the program-wide ledger of per-trial return streams, correlated variants clustered); PBO via CSCV.
- **What would live feel like?** Walk-forward with full costs + slippage + next-bar-open fills.

## Meta-labeling (the only ML; optional, bounded, gated — Phase 4.5)
Rule decides the **side**; a small calibrated model (LightGBM/logistic) decides **bet/no-bet + size** from context only (ATR, time-of-day, volume, regime — never lookahead), tuned under purged CV, probabilities calibrated (isotonic/Platt), size fractional-Kelly-capped. Runs **only** on a strategy already showing a *gross* edge worth filtering; every variant charged to the trial ledger. A meta-model that turns a losing rule into a winner is overfitting, not alpha.

## Honest trial ledger (effective, not raw)
Program-wide, machine-maintained store of **every variant's realized return stream** (including discarded), persisted across all sessions and phases. The DSR is deflated by the **effective number of independent trials, not the raw variant count**: cluster the trials by P&L correlation; each cluster contributes an effective count reflecting its internal correlation — a tight cluster of near-duplicate variants (a one-at-a-time parameter sweep) contributes far less than its member count, while genuinely distinct strategies each contribute ~their own weight. This is the same cluster-adjusted effective-trial-count pattern as López de Prado's covariance/clustering treatment. Feeding a raw integer *N* would over-deflate the whole program and kill strategies that held a real edge. The effective-N feeds the DSR automatically; no caller passes a literal N (CI-enforced).

## THE KILL-GATE — every threshold a single pre-committed number (`config/killgate.yaml`)
A range invites picking the lenient end when the strict end fails — that is the overfitting Inviolable Rule 1 forbids. So each criterion is one pinned value, fixed before running:

1. **CPCV median path-Sharpe > 1.0**, net of costs, on the fixed Sharpe convention.
2. **DSR ≥ 0.95** vs the live **effective** trial count (correlated variants clustered; never a raw count).
3. **PBO < 0.20** via CSCV.
4. **≥ 90% of CPCV paths positive** and **10th-percentile path-Sharpe ≥ 0** (both).
5. **Profit factor ≥ 1.3**, **top-5 winning trades < 40% of gross profit**, **expectancy > round-trip cost** (all three).
6. **Survives the robustness battery**, each sub-test with a pinned bar: parameter sensitivity (net Sharpe > 0.5 across ±1 config step of every parameter); Monte-Carlo trade-shuffle (real net Sharpe beats ≥ 95% of shuffles); noise injection (edge survives realistic OHLC perturbation); cross-symbol (net-positive on a majority of held-out symbols); two-engine reconciliation (vectorized vs event-driven within tolerance).
7. **Edge stable across pre-defined regimes:** median path-Sharpe > 0 in **every** bucket (partitioned by year and by vol/trend regime, fixed before running), > 0.5 in a majority, and net-positive with the single best bucket removed.

Fail any one → KILL, recorded honestly in the paper. Changing any threshold invalidates every prior verdict and re-runs the slate.

## Cost-viability pre-check (before the full battery)
If a strategy's median gross move per trade is smaller than modeled round-trip cost + slippage, it is cost-dead before validation — record that as the verdict, log the trial, and don't burn the full battery on it. (Scalping is the expected first casualty.)

## To expand later (as tasks demand)
- Exact DSR and PSR formulas, the benchmark-Sharpe derivation, and the trial-clustering that turns per-trial return streams into the effective trial count.
- CSCV construction for PBO and the logit-of-rank test.
- The two-engine reconciliation tolerance and the hand-computed cost/fill fixtures.
- Robustness sub-test parameterization and seeds.
