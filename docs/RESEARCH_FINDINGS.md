# Research Findings — Classic Intraday Strategies on NSE Equities (Kite Historical)

> **How this document is maintained.** This is a *living research paper*, authored and updated by the engineer/agent building each study — **not pre-filled**. The strategy reference and prior-evidence tables in §4 are background (what each strategy is and what the literature already suggests); the **results** fields (§4 scorecard and §5–§6) are scaffolds. As each strategy study (Phase 3 / Phase 4) completes, its author replaces the `‹placeholder›` fields with the study's **real, cost-inclusive, kill-gate numbers** and its honest verdict. Do not write results that have not been produced by an actual validated run. An honest KILL is a complete result.

*Status: scaffold — reference tables populated; no study results recorded yet.*

---

## Abstract

‹To be written after the studies are complete. One paragraph: what was tested (the 20-strategy slate), on what data (Kite historical NSE equities), under what discipline (purged CV / CPCV / Deflated Sharpe / PBO / full Indian costs / seven-point kill-gate), and the honest bottom line — which strategies (if any) held a real, cost-surviving edge.›

---

## 1. Objective & scope

- **Question:** Does any strategy in a fixed slate of classic intraday techniques hold a small, real, cost-surviving edge on liquid NSE cash equities, using only Kite historical candle data?
- **Slate:** 14 single-factor strategies + 6 multi-factor combinations (see §4).
- **Explicitly not claimed:** "high stable profit." SEBI studies find over 90% of retail F&O traders lose money (≈91% of individual traders in FY24; ~93% over FY22–FY24). *This is a humility anchor on retail active trading; note the SEBI figure is F&O-specific, whereas this program tests **cash-equity intraday** — a related but distinct population, not the same statistic.* The achievable deliverable is an honest per-strategy verdict.
- **Research-only:** no live trading, no capital at risk.

## 2. Data

| Field | Value |
|---|---|
| Source | Zerodha Kite Connect historical candle API (OHLCV[+OI]) |
| Universe | NIFTY 50 — 49 names backfilled. Snapshot & provenance in `config/universe/nifty50.yaml` (`as_of` 2025-09-30). **Survivor-only, NOT point-in-time** — see §2.1. |
| Date range | 2015-02-02 → 2026-07-03 (Kite's 5-min history begins Feb 2015) |
| Intervals | 5-minute (primary decision frequency; ≈ 8.96M candles) |
| Hygiene applied | corp-action adjustment (adjusted layer), bad-tick filter (zero/blank-price bars dropped at ingest, logged), gap detection, liquidity screen, ESM/T2T exclusion. **Survivorship NOT corrected** (see §2.1). |
| Data version | ‹hash / tag› |

**Data constraint:** Kite historical candles only. No live depth, no alternative feeds. Microstructure/order-book strategies (order-flow imbalance, depth) are out of scope. Every strategy in the slate is derivable from OHLCV(+volume) alone — see §4 reference table.

### 2.1 — Known limitation: survivor-only universe (bounded, documented)

The backfill uses **today's** index members over **past** data, so it carries **survivorship bias**. Scope and why it is bounded (not a validity gate):

- **Forward P&L is unaffected.** We trade the then-current index going forward (paper and live), so survivorship cannot touch forward returns — it biases only the *historical backtest*.
- **Direction is upward for long/rank/cross-sectional studies** (survivors outperformed), so a survivor-only backtest is an **upper bound** for those. It is **not guaranteed positive for intraday cross-sectional** signals.
- **Bounded for NIFTY 50 intraday MIS.** Index exits are **demotions, not deaths** — LTIM, IndusInd, Hero MotoCorp, BPCL, Britannia all still trade on NSE; only genuine corporate-action exits (HDFC Ltd merger, pre-demerger Tata Motors) are truly unrecoverable. Intraday square-off (MIS) neutralizes the hold-through-decline channel that drives most equity survivorship bias.
- **As-built delta.** The backfilled 49 wrongly include BPCL, BRITANNIA, HEROMOTOCO, INDUSINDBK and omit ETERNAL, INDIGO, JIOFIN, MAXHEALTH, TMPV vs the current set (they were pulled from a stale ~2024 list); full delta in the artifact's `backfill:` block.
- **Handling.** Study output carries a **provisional / upper-bound stamp only when a strategy clears its gate by a narrow margin** — the sole place a small bias could flip a verdict; wide-margin passes and KILLs are not stamped.
- **Roadmap (not built now):** point-in-time membership from NSE reconstitution notices (a few changes/year) plus pulling demoted names from Kite would close most of the residual, since demoted names still trade.

## 3. Methodology

- **Signal form:** deterministic rule-based `StrategySpec`s (no ML as the signal generator; meta-labeling is the only optional ML, and only on strategies that already show a gross edge — Phase 4.5).
- **Execution model:** next-bar-open fills; intraday square-off; full Indian cost model (brokerage / STT sell-side / exchange txn / SEBI turnover fee / GST / stamp buy-side ≈ 0.12–0.20% round trip; no DP charge — MIS intraday, not delivery) + size/liquidity-aware slippage (0.05–0.20%). All statutory rates read from `config/costs.yaml`.
- **Validation:** purged k-fold + embargo (**1-trading-day embargo**, ≥ the intraday holding horizon); Combinatorial Purged CV (path-Sharpe distribution); Deflated Sharpe Ratio against the **honest, program-wide cumulative EFFECTIVE trial count (correlated variants clustered)**; PBO via CSCV; walk-forward equity; robustness battery (parameter sensitivity, Monte Carlo shuffle, noise injection, cross-symbol); two-engine reconciliation.
- **Sharpe convention (fixed for all studies):** annualized by each strategy's **realized frequency** — `realized_periods_per_year` = return observations ÷ operating-span-years, derived per study, never a fixed calendar constant; scaled on **in-market periods, not calendar time**; identical convention across every study so verdicts are comparable. (Replaced the removed fixed `sharpe.periods_per_year` = 18750 constant, which contradicted the in-market rule; `MASTER_BLUEPRINT.md` §255 amendment / PR #17.)
- **Pre-registration:** each study's hypothesis, parameters, and kill thresholds were committed to git **before** its first test run (auditable in history).
- **The seven-point kill-gate** — every threshold a single pre-committed number in `config/killgate.yaml`, never a range, never adjusted to pass: (1) CPCV median path-Sharpe **> 1.0** net of costs; (2) **DSR ≥ 0.95** vs the effective trial count (correlated variants clustered); (3) **PBO < 0.20**; (4) **≥ 90% of CPCV paths positive and 10th-percentile path-Sharpe ≥ 0**; (5) **profit factor ≥ 1.3, top-5 wins < 40% of gross profit, expectancy > round-trip cost**; (6) survives the robustness battery (each sub-test with a pinned bar); (7) median path-Sharpe > 0 in **every** pre-defined regime bucket, and net-positive with the single best bucket removed. Fail any one → KILL.

**Cumulative effective trial count at time of writing:** ‹machine-maintained effective-N — clustered from the per-trial return streams in the ledger, never a raw variant count, never hand-typed›.

---

## 4. Strategy slate

The slate is fixed before testing. §4.1 defines each strategy and its exact Kite-OHLCV inputs. §4.2 records the prior-evidence view and known failure modes (background, not this study's result). §4.3 is the results scorecard, filled as studies complete.

### 4.1 — Strategy reference table (definition & mechanics)

**Single-factor (Phase 3) — all inputs are Kite historical OHLCV(+volume):**

| ID | Strategy | What it is (mechanic) | Kite OHLCV inputs | Natural timeframe |
|---|---|---|---|---|
| P3.1 | VWAP | Trade price relative to the intraday volume-weighted average price — fade back to VWAP (mean-reversion) or trade the side of a decisive cross (trend). | close, volume → cumulative VWAP, VWAP deviation | 5 / 15-min |
| P3.2 | Breakout / Breakout Filters | Enter when price breaks a prior range/level; filter the break by volume expansion and/or volatility to cut false breaks. | rolling high/low, ATR, volume | 5 / 15-min |
| P3.3 | Mean Reversion | Fade statistically stretched moves back toward a rolling mean or band. | rolling mean, z-score, Bollinger bands | 5 / 15-min |
| P3.4 | Reversal Trading | Trade exhaustion at extremes — swing-failure / divergence turning points against the prior move. | swing highs/lows, RSI, candle geometry | 5 / 15-min |
| P3.5 | Pivot Points | Use pre-computed classic/Fibonacci/Camarilla pivot levels as intraday support/resistance for entries and targets. | prior-session H/L/C → pivot levels | 15-min |
| P3.6 | Donchian Channels | Enter on a break of the N-bar highest-high / lowest-low channel (a rules-clean breakout/trend rule). | rolling N-bar high/low | 15-min |
| P3.7 | Adaptive Moving Averages | KAMA/AMA that speeds up in trends and slows in chop; trade the cross or the slope. | close → KAMA/AMA (efficiency-ratio adaptive) | 5 / 15-min |
| P3.8 | Volatility-Based Filters | Not a standalone entry — an overlay that gates other entries by ATR / realized-vol / range regime. | ATR, realized vol, true range | overlay |
| P3.9 | Momentum Pullback | In an established trend, re-enter on a shallow pullback to support (MA / prior level) rather than chasing. | trend MA, pullback depth, RSI | 5 / 15-min |
| P3.10 | Gap and Go | Trade continuation of a large opening gap when early volume confirms participation. | prior close vs open (gap %), opening volume, VWAP | 1 / 5-min (open) |
| P3.11 | Opening Range Breakout (ORB) | Mark the first-N-minute range; trade the break of its high/low. | first-N-min high/low, volume | 1 / 5-min (open) |
| P3.12 | Bull Flag | Pattern: sharp impulse, tight lower-volume consolidation, then breakout continuation. | impulse leg, consolidation range, volume profile | 5-min |
| P3.13 | Scalping | Many small, fast in/out trades on fine bars capturing micro mean-reversion/momentum — extremely cost-sensitive. | fine-bar micro signals, ATR | 1 / 3-min |
| P3.14 | Moving Average Crossovers | Trade the cross of a fast and slow SMA/EMA. | close → fast/slow SMA or EMA | 5 / 15-min |

**Multi-factor combinations (Phase 4) — composed from the same Phase-1 indicators:**

| ID | Combination | What it is (confluence logic) | Kite OHLCV inputs | Natural timeframe |
|---|---|---|---|---|
| P4.1 | VWAP + Breakout + Volume Surge | Take a range break only when it happens on the trend side of VWAP *and* on a relative-volume surge — three-way confluence. | VWAP, rolling high/low, relative volume | 5 / 15-min |
| P4.2 | ORB + VWAP Confirmation | Take the opening-range break only if price is on the confirming side of VWAP. | first-N-min range, VWAP | 1 / 5-min (open) |
| P4.3 | Bollinger MR + RSI Extreme | Fade a Bollinger-band touch only when RSI is simultaneously at an extreme (momentum exhaustion). | Bollinger %B, RSI | 5 / 15-min |
| P4.4 | Adaptive MA + ADX | Take the KAMA/AMA trend signal only when ADX confirms sufficient trend strength (gate out chop). | KAMA/AMA, ADX/DMI | 5 / 15-min |
| P4.5 | Pivot Confluence + MACD | Act at a pivot level only when a MACD crossover gives a momentum trigger. | pivot levels, MACD | 15-min |
| P4.6 | Donchian Breakout + ATR Stop | Donchian channel breakout with a volatility-scaled (ATR-multiple) stop-loss for risk control. | Donchian high/low, ATR | 15-min |

### 4.2 — Prior evidence & known failure modes (background, *not* this study's result)

> This column summarizes the general quantitative-trading literature and widely-reported backtests as **priors to be tested**, not as verdicts. The consistent theme across sources: many of these rules show *some* gross signal in-sample, but few survive realistic costs, multiple-testing correction, and out-of-sample regimes — which is exactly why every strategy is put through the same kill-gate here rather than trusted. Verdicts come only from §4.3 / §5–§6.

| ID | Strategy | Economic rationale (why it *might* work) | Known failure modes | Prior-evidence view (honest, to be tested) |
|---|---|---|---|---|
| P3.1 | VWAP | Institutions benchmark fills to VWAP, creating real intraday pull toward it. | Trends far from VWAP whipsaw mean-reversion; the "cross" version lags. | Widely used as an execution benchmark and context filter; as a *standalone* directional edge the evidence is weak and highly cost-sensitive. |
| P3.2 | Breakout / Filters | Ranges resolve; genuine breaks release pent-up order flow. | High false-break rate; whipsaw in chop; slippage worst exactly at the break. | Mixed — unfiltered breakouts historically decay; volume/volatility filters help but rarely enough to clear costs intraday. |
| P3.3 | Mean Reversion | Short-horizon overreaction reverts; liquidity provision earns the bounce. | Catastrophic in trends/news; fat left tail; "fade" fights momentum. | The most persistently documented short-horizon effect, but the edge is small and easily eaten by intraday costs. |
| P3.4 | Reversal Trading | Exhaustion and stop-runs create sharp snapbacks at extremes. | Picking tops/bottoms; low base rate; discretionary pattern definition. | Weak and hard to systematize; reversal signals are noisy and prone to overfitting. |
| P3.5 | Pivot Points | Widely-watched levels can become self-fulfilling S/R. | Arbitrary among many level systems; edge likely already arbitraged. | Popular but thin empirical support; performs no better than random levels in most published tests. |
| P3.6 | Donchian Channels | Classic trend-following capture of sustained moves. | Trend-following struggles intraday (few sustained intraday trends); many small losses. | Strong pedigree on *daily/higher* timeframes (Turtles); intraday evidence is much weaker and cost-sensitive. |
| P3.7 | Adaptive Moving Averages | Adapts smoothing to noise, cutting whipsaw vs fixed MAs. | Still a lagging trend tool; adaptivity adds parameters to overfit. | Marginal improvement over fixed MAs in studies; not a reliable standalone edge. |
| P3.8 | Volatility-Based Filters | Vol regime conditions when other signals work (overlay, not entry). | Can over-filter away all trades; regime lag. | Consistently useful as a *conditioning* overlay; not an edge by itself — evaluated as a gate on the others. |
| P3.9 | Momentum Pullback | Buys strength at a discount; aligns with trend, better entry. | Defining "pullback" is discretionary; trend can end at the pullback. | Momentum is among the most robust cross-sectional effects; the *intraday pullback* execution is far less established. |
| P3.10 | Gap and Go | Overnight information gaps + confirming volume can continue. | Gaps also fade ("gap fill"); news-driven and hard to model; open is high-slippage. | Genuinely mixed — continuation vs fade is regime-dependent; volume confirmation is the crux and is fragile. |
| P3.11 | Opening Range Breakout (ORB) | Opening auction sets a range; the break signals the day's direction. | Heavily popularized → likely crowded/decayed; sensitive to the range window. | Popular and occasionally survives costs in careful single-market studies, but broad backtests report it "doesn't work very well anymore" — a prime overfitting risk. |
| P3.12 | Bull Flag | Consolidation after impulse = continuation pattern with defined risk. | Pattern definition is subjective; hard to encode without lookahead. | Chartist staple with little rigorous out-of-sample support; encoding risk is high. |
| P3.13 | Scalping | Tiny frequent edges from micro-structure noise. | Costs dominate; needs execution quality this data cannot model; slippage brutal. | Least likely to survive on historical candles + realistic costs; included mainly to demonstrate the cost wall honestly. |
| P3.14 | Moving Average Crossovers | Simplest trend capture; catches sustained moves. | Whipsaw in range-bound intraday; heavy lag; many crossings. | The canonical "looks great in a trend, loses in chop" rule; broad evidence of decay after costs. |
| P4.1 | VWAP + Breakout + Volume Surge | Three independent confirmations should cut false breaks. | Confluence also cuts sample size; multiplies parameters to overfit. | Confluence *can* raise precision, but each added condition inflates the search space — DSR deflation matters most here. |
| P4.2 | ORB + VWAP Confirmation | VWAP side filters low-quality opening breaks. | Inherits ORB's crowding + fewer trades. | Plausible precision gain over raw ORB; still fighting ORB's decay and costs. |
| P4.3 | Bollinger MR + RSI Extreme | Two exhaustion signals agreeing should improve fades. | Both fail together in strong trends; double-fitting extremes. | A common textbook combo; agreement helps in-sample but rarely survives honest OOS + costs. |
| P4.4 | Adaptive MA + ADX | Trade the trend rule only when trend strength is real. | ADX lags; gating shrinks sample; two tools, more params. | Trend-strength gating is one of the more defensible filters, but the net intraday edge is unproven. |
| P4.5 | Pivot Confluence + MACD | Level + momentum trigger reduces acting on dead levels. | Inherits pivots' weak base; MACD lag. | Weak priors on both legs; combination unlikely to manufacture a real edge. |
| P4.6 | Donchian Breakout + ATR Stop | ATR-scaled risk control on a clean breakout rule. | The ATR stop improves risk, not the entry's weak intraday base. | Sound risk management on a weak intraday entry — expect better drawdowns, not necessarily a passing edge. |

### 4.3 — Results scorecard (filled as studies complete)

**Single-factor (Phase 3)**
| ID | Strategy | Category | Verdict | CPCV median Sharpe (net) | DSR | PBO | Section |
|---|---|---|---|---|---|---|---|
| P3.1 | VWAP | Trend/MR | ‹—› | ‹—› | ‹—› | ‹—› | §5.1 |
| P3.2 | Breakout / Filters | Breakout | ‹—› | ‹—› | ‹—› | ‹—› | §5.2 |
| P3.3 | Mean Reversion | Mean-reversion | ‹—› | ‹—› | ‹—› | ‹—› | §5.3 |
| P3.4 | Reversal Trading | Reversal | ‹—› | ‹—› | ‹—› | ‹—› | §5.4 |
| P3.5 | Pivot Points | S/R | ‹—› | ‹—› | ‹—› | ‹—› | §5.5 |
| P3.6 | Donchian Channels | Breakout | ‹—› | ‹—› | ‹—› | ‹—› | §5.6 |
| P3.7 | Adaptive Moving Averages | Trend | ‹—› | ‹—› | ‹—› | ‹—› | §5.7 |
| P3.8 | Volatility-Based Filters | Overlay | ‹—› | ‹—› | ‹—› | ‹—› | §5.8 |
| P3.9 | Momentum Pullback | Momentum | ‹—› | ‹—› | ‹—› | ‹—› | §5.9 |
| P3.10 | Gap and Go | Momentum | ‹—› | ‹—› | ‹—› | ‹—› | §5.10 |
| P3.11 | Opening Range Breakout | Breakout | ‹—› | ‹—› | ‹—› | ‹—› | §5.11 |
| P3.12 | Bull Flag | Pattern | ‹—› | ‹—› | ‹—› | ‹—› | §5.12 |
| P3.13 | Scalping | Fast | ‹—› | ‹—› | ‹—› | ‹—› | §5.13 |
| P3.14 | Moving Average Crossovers | Trend | ‹—› | ‹—› | ‹—› | ‹—› | §5.14 |

**Multi-factor combinations (Phase 4)**
| ID | Combination | Verdict | CPCV median Sharpe (net) | DSR | PBO | Section |
|---|---|---|---|---|---|---|
| P4.1 | VWAP + Breakout + Volume Surge | ‹—› | ‹—› | ‹—› | ‹—› | §6.1 |
| P4.2 | ORB + VWAP Confirmation | ‹—› | ‹—› | ‹—› | ‹—› | §6.2 |
| P4.3 | Bollinger MR + RSI Extreme | ‹—› | ‹—› | ‹—› | ‹—› | §6.3 |
| P4.4 | Adaptive MA + ADX | ‹—› | ‹—› | ‹—› | ‹—› | §6.4 |
| P4.5 | Pivot Confluence + MACD | ‹—› | ‹—› | ‹—› | ‹—› | §6.5 |
| P4.6 | Donchian Breakout + ATR Stop | ‹—› | ‹—› | ‹—› | ‹—› | §6.6 |

---

## 5. Single-factor study results

*Each subsection is filled by the study's author when the study completes. Template below — copy per strategy.*

### 5.1 — P3.1 · VWAP
- **Hypothesis (pre-registered):** ‹economic rationale — why this should have an edge›
- **Pre-registration commit:** ‹git SHA, dated before first test run›
- **Spec summary:** ‹entry / exit / holding / frequency / key params›
- **Variants tried (all charged to the trial ledger):** ‹list›
- **Results (cost-inclusive):** CPCV median path-Sharpe ‹—›; path distribution ‹—›; DSR ‹—›; PBO ‹—›; profit factor ‹—›; regime stability ‹—›; robustness ‹pass/fail per test›.
- **Seven-point kill-gate:** (1) ‹—› (2) ‹—› (3) ‹—› (4) ‹—› (5) ‹—› (6) ‹—› (7) ‹—›
- **Verdict:** ‹KILL / PASS›
- **Notes:** ‹what the numbers say; where the edge (if any) lives or dies; cost sensitivity›

### 5.2 — P3.2 · Breakout / Breakout Filters
‹same template›

### 5.3 — P3.3 · Mean Reversion
‹same template›

### 5.4 — P3.4 · Reversal Trading
‹same template›

### 5.5 — P3.5 · Pivot Points
‹same template›

### 5.6 — P3.6 · Donchian Channels
‹same template›

### 5.7 — P3.7 · Adaptive Moving Averages
‹same template›

### 5.8 — P3.8 · Volatility-Based Filters
‹same template›

### 5.9 — P3.9 · Momentum Pullback
‹same template›

### 5.10 — P3.10 · Gap and Go
‹same template›

### 5.11 — P3.11 · Opening Range Breakout (ORB)
‹same template›

### 5.12 — P3.12 · Bull Flag
‹same template›

### 5.13 — P3.13 · Scalping
‹same template›

### 5.14 — P3.14 · Moving Average Crossovers
‹same template›

---

## 6. Multi-factor combination results

### 6.1 — P4.1 · VWAP + Breakout Filter + Volume Surge
‹same template as §5.1›

### 6.2 — P4.2 · ORB + VWAP Confirmation
‹same template›

### 6.3 — P4.3 · Bollinger Band Mean Reversion + RSI Extreme
‹same template›

### 6.4 — P4.4 · Adaptive Moving Average + ADX
‹same template›

### 6.5 — P4.5 · Pivot Point Confluence + MACD Crossover
‹same template›

### 6.6 — P4.6 · Donchian Channel Breakout + ATR Stop-Loss
‹same template›

---

## 6b. Optional meta-labeling results (Phase 4.5)

*Filled only if a Phase-3/4 strategy showed a gross edge and qualified for meta-labeling. If nothing qualified, state that plainly here — it is a valid outcome.*

**Qualifying strategies:** ‹list, or "none qualified — Phase 4.5 skipped"›

*Per qualifying strategy — copy the template:*

### 6b.1 — ‹strategy ID› + meta-label
- **Why it qualified:** ‹gross expectancy / gross CPCV path-Sharpe before the meta-model›
- **Meta-model:** ‹classifier, context features (no lookahead), calibration method›
- **Variants tried (all charged to the trial ledger):** ‹list›
- **Net result after meta-labeling (cost-inclusive):** CPCV median path-Sharpe ‹—›; DSR ‹—›; PBO ‹—›; **delta vs un-filtered rule** ‹—›.
- **Seven-point kill-gate on the meta-labeled strategy:** ‹per-criterion›
- **Verdict:** ‹improved & PASS / improved but still KILL / no improvement — meta-model discarded›
- **Notes:** ‹did filtering the rule's instances add real net value, or just overfit context?›

---

## 7. Cross-strategy synthesis

‹Filled at Phase 5. Master results table ranking all 20 studies by DSR-adjusted, cost-inclusive path-Sharpe. Which cleared the seven-point kill-gate (if any). What distinguishes survivors from failures — category, frequency, cost sensitivity, regime dependence. How much the honest cumulative effective trial count (correlated variants clustered) deflated the raw Sharpes.›

## 8. Conclusion

‹The honest bottom line. Which strategies (if any) hold a real, cost-surviving edge, and which are bias/luck artifacts. If none cleared the gate, state it plainly — a complete and valuable result. Record the total cumulative effective trial count (correlated variants clustered) and its DSR implications. Note next steps only if a survivor exists (productionization is gated on a passing verdict).›

## 9. Reproducibility appendix

- **Data version / tag:** ‹—›
- **Universe file:** ‹config/universe.yaml @ SHA›
- **Config:** ‹config used for the runs›
- **Commands:** ‹exact `scripts/run_study.py ...` invocation per study›
- **Environment:** ‹python + key package versions›

---

*Not financial advice. Research-only; no capital risked. Over 90% of retail F&O traders lose money; no profit is promised. The prior-evidence column in §4.2 is background context to be tested by this program, not a claim about returns.*
