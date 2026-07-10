# Research Findings — Classic Intraday Strategies on NSE Equities (Kite Historical)

> **How this document is maintained.** This is a *living research paper*, authored and updated by the engineer/agent building each study — **not pre-filled**. The strategy reference and prior-evidence tables in §4 are background (what each strategy is and what the literature already suggests); the **results** fields (§4 scorecard and §5–§6) are scaffolds. As each strategy study (Phase 3 / Phase 4) completes, its author replaces the `‹placeholder›` fields with the study's **real, cost-inclusive, kill-gate numbers** and its honest verdict. Do not write results that have not been produced by an actual validated run. An honest KILL is a complete result.

*Status: **COMPLETE** — Phase 5 findings synthesis (2026-07-10). All 25 specs recorded (§5 / §6); cross-strategy synthesis (§7) + conclusion (§8) written. §4.3 scorecard populated per-spec (25 rows), every row stamped exploration-grade / provisional upper-bound (frozen-49 survivor-only panel; operator ruling A). Phase 4.5 meta-labeling not conducted (§6b, out of scope).*

---

## Abstract

This program asked whether any strategy in a fixed slate of classic intraday techniques holds a small, real, cost-surviving edge on liquid NSE cash equities using only Kite historical 5-minute OHLCV. **25 strategy specs** were tested — **19 single-factor** (across the 14 blueprint study-categories P3.1–P3.14, counting owed directional variants such as VWAP cross/reversion) + **6 multi-factor combinations** (P4.1–P4.6). Each was **pre-registered before its run** (parameters and kill thresholds committed to git first) and scored through an **identical seven-point kill-gate** — CPCV median path-Sharpe > 1.0, Deflated Sharpe ≥ 0.95 against the program-wide effective trial count, PBO < 0.20, plus distribution / expectancy / robustness / regime tests — **net of the full Indian round-trip cost**, on a frozen 10-name ADV-tiered panel (per-symbol-then-aggregate, with a 5-name held-out breadth check). **The result is a clean null: every spec was killed or recorded cost-dead; zero cleared the gate.** 15 single-factor specs KILLed on the panel battery, 4 were recorded §6-cost-dead (median per-trade gross below the ~0.18% round-trip cost, so the battery was skipped), and all 6 composites KILLed. The Deflated Sharpe was **0.000 for every scored spec**; the honest cumulative trial count was **99 raw → 9.06 effective** (correlated variants clustered). Precisely scoped (§8): on this **frozen large-cap survivor-only panel at 5-minute resolution**, these retail-accessible classic methods — single-factor and in confluence — **do not survive realistic costs**. This is *not* a claim that no intraday edge exists anywhere; it is the honest, pre-registered verdict for these methods on this universe at this resolution.

---

## 1. Objective & scope

- **Question:** Does any strategy in a fixed slate of classic intraday techniques hold a small, real, cost-surviving edge on liquid NSE cash equities, using only Kite historical candle data?
- **Slate:** 14 single-factor study-categories + 6 multi-factor combinations (see §4) — run as **19 single-factor specs + 6 combination specs = 25 total** (owed directional variants, e.g. VWAP cross/reversion and gap-and-go/gap-fade, are separate specs).
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
| Data version | frozen-49 survivor-only backfill (immutable raw Parquet layer); provenance & as-built delta in `config/universe/nifty50.yaml` + `docs/DATA_MANIFEST.md` (no single tag — see §9) |

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

**Cumulative effective trial count (program complete):** **99 raw trials → 9.06 effective** (machine-maintained, clustered from the per-trial return streams in the ledger via the correlation participation ratio — never a raw variant count). Note the DSR deflation was moot for verdicts here: every scored spec had a *negative* CPCV Sharpe, so DSR = 0.000 regardless of the trial count.

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

> **⚠ EXPLORATION-GRADE / PROVISIONAL UPPER-BOUND — every row (operator ruling A, populated 2026-07-10).** Scored on the **frozen-49 survivor-only** backfill (49 names = **45 current NIFTY-50 members + 4 demoted**; the 5 current members ETERNAL / INDIGO / JIOFIN / MAXHEALTH / TMPV are **absent** — §2.1), a **10-name ADV-tiered scored panel** (+ 5 held-out reserved, **never scored**), **5-minute OHLCV**. **No row is verdict-grade; zero cleared the seven-point kill-gate.** "CPCV median Sharpe (net)" is the aggregate equal-weight-panel path-Sharpe, net of the full round-trip cost; **DSR = 0.000 wherever the Sharpe is negative** (the deflation had no positive to shrink). **§6-cost-dead** rows had median per-trade gross below the ~0.182% round-trip cost, so the full battery was skipped (no panel Sharpe / DSR / PBO — see §5). Listed **per spec (25 rows: 19 single-factor + 6 composite)**; breadth, PF and diagnosis are in §5 / §6.

**Single-factor (Phase 3)**
| ID | Spec | Category | Verdict | CPCV median Sharpe (net) | DSR | PBO | § |
|---|---|---|---|---|---|---|---|
| P3.1-V1 | vwap_mean_reversion | fade | **KILL** | −18.793 | 0.000 | 0.000 | §5.1 |
| P3.1-V2 | vwap_cross | trend | **KILL** | −11.121 | 0.000 | 0.000 | §5.1 |
| P3.2 | breakout | breakout | **KILL** | −8.753 | 0.000 | 0.000 | §5.2 |
| P3.3 | mean_reversion | fade | **KILL** | −27.040 | 0.000 | 0.000 | §5.3 |
| P3.4 | reversal | reversal | **KILL** | −9.910 | 0.000 | 0.000 | §5.4 |
| P3.5 | pivot_reversion | S/R fade | **KILL** | −9.848 | 0.000 | 0.000 | §5.5 |
| P3.6 | donchian_breakout | breakout/trend | **KILL** | −6.132 | 0.000 | 0.000 | §5.6 |
| P3.7-V1 | adaptive_ma_cross | trend | **KILL** | −12.948 | 0.000 | 0.000 | §5.7 |
| P3.7-V2 | adaptive_ma_slope | trend | **COST-DEAD (§6)** | n/a — battery skipped (gross 0.122%) | n/a | n/a | §5.7 |
| P3.8-C1 | vol_expansion_breakout | vol-gated | **KILL** | −7.360 | 0.000 | 0.000 | §5.8 |
| P3.8-C2 | vol_contraction_reversion | vol-gated fade | **KILL** | −25.636 | 0.000 | 0.000 | §5.8 |
| P3.9 | momentum_pullback | momentum | **COST-DEAD (§6)** | n/a — battery skipped (degenerate; gross ~0.13%) | n/a | n/a | §5.9 |
| P3.10 | gap_and_go | gap continuation | **KILL** | −0.743 | 0.000 | 0.000 | §5.10 |
| P3.10b | gap_fade | gap fade | **KILL** | −0.578 | 0.000 | 0.043 | §5.10 |
| P3.11 | opening_range_breakout | breakout | **KILL** | −5.395 | 0.000 | 0.143 | §5.11 |
| P3.12 | bull_flag | pattern | **KILL** | −0.651 | 0.000 | **0.400** | §5.12 |
| P3.13-MR | scalp_mean_reversion | fast fade | **COST-DEAD (§6)** | n/a — battery skipped (gross 0.105%) | n/a | n/a | §5.13 |
| P3.13-mom | scalp_momentum | fast momentum | **COST-DEAD (§6)** | n/a — battery skipped (gross 0.105%) | n/a | n/a | §5.13 |
| P3.14 | ma_crossover | trend | **KILL** | −11.745 | 0.000 | 0.000 | §5.14 |

**Multi-factor combinations (Phase 4)**
| ID | Combination (spec) | Verdict | CPCV median Sharpe (net) | DSR | PBO | § |
|---|---|---|---|---|---|---|
| P4.1 | VWAP + Breakout + Volume (`vwap_breakout_volume`) | **KILL** | −8.202 | 0.000 | 0.000 | §6.1 |
| P4.2 | ORB + VWAP (`orb_vwap`) — ≡ P3.11 ORB | **KILL** | −5.395 | 0.000 | 0.143 | §6.2 |
| P4.3 | Bollinger MR + RSI (`bollinger_rsi`) | **KILL** | −13.126 | 0.000 | 0.000 | §6.3 |
| P4.4 | Adaptive MA + ADX (`adaptive_ma_adx`) | **KILL** | −39.807 | 0.000 | 0.000 | §6.4 |
| P4.5 | Pivot + MACD (`pivot_macd`) | **KILL** | −6.078 | 0.000 | 0.000 | §6.5 |
| P4.6 | Donchian + ATR Stop (`donchian_atr_stop`) | **KILL** | −9.215 | 0.000 | 0.000 | §6.6 |

---

## 5. Single-factor study results

*Each result below is transcribed from the study's committed pre-registration Result block (exploration-grade, panel scope). All are net of the full round-trip cost; DSR is 0.000 for every scored spec (negative CPCV Sharpe); "breadth" = fraction of the 10 panel symbols individually positive. Numbers are provisional / upper-bound (frozen-49 survivor-only; see §2.1). Owed directional variants are listed under their parent study.*

### 5.1 — P3.1 · VWAP
- **Hypothesis:** institutions benchmark fills to VWAP, creating intraday pull toward it (fade) — or a decisive cross of VWAP signals the trend side. **Both-owed directional dichotomy** → two specs.
- **Specs (5-min):** V1 `vwap_mean_reversion` (fade `|VWAP-deviation| > entry` back toward VWAP); V2 `vwap_cross` (trade the side of a `cross_threshold`-cross). Blind params; V1 run first as the corrected-pipeline rebuild regression check (reproduced the pilot).
- **Verdict: both KILL.** V1 aggregate CPCV path-Sharpe **−18.793** (DSR 0.000, PBO 0.000, PF 0.01, 0/10 breadth); V2 **−11.121** (DSR 0.000, PBO 0.000, PF 0.17, 0/10). Cumulative effective-N 1.37.
- **Notes:** opposite bets both dying is coherent — no standalone VWAP directional edge on large-caps; the fade fights momentum, the cross lags.

### 5.2 — P3.2 · Breakout / Breakout Filters
- **Hypothesis:** ranges resolve; a genuine break on expanding volume releases pent-up order flow and continues.
- **Spec (5-min):** `breakout` — intraday-reset 20-bar range break filtered by relative volume > 1.5×; ride to square-off. Volume filter the primary; volatility-entry filter deferred as a ledger variant.
- **Verdict: KILL.** aggregate **−8.753** (DSR 0.000, PBO 0.000, PF 0.23, 0/10 breadth). Effective-N 1.81.
- **Notes:** the volume filter did not rescue the intraday breakout; the false-break rate + cost drag dominate.

### 5.3 — P3.3 · Mean Reversion
- **Hypothesis:** short-horizon overreaction reverts toward a rolling mean.
- **Spec (5-min):** `mean_reversion` — fade `|z| > 2` back to within 0.5 of an **intraday-reset** rolling mean (new `intraday_zscore`, gap-blind by design).
- **Verdict: KILL — the most decisive of the first four.** aggregate **−27.040** (DSR 0.000, PBO 0.000, PF 0.00, 0/10 breadth). Effective-N 2.18.
- **Notes:** fading intraday moves on large-caps loses hard — the fat left tail (fading real moves) dominates. Reinforces "no intraday MR edge here."

### 5.4 — P3.4 · Reversal Trading
- **Hypothesis:** a failed breakout traps traders → a sharp snapback (swing-failure).
- **Spec (5-min):** `reversal` — fade a failed breakout of the **causal** intraday-reset prior swing high/low (`intraday_donchian`): poke by ≥ `break_buffer`, close back inside, fade. No-lookahead prefix-invariance = hard precondition, **PASSED**.
- **Verdict: KILL.** aggregate **−9.910** (DSR 0.000, PBO 0.000, PF 0.15, 0/10 breadth). Effective-N 2.77 — the largest independent-weight jump (most distinct mechanism).
- **Notes:** cost was not the killer (PF 0.15, purely directional) — failed breaks don't reverse on large-caps.

### 5.5 — P3.5 · Pivot Points
- **Hypothesis:** widely-watched classic pivots become self-fulfilling S/R.
- **Spec (5-min):** `pivot_reversion` — fade at classic daily R1/S1 from the **causal** prior session (`classic_pivot_levels`), target the central pivot. No-lookahead precondition (incl. no same-day leak) **PASSED**.
- **Verdict: KILL.** aggregate **−9.848** (DSR 0.000, PBO 0.000, PF 0.19, 0/10 breadth). Effective-N 3.61.
- **Notes:** myth-check — widely-watched pivots do not self-fulfill on large-caps.

### 5.6 — P3.6 · Donchian Channels
- **Hypothesis:** classic trend-following capture of a channel break.
- **Spec (5-min):** `donchian_breakout` — break of the prior **55-bar GLOBAL** Donchian channel (`prior_donchian`, excludes current bar, crosses the gap = multi-session level). No-lookahead precondition **PASSED**.
- **Verdict: KILL — the least-bad single-factor trend/continuation bet.** aggregate **−6.132** (DSR 0.000, PBO 0.000, PF 0.33, 0/10 breadth). Effective-N 4.07.
- **Notes:** the least-dead continuation bet — a multi-session channel is marginally more informative than the intraday range break (−8.75), but still no edge. (Anchors the turnover-gradient observation, §7.)

### 5.7 — P3.7 · Adaptive Moving Averages
- **Hypothesis:** KAMA speeds up in trends / slows in chop; trade the cross or the slope. **Both-owed dichotomy** → two specs.
- **Specs (5-min):** V1 `adaptive_ma_cross` (fast/slow KAMA 10/30 cross); V2 `adaptive_ma_slope` (single-KAMA slope). **Correction caught pre-run:** V1-as-price-vs-single-KAMA was mathematically identical to V2 slope; the divergence precondition test caught it, V1 was redefined as the fast/slow cross (34.5% divergence then proven on real RELIANCE). Prefix-invariance precondition **PASSED**.
- **Verdict: V1 KILL, V2 COST-DEAD.** V1 aggregate **−12.948** (DSR 0.000, PBO 0.000, PF 0.11, 0/10 breadth) — worse than the pure breakouts. V2 recorded **§6-cost-dead** (30,518 trades, median gross 0.122% < 0.182% round-trip; battery skipped per the pre-registered asymmetry — no panel stream). Effective-N 4.34.
- **Notes:** the fast adaptive slope dies on *turnover* (not direction); the cross dies on direction. Sharpens the turnover gradient (§7).

### 5.8 — P3.8 · Volatility-Based Filters
- **Hypothesis:** a vol filter has no standalone edge, so it is tested as **two INDEPENDENT, blind, regime-conditional strategies** (regime + signal defined whole), not an overlay.
- **Specs (5-min):** shared **causal ATR-ratio** regime (`atr(short)/atr(long)`). C1 `vol_expansion_breakout` (intraday breakout gated to expanding vol); C2 `vol_contraction_reversion` (z-fade gated to contracting vol). `atr_ratio` + per-spec no-lookahead preconditions **PASSED**.
- **Verdict: both KILL.** C1 aggregate **−7.360** (DSR 0.000, PBO 0.000, PF 0.27, 0/10) — the vol-gate *helped* (ungated breakout was −8.75). C2 **−25.636** (DSR 0.000, PBO 0.000, PF 0.00); C2 was §6-borderline-cost-dead on RELIANCE (0.174% vs 0.182%) but the §6-skip was **overridden** → full panel → decisively dead (the single-symbol read badly understated the panel; override vindicated). Effective-N 6.12.
- **Notes:** vol-gating helped *continuation* more than *reversion*, and rescued neither.

### 5.9 — P3.9 · Momentum Pullback
- **Hypothesis:** in a trend, enter on the resumption after a shallow in-trend pullback (better entry than chasing).
- **Spec (5-min):** `momentum_pullback` — `close` vs `SMA(50)` trend + `RSI(14)` crossing back up through 30 in-trend (same-day-cross guard). No-lookahead precondition **PASSED**.
- **Verdict: COST-DEAD via §6 (compound).** The blind base (`rsi_pullback=30`) is **near-degenerate on real data** — 5 trades in 11 years; only **0.9%** of RSI-30 recoveries fall in an uptrend (a dip deep enough for RSI(14)<30 almost always also breaks `close`<SMA50 — deep-oversold RSI and the trend filter are near-mutually-exclusive). A pre-run diagnostic across tradeable depths (35/40/45 → 76/586/3,731 trades, **non-blind, NOT scored**) confirmed cost-dead everywhere (~0.13% < 0.182%). Battery skipped; **no panel stream** (ledger unchanged). Recorded cost-dead per operator ruling.
- **Notes:** distinct from P3.7-V2 (which died on turnover) — P3.9 is degenerate at the canonical depth and dies on small per-trade gross at every tradeable depth.

### 5.10 — P3.10 · Gap and Go
- **Hypothesis:** a large opening gap confirmed by early participation continues intraday (gap-and-go); the **gap-fill fade** is the owed directional twin (P3.10b — a directional opposite must not be cherry-picked, so it was promoted to both-owed).
- **Specs (5-min):** `gap_and_go` (gap ≥ 1% + relative-volume surge + `close` on the gap side of VWAP → ride); `gap_fade` (P3.10b — same qualifying gap, trigger flipped to **VWAP rejection** → fade toward the fill). **Divergence from the twin PROVEN** on RELIANCE (226 vs 189 entries, **0% shared bars** — reject vs hold are disjoint). No-lookahead **PASSED**.
- **Verdict: both KILL.** gap_and_go aggregate **−0.743** (DSR 0.000, PBO 0.000, PF 0.75, breadth **0.20**); gap_fade **−0.578** (DSR 0.000, PBO 0.043, PF 0.87, breadth **0.30**). Effective-N 8.69.
- **Notes:** the **two gap studies are the two least-bad of the whole program** (PF 0.75 / 0.87) — they trade sparsely on big-move events. Both gap directions dying is coherent (cf. P3.1). Held conditional in §7 (sparse-big-move observation), not "getting warmer."

### 5.11 — P3.11 · Opening Range Breakout (ORB)
- **Hypothesis:** the first-N-minute range is the day's initial balance; a break signals the day's direction.
- **Spec (5-min):** `opening_range_breakout` — break of the first-30-min range after the window closes; ride to square-off. No-lookahead precondition **PASSED**.
- **Verdict: KILL.** aggregate **−5.395** (DSR 0.000, **PBO 0.143** — elevated, the highest of the single-factor batch, but within the < 0.20 bar; PF 0.37, 0/10 breadth). Effective-N 7.74.
- **Notes:** the popularized ORB shows no edge on large-caps — consistent with the a-priori "crowded / decayed" prior. (The P4.2 composite `orb_vwap` reproduced this result *exactly* — §6.2.)

### 5.12 — P3.12 · Bull Flag
- **Hypothesis:** a sharp impulse + a tight low-volume consolidation + a breakout = continuation.
- **Spec (5-min):** `bull_flag` — impulse (return over K bars) + tight consolidation (`prior_donchian` range ≤ `tight_frac` × impulse) + breakout, with a same-day guard. **Largest parameter surface in the program (4 knobs);** blind base run as committed, **not tuned**. No-lookahead **PASSED**.
- **Verdict: KILL.** aggregate **−0.651** (DSR 0.000, **PBO 0.400 — the ONLY criterion-3 (PBO) failure in the entire program**, the gate explicitly pricing the 4-knob overfitting surface; PF 0.87, breadth **0.30**). 523 active days (sparse) but graded cleanly — **not** INSUFFICIENT.
- **Notes:** scrutinized as the highest-suspicion study (largest surface): not near-threshold, decisively net-negative. Lands in the least-bad cluster with the gaps (net expectancy −0.007%, closest-to-breakeven of any spec) — but the high PBO is the surface *being caught*, not an edge.

### 5.13 — P3.13 · Scalping
- **Hypothesis:** many small fast trades capture micro mean-reversion or micro momentum. **Both-owed dichotomy** → two specs (exact opposites).
- **Specs (5-min):** `scalp_mean_reversion` (fade the last bar's move) + `scalp_momentum` (chase it), per-bar same-day last-bar return. No-lookahead **PASSED**; divergence definitional (opposite sides on every triggered bar).
- **Verdict: both COST-DEAD via §6.** 20,804 trades / 11 yr (7.39/day), median gross **0.105% < 0.182%** round-trip for both. Battery skipped; **no panel stream**.
- **Notes:** the cost wall, demonstrated honestly. **Frequency caveat:** scalping's natural 1–3 min bars are unavailable (archive is 5-min only), so this is a coarser-than-natural proxy; a 1–3 min re-test is deferred to a future finer-data study.

### 5.14 — P3.14 · Moving Average Crossovers
- **Hypothesis:** a fast MA crossing a slow MA marks a trend change; hold the trend side.
- **Spec (5-min):** `ma_crossover` — position = sign of `SMA(20) − SMA(50)`; plain, non-adaptive (distinct from P3.7 KAMA). No-lookahead precondition **PASSED**.
- **Verdict: KILL.** aggregate **−11.745** (DSR 0.000, PBO 0.000, PF 0.15, 0/10 breadth). Effective-N 7.50.
- **Notes:** whipsaws in intraday chop exactly as the prior warned; lands right beside the P3.7 KAMA cross (−12.948) — plain and adaptive MA crosses die together (a tight trend cluster).

---

## 6. Multi-factor combination results

*Phase-4 policy (operator ruling): the §6 cost pre-check was **informational only** — no battery was skipped; every composite ran the full panel battery. All are minimal AND-confluences (≤ 2 knobs), blind, with no construction drawn from any Phase-3 pattern. Numbers exploration-grade; DSR 0.000 throughout.*

### 6.1 — P4.1 · VWAP + Breakout Filter + Volume Surge
- **Hypothesis:** three independent confirmations (range break + VWAP trend-side + volume surge) should cut false breaks.
- **Spec (5-min, 2 knobs):** `vwap_breakout_volume` — `close > intraday_donchian_high` AND `close > VWAP` AND `relative_volume ≥ 1.5`. No-lookahead **PASSED**. §6 (informational): viable-side (4,068 trades, 0.432%).
- **Verdict: KILL.** aggregate **−8.202** (DSR 0.000, PBO 0.000, PF 0.25, 0/10 breadth). Effective-N 8.90.
- **Notes:** ≈ the single-factor breakout (P3.2 −8.753); the three-way confluence added nothing.

### 6.2 — P4.2 · ORB + VWAP Confirmation
- **Hypothesis:** the VWAP side filters low-quality opening breaks.
- **Spec (5-min, 2 knobs):** `orb_vwap` — opening-range break confirmed by the VWAP side. No-lookahead **PASSED**. §6: viable-side (2,488 trades, 0.524%).
- **Verdict: KILL.** aggregate **−5.395** (DSR 0.000, PBO 0.143, PF 0.37, 0/10 breadth). Effective-N 8.68.
- **Notes — logical redundancy (not a bug):** these numbers are **numerically identical** to the single-factor P3.11 ORB (−5.395 / PBO 0.143 / PF 0.37). Breaking above the opening-range high **already** puts `close` above the intraday VWAP, so the VWAP-side condition filters nothing — the confluence added *no information*. A genuine live-AND logical redundancy, not an inert-filter bug.

### 6.3 — P4.3 · Bollinger Band Mean Reversion + RSI Extreme
- **Hypothesis:** two exhaustion signals agreeing (band touch + RSI extreme) should improve the fade.
- **Spec (5-min, 2 knobs):** `bollinger_rsi` — fade `close` beyond the band AND `RSI` at an extreme; exit to the middle band. No-lookahead **PASSED**. §6: viable-side (3,513 trades, 0.325%).
- **Verdict: KILL.** aggregate **−13.126** (DSR 0.000, PBO 0.000, PF 0.09, 0/10 breadth). Effective-N 8.63.
- **Notes:** a fade confluence dies hard, like the other intraday fades (P3.3 −27.0, P3.8-C2 −25.6) — the two exhaustion signals **fail together in strong trends**, exactly the a-priori caution.

### 6.4 — P4.4 · Adaptive Moving Average + ADX
- **Hypothesis:** trade the KAMA trend only when ADX confirms real trend strength (gate out chop).
- **Spec (5-min, 2 knobs):** `adaptive_ma_adx` — KAMA slope held only while `ADX > 25`. No-lookahead **PASSED**. §6: cost-dead-side (16,236 trades, 0.133%).
- **Verdict: KILL — the worst result in the whole program.** aggregate **−39.807** (DSR 0.000, PBO 0.000, PF 0.00, net expectancy −1.21%/trade, 0/10 breadth). Effective-N 9.13.
- **Notes:** the ADX gate **did not cure** the KAMA-slope turnover — ~16k trades × the round-trip cost = catastrophic drag (cf. P3.7-V2, recorded §6-cost-dead; here the informational-§6 policy runs the full battery and makes the cost bleed explicit).

### 6.5 — P4.5 · Pivot Point Confluence + MACD Crossover
- **Hypothesis:** a MACD crossover at a classic pivot level reduces acting on dead levels.
- **Spec (5-min, 1 knob — lowest surface):** `pivot_macd` — a bullish MACD crossover near S1 (LONG) / bearish near R1 (SHORT); MACD fixed textbook. No-lookahead **PASSED**. §6: viable-side, sparse (986 trades, 0.426%). **Degeneracy watch RESOLVED** (986 trades — the level+crossover AND is not near-mutually-exclusive).
- **Verdict: KILL.** aggregate **−6.078** (DSR 0.000, PBO 0.000, PF 0.32, 0/10 breadth). 2,702 active days (sparse) but graded cleanly — **not** INSUFFICIENT. Effective-N 9.71.
- **Notes:** weak priors on both legs (pivots' weak base + MACD lag), as the a-priori caution warned.

### 6.6 — P4.6 · Donchian Channel Breakout + ATR Stop-Loss
- **Hypothesis:** an ATR-scaled trailing stop provides risk control on a clean breakout entry.
- **Spec (5-min, 2 knobs):** `donchian_atr_stop` — global Donchian breakout entry + ATR-trailing-stop exit. No-lookahead **PASSED**. §6: viable-side (4,041 trades, 0.360%).
- **Verdict: KILL.** aggregate **−9.215** (DSR 0.000, PBO 0.000, PF 0.20, 0/10 breadth). Effective-N 9.06.
- **Notes — distinct from P3.6 (identity check passed):** −9.215 ≠ P3.6 donchian's −6.132, and the trade count differs (4,041 vs ~3,298), so the ATR stop **is live** (no inert-exit bug). But the stop made the breakout **worse** — it cuts winners short and whipsaws out on intraday noise, adding turnover/cost. The honest "improves risk, not the entry" read, carried to its conclusion.

---

## 6b. Optional meta-labeling results (Phase 4.5)

**NOT CONDUCTED — out of scope for this program.** Meta-labeling (Phase 4.5) is an *optional* ML tier, gated on a strategy first showing a **gross edge** worth filtering. **No strategy qualified:** all 25 specs were KILLed or recorded §6-cost-dead; none exhibited a gross edge for a meta-model to refine. Phase 4.5 was therefore not run — a valid, expected outcome given the null. (Meta-labeling cannot manufacture an edge from a rule that has none; it can only prune the instances of a rule that already has positive gross expectancy.)

---

## 7. Cross-strategy synthesis

**Master ranking — the 21 battery-scored specs, least-bad → worst by aggregate CPCV path-Sharpe (net of cost).** All are decisively negative; breadth is 0/10 panel symbols for every spec except the three least-bad. The 4 §6-cost-dead specs (no panel Sharpe) follow. *Exploration-grade throughout.*

| # | Spec (study) | Aggregate CPCV | PF | Category |
|---|---|---|---|---|
| 1 | gap_fade (P3.10b) | −0.578 | 0.87 | gap fade |
| 2 | bull_flag (P3.12) | −0.651 | 0.87 | pattern |
| 3 | gap_and_go (P3.10) | −0.743 | 0.75 | gap continuation |
| 4 | orb_vwap (P4.2) *≡ ORB* | −5.395 | 0.37 | composite |
| 5 | opening_range_breakout (P3.11) | −5.395 | 0.37 | breakout |
| 6 | pivot_macd (P4.5) | −6.078 | 0.32 | composite |
| 7 | donchian_breakout (P3.6) | −6.132 | 0.33 | breakout/trend |
| 8 | vol_expansion_breakout (P3.8-C1) | −7.360 | 0.27 | vol-gated |
| 9 | vwap_breakout_volume (P4.1) | −8.202 | 0.25 | composite |
| 10 | breakout (P3.2) | −8.753 | 0.23 | breakout |
| 11 | donchian_atr_stop (P4.6) | −9.215 | 0.20 | composite |
| 12 | pivot_reversion (P3.5) | −9.848 | 0.19 | S/R fade |
| 13 | reversal (P3.4) | −9.910 | 0.15 | reversal |
| 14 | vwap_cross (P3.1-V2) | −11.121 | 0.17 | trend |
| 15 | ma_crossover (P3.14) | −11.745 | 0.15 | trend |
| 16 | adaptive_ma_cross (P3.7-V1) | −12.948 | 0.11 | trend |
| 17 | bollinger_rsi (P4.3) | −13.126 | 0.09 | composite fade |
| 18 | vwap_mean_reversion (P3.1-V1) | −18.793 | 0.01 | fade |
| 19 | vol_contraction_reversion (P3.8-C2) | −25.636 | 0.00 | vol-gated fade |
| 20 | mean_reversion (P3.3) | −27.040 | 0.00 | fade |
| 21 | adaptive_ma_adx (P4.4) | −39.807 | 0.00 | composite trend |

**§6-cost-dead (no panel battery — median per-trade gross < the 0.182% round-trip cost):** adaptive_ma_slope (P3.7-V2 — turnover), momentum_pullback (P3.9 — degenerate base + small gross), scalp_mean_reversion + scalp_momentum (P3.13 — 5-min proxy of a 1–3 min method).

**What distinguishes the failures.** There is no survivor to distinguish. Among the failures, three patterns are visible — recorded strictly as **candidate hypotheses for future work, NOT as "getting warmer":** every member of the least-bad cluster is decisively net-negative (negative Sharpe, PF < 1, breadth « the 60% bar).

1. **Turnover gradient (within trend/continuation bets).** Lower-turnover, longer-horizon continuation is less-bad: the slowest single-factor continuation (55-bar Donchian, −6.13) is the least-dead trend bet; faster MA crosses land worse (−11.7 to −12.9); the fastest adaptive slope died on turnover outright (cost-dead). A conditional pointer toward *low-turnover* continuation **if** any edge exists — magnitude firmly dead.
2. **Sparse big-move events are least-bad overall.** The two gap studies (−0.578, −0.743) and bull_flag (−0.651) — all trading sparsely on large-move events — are the three least-bad of the entire program (PF 0.75–0.87, net expectancy nearest zero). A pattern *within failures*, not an edge.
3. **Phase-4 meta-finding: confluence of dead factors mostly re-tests them.** Combining already-dead single factors manufactured no edge — the 6-composite slate added 28 raw trials (71 → 99) but only **−0.9 net effective** (9.95 → 9.06), because composites cluster tightly with their components. `orb_vwap` came back *numerically identical* to single-factor ORB (the VWAP leg logically redundant with an OR-high break); `adaptive_ma_adx` was the program's worst.

**Overfitting control worked as intended.** DSR was 0.000 for all 21 scored specs (each negative — the deflation never had a positive Sharpe to deflate). PBO cleared < 0.20 for all but two, and the sole criterion-3 (PBO) failure — bull_flag at PBO 0.400 — was exactly the largest-surface (4-knob) study: the gate **catching** the surface, not being fooled by it. Effective-N (9.06 on 99 raw) reflects the heavy clustering of correlated variants — the honest anti-double-counting.

## 8. Conclusion

**The honest bottom line: zero of the 25 specs cleared the seven-point kill-gate.** 15 single-factor specs and all 6 multi-factor composites KILLed on the panel battery; 4 single-factor specs were recorded §6-cost-dead (median per-trade gross below the ~0.18% round-trip cost). The Deflated Sharpe was **0.000 for every scored spec** (each had a negative CPCV Sharpe, so the multiple-testing deflation never had a positive result to shrink). The honest cumulative trial count was **99 raw → 9.06 effective** (correlated variants clustered). No spec is a bias/luck artifact to be explained away — there is no positive to explain.

**Precisely scoped — what this claims and what it does NOT.** The claim: *these classic, retail-accessible strategies — single-factor and in blind confluence — do not survive realistic Indian round-trip costs* on —

- **this universe:** the frozen **large-cap** NSE panel — a **10-name ADV-tiered scored panel** (HDFCBANK, RELIANCE, ICICIBANK, TCS, TATASTEEL, ULTRACEMCO, TITAN, COALINDIA, DRREDDY, TATACONSUM) plus a **5-name held-out** breadth set, drawn from the **frozen-49** survivor-only backfill. To reconcile the descriptors: the backfill holds **49 names** = **45 current NIFTY-50 members + 4 demoted names** (BPCL, BRITANNIA, HEROMOTOCO, INDUSINDBK — still trading, no longer in the index); the **5 current members ETERNAL / INDIGO / JIOFIN / MAXHEALTH / TMPV are absent** (a stale ~2024 list — see §2.1). Of those 49, **10 were scored** and **5 held out**;
- **at this resolution:** **5-minute OHLCV** only — no live depth / order-book; scalping's natural 1–3 min bars are unavailable;
- **exploration-grade:** survivor-only, so every number is a **provisional upper bound**, not a verdict-grade result.

This is **NOT** the claim that "no intraday edge exists." It does not touch: finer resolutions or order-flow data; smaller / less-efficient names outside the large-cap panel (where such edges are classically strongest, and are untested here); point-in-time-correct index membership; or non-classical / ML-combined signals. The everything-kills result is fully consistent with *"no durable, retail-accessible edge on this universe at this resolution with OHLCV alone"* — the truthful answer the whole apparatus was built to deliver, pass or fail.

**The held-out set was never spent.** The 5-name criterion-6d held-out set (INFY, SBIN, SUNPHARMA, ADANIPORTS, NESTLEIND) was reserved to *confirm a positive* — a strategy that cleared the panel would then have to hold on genuinely untouched names. Because **no strategy passed, the held-out set was never scored on any study** and remains a clean, unspent confirmation reserve for any future candidate.

**Next steps are gated on a survivor — of which there are none.** No productionization follows (it is gated on a passing verdict). Candidate directions for future work — a broader / less-efficient (small-cap) universe, structural time-of-day / cross-sectional / regime signals, finer bars, and an honestly-constructed ensemble — are recorded, deferred, and forward-only in `POST_PROJECT_DIRECTIONS.md`; none influenced this program's blind pre-registrations.

## 9. Reproducibility appendix

- **Data:** Kite Connect 5-minute OHLCV, 2015-02-02 → 2026-07-03, **frozen-49 survivor-only** backfill (immutable raw Parquet layer). Provenance in `config/universe/nifty50.yaml` (`as_of` snapshot) + `docs/DATA_MANIFEST.md`.
- **Universe files:** `config/universe/nifty50.yaml` (index list) + `config/universe/study_panel.yaml` (the frozen 10-name scored panel + 5-name held-out — **Lock A**, pre-committed 2026-07-09, not a tunable knob).
- **Config (all pre-committed, unchanged across every study):** `config/killgate.yaml` (the seven-point thresholds + the `panel:` two-part-verdict block), `config/costs.yaml` (Indian round-trip cost model), `config/default.yaml`.
- **Registry & pre-registrations:** every spec's frozen blind params in `src/lab/research/strategies/registry.py` (pinned by `tests/unit/test_registry.py`); each spec's pre-registration in `docs/pre_registration/` with its Result block, committed **before** its run.
- **Command (per study):** `uv run python scripts/run_panel_study.py --strategy <name> --interval 5minute --start 2015-02-02 --end 2026-07-03 --data-root data --ledger-dir data/ledger --config-dir config`. The 4 §6-cost-dead specs were recorded from the RELIANCE §6 pre-check (battery skipped by the pre-registered rule).
- **Environment:** Python 3.11; TA-Lib (prebuilt wheels), numpy, pandas, pyarrow, scipy, structlog. Trial ledger (effective-N) in `data/ledger/` (gitignored local state; the checkpoint numbers are in `docs/PROGRESS.md`). Session-by-session build log: `docs/PROGRESS.md`.

---

*Not financial advice. Research-only; no capital risked. Over 90% of retail F&O traders lose money; no profit is promised. The prior-evidence column in §4.2 is background context to be tested by this program, not a claim about returns.*
