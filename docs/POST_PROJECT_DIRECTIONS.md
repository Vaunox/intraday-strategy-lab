# Post-project research directions

> **Deferred until the single-factor catalog (P3.1–P3.14) and multi-factor phase are complete. Recorded for future work. None of these are active; do not let them influence current pre-registrations.**

*Status: **record only.** This is a durable roadmap so the directions and their reasoning survive across sessions. Do **not** act on any item here while the catalog / multi-factor phase is in progress, and do **not** let any of it shape a current pre-registration. If you ever find yourself about to import an idea from this doc into an active-catalog study, **stop and flag it** — that is exactly the leak this separation exists to prevent.*

---

## Section 1 — Candidate directions

Each is a hypothesis to test *later*, through the same blind, pre-registered, unchanged kill-gate. Each carries its own caution, and the caution is load-bearing.

### 1. Broader / less-efficient universe

**Hypothesis.** Re-run the same strategies on a wider, liquidity-tiered universe (NIFTY 200 / 500) that reaches smaller, less-efficient names — with **honest thinner-name cost modeling** (fatter slippage, circuit-limit exclusions).

**Rationale.** Edges are structurally more likely where institutional competition is thinnest; large-caps are the most-arbitraged corner of the market. This is the **highest-value next experiment** because it isolates the cleanest open question: *is the everything-kills result the strategy, or the universe?*

**Caution.** The illiquidity that *creates* the edge also *endangers* the execution. The backtest is **least reliable exactly where the edge looks best** — so treat any small-cap edge with **more** skepticism, not less.

### 2. Structural time-of-day effects

**Hypothesis.** Open/close auction flows, index-rebalancing flows, first-30-minute intraday momentum.

**Rationale.** Durable **because they arise from forced institutional flows** (index funds *must* trade at the close) — not patterns that get arbitraged away.

**Caution.** Still requires honest cost survival — a structural flow is not an edge until it clears the round-trip cost.

### 3. Cross-sectional relative strength

**Hypothesis.** Fire a signal only when the stock's sector / peers confirm — e.g. take the breakout only when the sector is *also* leading.

**Rationale.** Adds **institutional-flow / cross-sectional information the isolated-stock catalog never tested** — the biggest genuine gap in the whole program.

**Caution.** The choice of **ranking variable and reference universe is itself an overfitting surface**, even though cross-sectional ranks avoid absolute-threshold tuning.

### 4. Regime-conditional strategy selection / filtering

**Hypothesis.** Deploy a strategy only in the market state where it has edge: an **India VIX / realized-vol overlay for the volatility axis**, and a **lagging trend measure for the direction axis**.

**Rationale.** A strategy dead *on average* can be excellent *in its native regime*; filtering to that regime can **invert the math, not just lift it**.

**Two load-bearing cautions:**

- **(a) Real-time regime detection is itself a hard prediction problem.** You must classify the regime **before it reveals itself**, from lagging indicators that are wrong **exactly at the transitions** that matter most. Lean on the **reliable, causal volatility axis** (VIX / realized-vol) and **distrust the trend axis**.
- **(b) "The filter is the edge" makes the filter the highest-risk overfitting surface.** A filter tuned on history **deletes exactly the visible historical losses**, producing a backtest that looks identical to a genuine one. So: the filter must be **pre-registered blind, causal, with every filter parameter counted in the trial penalty, judged at the decisive bar, and equally willing to kill.** **Tell:** a real conditional edge shows up with **one or two** blind filters; if you keep *adding* conditions because it "almost passes," that is overfitting announcing itself.

### 5. Context-conditioned hypotheses

**Hypothesis.** Regime filter, relative strength, time-of-day window — the same family as #4, generalized.

**Discipline (identical to #4).** The objective is **honest edge-detection, NOT "make something pass the gate."** Each context condition is pre-registered blind, the gate is unchanged, and **every condition is counted as a trial**.

### 6. Original strategy from catalog learnings

**Hypothesis.** A new strategy informed by the *pattern across all catalog results* — where **learnings shape the mechanism, never the fitted parameters** — pre-registered blind through the same gate.

**Anchor already on record.** The **continuation gradient**: continuation bets get *less-bad* as they get **slower / longer-horizon / lower-turnover** (logged in `PROGRESS.md` → "Cross-study observation — the continuation gradient"). Any original strategy should inherit that *direction* as a mechanism prior, never as a tuned number.

### 7. Probabilistic-classifier / ML tier

**Hypothesis (gated).** Only pursue if **simple rules keep failing** *and* the work has moved to a universe / signal-combination where **weak signals genuinely stack**.

**Form.** **Calibrated-probabilistic** (not hard-label), carrying the **sacred-recalibration discipline** — walk-forward OOS, explicit promotion, **never auto-swap** — plus institutional staleness defenses: a **live-vs-expected IC monitor**, **governed retraining**, and a **measured decay rate**. Borrow the daily **Alpha-Hunter** system's *discipline and architecture*, **not** its daily-horizon model.

---

## Section 2 — The capstone architecture

Everything in Section 1 feeds one destination: **an ensemble of structurally-distinct, individually-weak, genuinely-uncorrelated edges, judged decisively at the aggregate level.** The full rule set:

- **Components may be weak — decisiveness is demanded at the ensemble, not the component.** A component Sharpe of ~0.35 is fine. Killing every slightly-positive component individually **destroys the building blocks** the ensemble is made of.
- **Independence is earned and measured, never assumed.** √N diversification is real but **proportional to actual uncorrelatedness**. The catalog's own **effective-N** (well below the raw trial count) already shows the current strategies are **correlated directional bets on the same prices**. Genuine independence comes from **structurally distinct edge sources** (time-of-day, cross-sectional, regime) that **fail at different times** — not variants of the same bet.
- **1/N equal weighting for the gate test** — zero allocation parameters, no weight-overfitting. **Mean-variance / Markowitz is banned as the gate allocator** (an error-maximizer that exploits lucky in-sample non-correlation).
- **Pre-committed membership.** Every edge meeting a **blind inclusion criterion** goes in; you may **NOT drop members to flatter the aggregate** — that is selection overfitting migrating from weights to membership.
- **Effective-N-penalized DSR at the aggregate.** The existing machinery prices the real correlation: **genuine diversification clears the bar; fake diversification collapses effective-N and is killed.**
- **Inverse-vol parity is a deferred refinement, never a rescue.** If the ensemble dies under pure 1/N and only lives under vol-parity, **be suspicious** — that is estimation error creeping back in.

**Through-line.** If a **blindly equal-weighted** ensemble of **structurally-distinct** edges with **pre-committed membership** cannot **decisively clear the effective-N-penalized gate**, then the edges are **too weak or too correlated — and it dies.** No reweighting, no membership-pruning, no vol-parity lifeline.

---

## Section 3 — The overriding caveat

**None of this guarantees an edge exists.** It is a **sound architecture for finding one honestly if it is there, and for not fooling yourself if it isn't.** The everything-kills catalog result is **fully consistent** with *"no durable retail edge on this universe with OHLCV data."* The value of the whole program is a **truthful answer, pass or fail — not a manufactured edge.**
