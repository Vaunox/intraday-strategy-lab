# Phase 3 — Handoff (corrected pipeline, clean start line)

*A map, not a history. The corrected Phase-3 machinery is built and merged; the first real study runs in the NEW session, gated only by the P3.1 pre-registration sign-off. Read this first, then `MASTER_BLUEPRINT.md`, `docs/PROGRESS.md`, `docs/RESEARCH_FINDINGS.md`, and the per-study `docs/pre_registration/`.*

## 0. EXACT first action for the new session

1. **Read** this handoff + all `.md` (blueprint, PROGRESS, RESEARCH_FINDINGS, pre_registration/).
2. **Confirm on `main`:** the Sharpe fix, panel machinery, intraday-reset, P3.2-under-panel, and the frozen `study_panel.yaml` are all merged (they are — see §1); `git log --oneline` shows merges of PRs #17–#21.
3. **Confirm the corrected P3.1 pre-registration** — BOTH variants (VWAP reversion re-run + the owed VWAP cross), full panel scope, blind params, committed-before-run. It is drafted + committed on **held PR #22** (`feat/p3.1-vwap-panel-prereg`) — NOT yet on main, NOT run. Review it; revise the blind V2 `cross_threshold` if your prior differs (revising before any run preserves pre-registration integrity).
4. **HOLD for operator sign-off** (integrity gate 2). Do NOT run P3.1 until signed off.
5. On sign-off: run it via `scripts/run_panel_study.py --strategy vwap_mean_reversion …` (and the cross variant). It doubles as the rebuild regression check — a known-KILL on VWAP-reversion confirms the corrected pipeline end-to-end before trusting any new verdict.

## 1. Current state (what's merged / on `main`)

- **#17 Sharpe fix** — realized-frequency annualization; the fixed `18750` constant + `SharpeConvention` removed. `MASTER_BLUEPRINT.md:255` amended.
- **#18 Lock-A frozen panel** — `config/universe/study_panel.yaml` on main (the panel config that was stranded on the old P3.2 branch; recovered).
- **#19 panel machinery** — `lab.research.panel.run_panel_study` (two-part verdict) + `panel:` thresholds in `killgate.yaml`.
- **#20 intraday-reset** — `intraday_donchian` (beside `donchian`); `BreakoutSpec` retargeted.
- **#21 P3.2-under-panel** — shared `lab.research.strategies.registry` (breakout registered), `scripts/run_panel_study.py` driver, `P3.2_breakout.md` amended to panel scope. **Not run.**
- **Program trial ledger `data/ledger` = 0.** No study has run on it. P3.1's pilot 5 trials were cleared (reset) and the pilot reclassified — see §7.
- Gates green throughout (ruff/black/mypy strict, full pytest). Old PR #16 is **closed** (its content pulled forward; main is the single source of truth).

## 2. Merge model (operator ruling, 2026-07-09) — see [[no-attribution-and-merge-gating]]

- **Claude owns ALL merges for the rest of the project.** build → **prove at the call site (file:line)** → merge (`gh pr merge N --merge --delete-branch`, merge commit not squash). Call-site proof + stop-and-flag ARE the review substitute; the operator does not merge.
- **The operator's ONLY two approval gates are integrity gates:** (a) the **frozen-panel Lock** (Lock A — DONE, on main via #18); (b) the **P3.1 pre-registration sign-off** (PENDING).
- **No AI/tool attribution anywhere** (hard rule; commits authored as `Vaunox <nevesia26@gmail.com>`).
- **Stop-and-flag** (overrides autonomy): a new correctness issue / blueprint conflict / number-behaviour mismatch; an "already done" item found missing or off-mainline (spot-check as you go); a design ambiguity that would bias a study if frozen wrong; anything that would touch settled infrastructure (frozen sets, thresholds, the seven-criterion logic, the Sharpe fix).

## 3. Settled rulings (do NOT re-litigate)

**Study scope — per-symbol-then-aggregate panel.** Single-symbol scoring is insufficient. Every catalog study (P3.x) runs via `run_panel_study` across the FROZEN panel.
- **Frozen sets (Lock A, `study_panel.yaml`):** panel (10) = HDFCBANK, RELIANCE, ICICIBANK, TCS, TATASTEEL, ULTRACEMCO, TITAN, COALINDIA, DRREDDY, TATACONSUM; criterion-6d held-out (5, never scored) = INFY, SBIN, SUNPHARMA, ADANIPORTS, NESTLEIND. BEL→ULTRACEMCO swap applied (BEL is trend/breakout-favourable). **Frozen — never changed to rescue a study.**
- **Two-part verdict (BOTH required for PASS):** (1) the equal-weight panel-**portfolio** daily stream clears the seven-point kill-gate; (2) **breadth** — median per-symbol CPCV path-Sharpe **> 1.0** AND **≥ 60%** of panel symbols individually positive (positive = CPCV median > 0).
- **Contribute-zero aggregation:** a symbol with no trade on a day contributes 0 to the equal-weight mean, divisor = fixed panel size (thin-participation dilutes toward zero; not re-weighted). A trading day is the unit (intraday square-off) — pooling into one trade series is INCORRECT (cross-symbol same-timestamp leakage across CPCV purge).
- **Ledger: K aggregate-portfolio streams per study, NOT N×K.** The panel is scope, not extra trials — N×K would over-deflate every future DSR.
- **Pinned `panel:` thresholds (`killgate.yaml`):** breadth_median_path_sharpe_min 1.0; breadth_positive_fraction_min **0.60** (not 0.70: the frozen-49 are correlated large-caps, a real edge misses on choppy laggards → 0.70 risks false-KILL); noise_survives_fraction_min 0.60; two_engine_reconcile_fraction_min 1.0 (all symbols); min_panel_symbols 6; min_portfolio_days 250.
- **Corrected Sharpe convention:** every Sharpe annualized by its **REALIZED frequency** (`realized_periods_per_year` = observations ÷ operating-span-years), never a fixed constant — the daily panel aggregate at its ~252/yr, each per-symbol stream at its own trade rate. DSR/PBO/P&L consume per-period/rank/count values → unaffected (proven at the call site).
- **Criterion 7 (regime) — panel-index vol/trend, EXOGENOUS.** Year × hi/lo-vol × up/down-trend read off an equal-weight panel PRICE index (the market itself, not the strategy's P&L) — non-circular. Frozen as `regime_method` in `study_panel.yaml`; must not drift to year-only (easier gate) or P&L-derived (circular).
- **Scope caveat auto-stamped on every panel result:** *large-caps only (frozen-49, ADV ₹122–1647 cr); a KILL = "no edge on large-caps", not "no edge anywhere".*
- **Fail-closed guards:** structural floors (min_cpcv_paths 8, min_cross_symbols 3, min_regime_buckets 4, min_panel_symbols 6, min_portfolio_days 250) → **INSUFFICIENT**, never a pass-by-absence. **Near-zero-trades guard:** a base with `< min_base_observations` (30) trades → INSUFFICIENT (the realized-frequency factor is data-dependent, so a few lucky trades can't certify).

## 4. Dual-variant + frequency protocol

- **Genuine DIRECTIONAL dichotomies — BOTH owed** (surface at pre-registration, never silently pick one): **P3.1 VWAP** cross/reversion; **P3.7 Adaptive-MA** cross/slope; **P3.13 Scalping** MR/momentum; **P3.10 Gap** and-go / gap-fade (**added 2026-07-10** — a directional opposite must not be direction-cherry-picked; the gap-fill fade is the owed twin `gap_fade`, run as a genuine distinct trial, NOT a life-gated variant).
- **METHOD / FILTER / FREQUENCY choices — pick the natural PRIMARY**, log it, run the other only as a **ledger-charged variant if the primary shows life**. Frequency: `MASTER_BLUEPRINT.md:179` — the natural frequency is spec-set; "5/15-min" is a candidate range, not both-owed. A 5+15 pair of one idea is correlated (clusters ~1 trial); 15-min lowers turnover so it can flip a cost verdict — note the frequency + that cost-viability is frequency-specific.

## 5. Standing run discipline

- **Pre-register FIRST, committed before the run** (the commit timestamp is the honesty proof): hypothesis; mechanism (not pattern); **blind** frozen params (never peek at data); unchanged kill-gate + panel thresholds; the spec; cost-viability pre-check.
- **Cost-viability pre-check** (median |gross/trade| vs ~0.18% round-trip) → cost-dead is the logged finding, skip the battery.
- **Believe the gate; never tune** spec/params/thresholds/frozen-sets to rescue a result. A logged honest KILL/INSUFFICIENT is the process working.
- **Exploration-grade on frozen-49:** provisional + large-cap stamp regardless of PASS/KILL; **never** written to the `RESEARCH_FINDINGS.md §4.3` scorecard as a verdict. A promising frozen-49 result triggers **backfill-the-5-then-rerun** for a verdict — never "record the pass."
- **ABSOLUTE HOLD:** any open operator question / pending ruling freezes ALL runs until ruled.
- **STOP-and-flag (do not auto-proceed):** verdict PASS or near-threshold; INSUFFICIENT; any pipeline error / unexpected number; a result contradicting a prior study (e.g. a bet and its near-opposite both passing).
- **Batching (P3.2–P3.14):** run 3–4 studies (or until a STOP), report the compact record per study (hypothesis, blind params + prereg commit hash, cost-viability, verdict + per-criterion table, running cumulative effective-N). See [[catalog-study-protocol]].
- **Batch-draft ruling (operator, 2026-07-10).** The remaining single-factor preregs (P3.10 gap-and-go · P3.11 ORB · P3.12 bull flag · P3.13 scalping [both-owed: MR + momentum, two specs] · P3.14 MA crossovers) are **drafted + committed in ONE blind block**, all params frozen blind / a-priori — **nothing from the continuation gradient or `POST_PROJECT_DIRECTIONS.md` may enter any construction** (if a mechanism seems to need a learned input → STOP-and-flag, don't import it). This **increases blindness** vs per-study drafting (no design can be shaped by an earlier study's outcome). **Running stays incremental** (batches of 3–4 or until a STOP), STOP-and-flag intact; **pre-run revision allowed for a genuine mechanism bug** (P3.7/P3.9 precedent). Each prereg surfaces its **§6 landing** and **flags a degenerate trade-count** base rather than recording a hollow median. **All six held for individual sign-off before any run.** Phase 4 (multi-factor) EXCLUDED — drafted later. *Data note:* the archive holds **5-min bars only**, so **P3.13 scalping is tested at 5-min** (coarser-than-natural proxy; flagged).

## 6. Running the machinery

- Panel study: `uv run python scripts/run_panel_study.py --strategy <name> --interval 5minute --start 2015-02-02 --end 2026-07-03 --data-root data --ledger-dir data/ledger --config-dir config`. Panel/held-out come from the frozen `study_panel.yaml` (not CLI args). Registered strategies: `reference_momentum`, `vwap_mean_reversion`, `breakout` (`lab.research.strategies.registry`).
- Verification/throwaway runs use an **isolated ledger dir** (never `data/ledger`), so the program ledger stays 0 until a signed-off study.

## 7. P3.1 status

- **The 2026-07-09 P3.1 VWAP run is PILOT / SUPERSEDED** (`docs/pre_registration/P3.1_vwap.md` result block) — single-symbol + pre-Sharpe-fix shakedown; kept as the honest record, never in the scorecard. Pilot knowledge retained: VWAP-fade had **no reversion edge** (lost on direction, not cost); the pipeline validated. Its 5 trials were cleared from the ledger (now 0).
- **The corrected P3.1 (the real study) is the FIRST run of the new session:** BOTH variants — VWAP **reversion** re-run + the owed VWAP **cross** — under full panel scope, blind params. Drafted-and-committed on **held PR #22** (`docs/pre_registration/P3.1_vwap_panel.md`; merges on sign-off). On sign-off: merge #22, implement + register the **V2 cross spec** (`VwapCrossSpec` / `vwap_cross`, not yet built — its logic + blind params are in the prereg §3-4), run V1 first as the rebuild regression check, then V2.
