# PROGRESS — Intraday Strategy Research Lab

*The authoritative, session-by-session build log. `MASTER_BLUEPRINT.md` Part VI mirrors the top-level status of this file. Update this at the end of **every** session, before the phase PR is opened.*

*Completion Standard (process note): every "done" recorded here carries **call-site evidence** — `file:line` of the call, not the definition — per `MASTER_BLUEPRINT.md` Part I §9 "Completion Standard", the Definition of Done + PR-review checklist this Phase-2→3 audit arc established. Reviews reject "done" without it.*

*Status: **Phase 3 in progress.** The corrected pipeline is on `main` — realized-frequency Sharpe (#17), Lock-A frozen panel (#18), `run_panel_study` two-part-verdict machinery (#19), intraday-reset breakout (#20/#21). **P3.1 · VWAP is complete:** both owed variants KILLed under panel scope, exploration-grade (frozen-49 large-caps) — V1 `vwap_mean_reversion` (fade, run first as the rebuild regression check, reproduced the pilot) and V2 `vwap_cross` (the owed cross), each pre-registered + signed off (#22) and, for V2, built (#24) BEFORE its run. Program ledger 0 → 18 (P3.1 VWAP both variants + P3.2 volume-breakout + P3.3 mean reversion — all KILL, exploration-grade, effective 2.18); nothing enters the `RESEARCH_FINDINGS.md §4.3` scorecard. Next study: P3.4 reversal trading. The gate tags (`gate-1-data`, `gate-2-harness`) predate their deliverables — treat them as never being deliverable snapshots; work from `HEAD`.*

---

## Gate status

| Gate | Meaning | State |
|---|---|---|
| Gate 0 | Foundation & scaffolding | ☑ |
| Gate 1 | Data & feature layer (Kite historical) | ☑ |
| Gate 2 | Research & validation harness | ☑ |
| Gate 3 | All 14 single-factor studies | ☐ |
| Gate 4 | All 6 multi-factor studies | ☐ |
| Gate 4.5 | Optional meta-labeling *(gated)* | ☐ |
| Gate 5 | Synthesis & findings | ☐ |

---

## Standing rule — frozen-49 catalog studies are EXPLORATION, not verdicts (2026-07-09)

Governs **every** Phase-3/4 catalog study while the universe is the **frozen-49 survivor-only** backfill (missing the 5 current NIFTY-50 members ETERNAL, INDIGO, JIOFIN, MAXHEALTH, TMPV — see `RESEARCH_FINDINGS.md` §2.1 and `config/universe/nifty50.yaml`):

- A study's kill-gate result is an **exploration signal (promising / not-promising), NOT a verdict-grade result.** It carries the **provisional / upper-bound stamp regardless of PASS or KILL**, and is **not recorded as a final verdict** in the `RESEARCH_FINDINGS.md` §4.3 scorecard.
- A **verdict-grade** run requires the 5 current members **backfilled first**. If a frozen-49 study looks promising, the next step is **backfill-the-5-then-rerun for the real verdict** — never "record the pass."
- **Believe the gate; never tune toward a pass.** Whatever the kill-gate says on a real strategy stands: do **not** adjust thresholds, the spec, or the universe to rescue a result. A KILL on a genuine idea is a real, valuable finding — log the honest negative and move to the next study. (The reference spec was built to fail; catalog strategies can genuinely pass or fail — the discipline is identical either way. This is the Inviolable-Rule-1 "kill-gate is sacred" discipline, restated for the first real study.)

**Honest-history notes (on record):**
- The Muhurat / regular-session filter **never trims the raw store** — it filters the intraday grid **downstream at the ingest boundary** (09:15–15:30); the raw Parquet is kept whole (same principle as the square-off fix — fix the boundary, keep the data).
- The pre-P3.1 clean-slate step **cleared nothing** — the system was already at **0 → 0**; it was an evidence-based **confirmation, not a destructive operation** (no ledger / results / paper data was deleted).

---

## Phase 3 readiness — BLOCKING issues (opened 2026-07-08, call-site audit)

> A call-site verification of the Gate-2 harness (proving each primitive is actually *invoked*, not merely defined) found four issues that must close before **Phase 3.1** opens. Two are genuine correctness gaps, not documentation. Until all four close, kill-gate criteria **3 (PBO)** and **6/7 (robustness/regime)** are **provisional**.

**Fix order:** B-1 first — once the gate fails closed on stub inputs it refuses to green-light any study whose 6/7 inputs aren't genuinely computed, which makes B-3/B-4 self-enforcing. Then B-2 (correctness) → B-3 → B-4.

| # | Blocker | Criterion | Evidence (call-site) |
|---|---|---|---|
| B-1 | **No stub/sentinel guard.** The gate grades a plausible-looking placeholder as real — e.g. `regime_bucket_medians=(1.0,)` passes criterion 7 on one fake bucket. Only NaN fails closed. | all, esp. 6/7 | `evaluate_kill_gate` is numeric-only with no INSUFFICIENT path (`killgate.py:122-195`); `Verdict` is PASS/KILL only (`types.py:52-56`) |
| B-2 | **PBO/CSCV misaligned + un-purged.** `np.column_stack([s[:length] …])` positionally aligns the *j-th trade* of each config (different timestamps) so CSCV blocks aren't time-coherent; no purge/embargo; does not share CPCV's overlap primitive. | 3 (PBO) | positional stack `study.py:465`; no `label_overlaps` in `pbo.py:66-71`; `pbo.py` imports nothing from `splitter` |
| B-3 | **Named Gate-2 test stubs 6/7.** `test_gate2_end_to_end…` would pass unchanged if the 6/7 machinery were deleted — it hand-feeds numbers instead of driving `run_study`. | 6/7 | `test_report_killgate.py:180` (param = base Sharpe), `:182` (cross-symbol = noise run of the *same* symbol), `:185-186` (regime = 1-tuple) |
| B-4 | **Default regime partition too weak.** Default is session-thirds (time-of-day), not blueprint criterion 7's *year × volatility/trend* — testing a weaker property than criterion 7 exists to test. | 7 | `_time_of_day_bucket` is the default labeler (`study.py:278-285`) vs `MASTER_BLUEPRINT.md:228` |

**Confirmed OK (not blockers):** CPCV *does* purge, via the shared `label_overlaps` primitive (`cpcv.py:130`; wired in PR #8 `d7b7c65`). The 6/7 *machinery* is real and wired in the orchestrator (`run_robustness_battery`/`regime_bucket_stats`, consumed at `study.py:404,416,430-436`) and covered by `test_study.py:189-202` — so B-3 is a test-honesty gap, not missing machinery.

### Resolution (2026-07-08) — all four fixed in the working tree (uncommitted)

| # | Fix | Call-site evidence |
|---|---|---|
| B-1 | Kill-gate returns `INSUFFICIENT` on a structurally under-shaped (stub) input — distinct-key cardinality/identity, never a value, so real extremes still grade. Keyed evidence for 6a/6d/7; summaries derived inside the gate. | `Verdict.INSUFFICIENT` (`types.py:59`); `_evidence_failures` (`killgate.py:181`) invoked at `killgate.py:238`, returns INSUFFICIENT at `killgate.py:240`; `run_study` feeds keyed evidence + provenance at `study.py:497-504`; floors in `config/killgate.yaml` `evidence:` (`min_regime_buckets: 4`, `min_cross_symbols: 3`) |
| B-2 | PBO rows aligned by trading day (per-config daily net P&L reindexed onto the union of days, no-trade = 0.0); purge/embargo routed through the same primitive CPCV uses. | shared `purge_indices` (`splitter.py:38`) called by CPCV (`cpcv.py:126`) **and** CSCV (`pbo.py:99`); day-aligned matrix in `_pbo_across_configs` (`study.py:536`), called at `study.py:464`; NaN PBO → criterion 3 fails closed (test `test_nan_pbo_fails_closed_not_pass_by_absence`) |
| B-3 | Gate-2 test now drives `run_study` (real 6/7 machinery), stubs deleted. | `run_study(...)` at `test_report_killgate.py:189` |
| B-4 | Default regime partition is now year × vol/trend, built from the scored candles. | `build_regime_labeler` (`study.py:304`) is the default at `study.py:481` |

**Criteria-1/4 stub-guard extension (done):** CPCV path-Sharpes are now keyed evidence (`cpcv_path_sharpes`, `killgate.py:118`); the gate derives median / positive-fraction / 10th-pct via the shared `cpcv_distribution_summary` (`cpcv.py:48`, also the single definition behind `CPCVResult.summary` at `cpcv.py:79`) and returns INSUFFICIENT below `min_cpcv_paths: 8` **finite post-purge** paths (`killgate.py:210-211`; the finite filter counts only scorable combinations, purged-empty ones score NaN in `cpcv.py:162-163`). Fed by `run_study` at `study.py:488`.

**Stub-guard invariant (now closed for the distribution criteria):** criteria **1, 4, 6a, 6d, 7** are un-forgeable — each arrives as machinery keyed evidence and the graded summary is derived inside the gate, so a hand-passed scalar cannot masquerade as a computed one. DSR (2), PBO (3), and P&L stats (5) remain single scalar machinery-outputs (production-computed via the ledger / PBO / trade-stats; never the stub vector) — closing those structurally would move their computation into the gate, out of this arc's scope.

**Remaining before Phase 3.1:** review + commit this batch. The non-blocking sweep below is optional and does not gate.

### Criterion-5 correction (2026-07-08) — expectancy hurdle was accidentally 2×, now matches the blueprint

Surfaced by the study-#1 machinery-validation run (first real-data `run_study`). Kill-gate **criterion 5's expectancy leg** compared a **net-of-cost** expectancy (`report.py:88`, mean of `Trade.net_return = gross − cost_fraction`, `backtester.py:49-51`) **against the round-trip cost a second time** (`killgate.py`, the old `i.expectancy > i.round_trip_cost`), so it required `net > cost ⇔ gross > ~2× round-trip cost` — an accidental double-count with no basis in the spec. The blueprint says "per-trade expectancy exceeds the modeled round-trip cost", counted once (`MASTER_BLUEPRINT.md:248`; restated `deep_dives/02:37`, `RESEARCH_FINDINGS.md:53`); the only other cost-hurdle framing is gross-vs-cost (`deep_dives/02:44`), and docs say "gross expectancy" explicitly when they mean pre-cost (`MASTER_BLUEPRINT.md:371`). **Corrected** the leg to `net expectancy > 0` — equivalently the gross edge exceeds the round-trip cost, counted once, on the **same participation-adjusted cost basis the backtester charges** (`costs.py:113`), so no flat-vs-participation double standard is introduced. Call-site: `killgate.py` criterion 5 (`i.expectancy > 0.0`); pinned by `test_criterion5_expectancy_is_net_positive_not_double_counted` (`tests/unit/test_report_killgate.py`) — a marginal edge with gross ∈ (1×, 2×) cost now **PASSES** (would have wrongly KILLed at 2×); a cost-dead edge (net ≤ 0) and break-even (net = 0) **FAIL**. Immaterial to study #1's KILL, and no published verdict depended on it (empty scorecard). Landed on branch `fix/killgate-criterion5-expectancy-single-count`, **merged via PR #12 (`a5444c9`)**.

### Pre-P3.1-proper hygiene batch + clean-slate confirmation (2026-07-08)

Three mechanical, **outside-the-gate** fixes before the first real catalog study (branch `fix/pre-p3.1-proper-hygiene`; confirmed none touches `killgate.py` or the gate criteria — data-boundary/display only):

- **Regular-session filter (Muhurat).** Raw 5-min data carries Diwali Muhurat evening bars (~18:15–19:15 IST, one/year). `regular_session_candles` (`data/hygiene/session.py`) via `NseCalendar.is_regular_session_time` filters the intraday grid to 09:15–15:30 at the **ingest boundary** (`scripts/run_study.py`), so out-of-session bars cannot enter feature/backtest computation; the immutable raw store is untouched (same principle as the square-off fix). Verified on real data (RELIANCE 2024-10..11 → `session_filter dropped=12`). Tests: `test_session_filter.py`, `test_is_regular_session_time`. `DATA_MANIFEST.md` grid claim corrected (09:15–19:15 raw; 09:15–15:30 used; Muhurat noted + square-off section marked resolved).
- **UTF-8 report output.** `scripts/run_study.py` forces UTF-8 stdout so the rendered report's glyphs (sparkline, ·, φ) print under a redirected pipe on Windows. Verified by reproducing the study-#1 cp1252 condition (no `PYTHONUTF8`, redirected) → exit 0, report printed.
- **CPCV path-count label (display only).** `render_report` shows "*N* combination-paths judged, φ=*M* reconstructed" — *N* = `n_finite_paths`, exactly the count the gate's evidence floor checks — instead of a bare "*M* paths". Test: `test_cpcv_label_shows_judged_count_and_phi_unambiguously`. No computation change.

**Clean-slate confirmation (precondition for P3.1-proper).** Verified with evidence that nothing from any test/validation run lingers where a catalog study would read from or append to:
- Trial ledger `data/ledger`: **0 files** — effective-N starts at 0.
- `RESEARCH_FINDINGS.md`: **no real verdict** — `reference_momentum` absent (study #1 correctly never `--paper`-written); §4.3 scorecard and §5/§6 templates still placeholders.
- No results / cache / paper-trade artifacts anywhere under `data/` (only `raw/` + empty `ledger/`).
- Program-wide ledger + results store at the same **0 → 0** as the Gate-2 checkpoint.
- (Validation runs used throwaway ledgers in the session scratchpad, **outside the repo** — not read by catalog studies, which default to `data/ledger`.)

**Non-blocking cleanups (sweep whenever):**
- Session log below stops at Phase 2; record PRs #6–#11 (Kite setup kit, feature-library completion, harness-gap fixes, ingestion/hygiene hardening, universe provenance, run_study CLI). **Note the tag-before-deliverable ordering:** `gate-1-data` and `gate-2-harness` were tagged *before* their deliverables were complete — feature-library completion (PR #7), the CPCV purge + orchestrator (PR #8 `d7b7c65`), and hygiene hardening (PR #9) all landed **post-tag**.
- Backtester squares off at the day's last bar, not the configured `square_off` 15:20 (`backtester.py:148-151` vs `default.yaml:68` / `nse_calendar.py:106-114`).
- Dead `pandas` / `pandas-stubs` dependency (`pyproject.toml:21,36` — no `import pandas` anywhere in `src`).
- Vestigial empty `config/universe.yaml` (superseded by `config/universe/nifty50.yaml`).
- Orphaned `PurgedKFold` (`splitter.py:46` — no call site in `src`/`scripts`; CPCV purges inline via the primitive). Delete or wire.

---

### Sharpe-convention correction + research-state reset (2026-07-09) — realized frequency replaces the fixed 18750 constant

**The catch.** A source audit found the pinned `sharpe.periods_per_year: 18750` (all 5-min bars in a year) was applied to **per-trade** return series, so a strategy trading a few hundred times a year was annualized as though it traded ~18,750 times a year — over-annualizing every Sharpe-magnitude criterion (**1, 4, 6a, 7**) toward false passes. This **contradicted the blueprint's own "in-market, not calendar time" rule** (`MASTER_BLUEPRINT.md:255`) — code-vs-label drift, found **before the first catalog verdict**.

**The fix (`fix/sharpe-realized-frequency`, held for review).** Annualize by each strategy's **realized frequency** — `realized_periods_per_year = trades ÷ operating-span-years`, computed per study from the base backtest (`sharpe.py:realized_periods_per_year`; threaded in `run_study` right after the base backtest, `study.py`). `SharpeConvention` + the `sharpe.periods_per_year`/`basis` config + the CLI plumbing are **removed**; `sharpe.py`/`cpcv.py` docstrings corrected to match behavior; `MASTER_BLUEPRINT.md:255` amended (contradiction removed).
- **Blast radius proven, not asserted (DSR/PBO/P&L unaffected):** **DSR (2)** consumes the *non-annualized* per-period Sharpe end-to-end (`metrics.py:6-8` docstring + path `study.py:484`→`ledger.deflated_sharpe`→`metrics.deflated_sharpe_ratio`, all per-period); **PBO (3)** ranks configs on an un-annualized `mean/std` and is rank-invariant to any positive scale (`pbo.py:40-46,110-122`); **P&L (5)** is sums/means/ratios (`report.py:64-91`). 6c/6d are sign-based → verdict-invariant. Only **1/4/6a/7** (annualized-Sharpe magnitude) move.
- **Near-zero-trades guard (data-dependent factor).** The factor is now `len(trades)/span`, so a thin base (a couple of lucky trades → small n, high per-trade mean) could yield an unstable/misleading annualized Sharpe — and CPCV cannot form `< n_groups` observations. Guarded: `< evidence.min_base_observations` (30) base trades → **INSUFFICIENT** (fail-closed, no trial logged; `study.py` short-circuit before CPCV), pinned by `test_run_study_insufficient_when_base_too_thin`. 30 sits far below any genuinely-trading strategy over the span (even 1 trade/week ≈ 570) and above the degenerate case.
- **Pinning test:** `test_annualization_uses_realized_not_fixed_frequency` — a low-freq (~100/yr) and a high-freq (~18,750/yr) strategy each annualize by √(their own rate); fails if the fixed 18750 creeps back.
- **P3.1 re-verified (isolated ledger, program ledger untouched):** realized **432 trades/yr** vs the old fixed 18,750 (annualization √-ratio **0.152×**); CPCV median path-Sharpe **−7.095** (pilot fixed-18750 was −46.733 — exactly 0.152× the pilot, i.e. de-inflated by the annualization ratio, not re-signed); verdict **still KILL** (6/7 fail, PBO passes). Confirming the blast-radius proof, **DSR (0.000), PBO (0.000), and the P&L stats (PF 0.27, net expectancy −0.00267) are byte-identical to the pilot** — only the annualized-Sharpe criteria (1/4/6a/7) moved, each by that ratio. The fix corrects the previously-inflated magnitude, never the verdict. Program ledger `data/ledger` confirmed still **0** after the isolated run.

**Research-state reset (2026-07-09, committed `45c932a`).** Ahead of the corrected Phase 3: the program trial ledger `data/ledger` is **reset to 0** (pre-reset evidence: it held exactly the 5 `vwap_mean_reversion` pilot config trials — base + entry±/exit± — and nothing else; `data/` holds only `raw/`+`ledger/`, so no DSR/PBO/CPCV output was ever persisted). The P3.1 VWAP run is **reclassified PILOT / SUPERSEDED** (single-symbol + pre-fix shakedown; kept as the honest record, not deleted; never in the §4.3 scorecard). Machinery, config, thresholds, and the frozen panel/subset are **untouched** — the reset is scoped to outputs + accumulated trial state only. Corrected Phase 3 starts its cumulative trial count at **0**.

---

## Session log

*One row per working session. A study-phase session records the category batch completed, not just "Phase 3."*

| Date | Phase / batch | Work done | Tests | PR / commit / tag | Notes |
|---|---|---|---|---|---|
| — | — | *(pre-Phase-0 scaffold: docs + directory skeleton only)* | — | — | Blueprint, README, findings scaffold, deep-dive outlines in place |
| 2026-07-05 | Phase 0 — Foundation | P0.1 tooling + CI + package skeleton · P0.2 layered config + secrets · P0.3 structured logging (IST, correlation IDs, redaction) · P0.4 NSE calendar · P0.5 domain types + interface Protocols | 61 unit tests; ruff + black + mypy (strict) + pytest green; pre-commit clean | branch `feat/p0-foundation` → PR #1 → `main`; tag `gate-0-foundation` | **Gate 0 passed.** Runtime deps kept minimal (pandas/numpy/TA-Lib/pyarrow deferred to the phases that use them). Effective-N DSR spec fix folded in. |
| 2026-07-05 | Phase 1 — Data & Feature Layer | P1.1 Kite historical adapter + daily auth · P1.2 Parquet archive (immutable raw + adjusted) · P1.3 resumable backfill + script · P1.4 hygiene (corp-actions, survivorship, bad-ticks, gaps, liquidity, ESM/T2T) · P1.5 point-in-time indicator library + dual-path skew harness · P1.6 leakage/skew suite in CI | 114 tests (incl. dual-path skew + adversarial leakage); ruff + black + mypy (strict) + pytest green; pre-commit clean | branch `feat/p1-data-layer` → PR to `main`; tag `gate-1-data` | **Gate 1 passed.** TA-Lib installs via prebuilt wheels (no C build) — CI green. Kite SDK isolated to `data/brokers/` (enforced by a test). |
| 2026-07-05 | Phase 2 — Research & Validation Harness | P2.1 purged CV + embargo, event-driven backtester (next-bar-open, square-off), full cost model · P2.2 CPCV path distribution + DSR/PSR + PBO via CSCV · P2.3 honest effective-N trial ledger (correlation participation ratio) · P2.4 StrategySpec adapter + reference spec · P2.5 robustness battery + two-engine reconciliation · P2.6 seven-point kill-gate + report + paper updater | 173 tests (incl. hand-computed costs/DSR/PBO, effective-N clustering, two-engine reconciliation, Gate 2 end-to-end); ruff + black + mypy (strict) + pytest green; pre-commit clean | branch `feat/p2-validation-harness` → PR to `main`; tag `gate-2-harness` | **Gate 2 passed.** DSR auto-deflates from the ledger's effective trial count; kill-gate thresholds pinned in `config/killgate.yaml`; the reference spec KILLs (correct honest outcome). scipy added for the (D/P)SR/PBO math. |
| 2026-07-09 | Phase 3 — P3.1 · VWAP (both owed variants) | Corrected-pipeline FIRST study, panel scope. V1 `vwap_mean_reversion` (fade) run first as the rebuild regression check → **KILL** (reproduces the pilot; per-symbol breadth −7.018 ≈ pilot RELIANCE −7.095). Built + registered V2 `vwap_cross` (owed cross; blind `cross_threshold` 0.002, 5-min) → **KILL**. §4 cross-specific 5-min rationale committed pre-run; stale √(periods_per_year) prose corrected to realized-frequency. | 18 new spec/registry tests (band-flip: entry / hold-through-band no-whipsaw / flip / daily-reset / prefix-invariance / factory); ruff + black + mypy strict + full pytest green | PRs #22 (corrected prereg, signed off), #24 (V2 spec + §4 rationale), + this write-up | **Both KILL**, exploration-grade (frozen-49 large-caps), NOT in the §4.3 scorecard. Ledger 0 → 8 (effective 1.37). Not a contradiction — opposite bets both dying is coherent. |
| 2026-07-09 | Phase 3 — P3.2 · Volume-filtered breakout | Panel scope. `breakout` (intraday-reset 20-bar range + rel-vol > 1.5 filter, ride the continuation to 15:20 square-off), blind params, 5-min — breakout-specific frequency rationale committed pre-run (15-min mechanically non-viable for a 20-bar reset range → coarser follow-up is a re-parameterized strategy, not a pure frequency variant); volume filter primary, volatility **entry** filter deferred as a ledger variant (distinct from the ATR **stop**). → **KILL** (aggregate CPCV −8.753, breadth −3.845 / 0.00 positive; PF 0.23). | no new code (breakout registered in #21); full pytest green | PR #26 (prereg sign-off + rationale) + this write-up | **KILL**, exploration-grade, NOT in §4.3. Ledger 8 → 13 (effective 1.81). No contradiction — fade + breakout both dying is coherent. |
| 2026-07-09 | Phase 3 — P3.3 · Mean reversion (intraday-reset z-score fade) | Panel scope. `mean_reversion` (fade \|z\|>2 back to within 0.5 of an INTRADAY-RESET rolling mean; new `intraday_zscore` indicator, gap-blind by design), blind params, 5-min (mechanical rationale — 20-bar reset window, like breakout). → **KILL**, the most decisive of the first four (aggregate CPCV −27.040, breadth −10.943 / 0.00; PF 0.00). | `intraday_zscore` + `MeanReversionSpec` + 16 new tests (incl. gap-blindness); ruff + black + mypy strict + full pytest green | PR #28 (prereg + indicator + spec) + this write-up | **KILL**, exploration-grade, NOT in §4.3. Ledger 13 → 18 (effective 2.18). Reinforces "no intraday MR edge on large-caps" (both fade anchors dead), not a contradiction. |

---

## Cumulative trial ledger — checkpoint

*The per-trial return streams live in `research/trials/`; the DSR is deflated by the **effective** (cluster-adjusted) trial count automatically, not by a raw variant count. Record a human-readable checkpoint here at each gate so the deflation is auditable at a glance.*

| At gate | Variants evaluated → effective-N | Notes |
|---|---|---|
| Gate 2 | 0 → 0 | Ledger built (P2.3): persists per-trial return streams; effective-N via correlation participation ratio. No study trials logged yet — Phase 3 populates it. |
| P3.1 · VWAP (both variants) | 8 → 1.37 | First corrected-pipeline study. V1 `vwap_mean_reversion` (5 configs: base + entry± + exit±) + V2 `vwap_cross` (3 configs: base + cross_threshold±) = 8 raw trials, all on the shared VWAP-deviation substrate → cluster to **1.37 effective** (the cluster-adjustment working as designed: 8 correlated one-idea trials ≠ 8 independent bets, so P3.2+ are not over-deflated). Both KILL, exploration-grade, not in the §4.3 scorecard. |
| P3.2 · breakout | 13 → 1.81 | Volume-filtered breakout, 5 configs (base + breakout_lookback± + volume_mult±). Cumulative with P3.1's 8 = 13 raw trials across three distinct ideas (VWAP fade, VWAP cross, volume-breakout) → **1.81 effective**. Breakout adds more independent weight than V2 did (a genuinely distinct strategy) but stays correlated (same panel / exposure / square-off). KILL, exploration-grade, not in the §4.3 scorecard. |
| P3.3 · mean_reversion | 18 → 2.18 | Intraday-reset z-score fade, 5 configs (base + entry_z± + exit_z±). Cumulative 18 raw trials across four ideas (VWAP fade, VWAP cross, volume-breakout, rolling-mean fade) → **2.18 effective**. The two mean-reversion fades (P3.3, P3.1 V1) cluster tightly (same fade direction) → `mean_reversion` adds only ~0.37 effective, not a full independent bet. KILL, exploration-grade. |

---

## Open decisions / surfaced ambiguities

*Per Inviolable Rule 8: when Part III is genuinely silent or self-contradictory on a decision, STOP and log it here with the options considered and the resolution. Nothing in Phase 0 was genuinely blocking; the decisions below were resolved within the Part I ground rules and Part III, and are recorded here for transparency.*

- **Config-driven NSE calendar (not a third-party calendar library).** The timezone, session boundaries, square-off time, and holiday list live in `config/default.yaml` and load into a typed `CalendarSettings`. Grounded in Part I §2 — "all parameters in versioned configuration; one source of truth; every run reproducible from its config." A vendored calendar package would bury exchange data outside our versioned source of truth and couple the calendar to a package's release cadence. The 2024–2025 holiday set is seeded from NSE circulars; it **must be verified and extended against the official NSE circular for the full backfill date range in Phase 1 (P1.4)** — Deep Dive 01 notes intraday depth caps CPCV/DSR sample power, so the exact range matters.
- **Square-off default 15:20 IST.** Zerodha MIS intraday auto-square-off policy, pinned as configuration. Part III Layer 2 mandates "intraday square-off at the configured session end"; the exact time is broker policy, so it is config, not a code literal.
- **Phase-0 runtime dependencies kept minimal (`pyyaml`, `structlog`, `tzdata`).** pandas, numpy, pyarrow, and TA-Lib are deferred to the phases that use them (P1.x/P2.x), keeping CI fast and green and avoiding an early pin of an unused native dependency. Grounded in Part I §2 and the "simpler, more robust, more testable" tie-breaker (Rules of Engagement).
- **black formats, ruff lints.** Per Part I §7 ("ruff + black"). Line length (100) is owned by black; ruff's `E501` is disabled so the two tools do not conflict. structlog was chosen for logging (clean processor pipeline for redaction, contextvars for correlation IDs) over hand-rolled stdlib logging.

**Phase 1 (Data & Feature Layer):**

- **TA-Lib via prebuilt wheels.** Part III says "prefer TA-Lib" (avoid hand-rolled indicator bugs). Modern `ta-lib` ships binary wheels for CPython on Windows and manylinux, so it installs with no local C build — honoring the blueprint without making CI fragile. Verified importing/computing on the dev machine and green in CI.
- **Feature parameters live in one typed `FeatureConfig`** (periods, band widths, opening-range window) rather than scattered magic numbers; the feature set is versioned (`FEATURE_SET_VERSION`). *Follow-up done:* now loaded from a `features:` section in `config/default.yaml` via `FeatureConfig.from_settings` (so `LAB__FEATURES__*` overrides flow through); Phase-3 study specs pin these per strategy.
- **NSE holiday list generated, not hand-typed** *(follow-up done, P1.4).* The 2018-2026 list in `config/default.yaml` is generated from the maintained `exchange_calendars` XBOM calendar (weekday non-sessions; NSE/BSE share equity holidays) via `scripts/generate_nse_holidays.py` — more accurate and complete than the original hand list (picks up special/ad-hoc closures). `exchange_calendars` is an optional `tooling` group, so CI and the runtime never depend on it; regenerate to widen the range.
- **Dual-path skew harness as the point-in-time contract.** Each feature is written once (vectorized); the incremental path is that same function on a prefix, and the harness asserts they match bar-by-bar. This structurally catches lookahead and is reused by the P1.6 leakage suite (which points it at deliberately leaky features). Grounded in Part III Layer 1 (train/serve skew tripwire).
- **Kite SDK isolation is test-enforced.** `kiteconnect` is imported only in `data/brokers/`; an architecture test fails if it appears elsewhere (Part I §1). pyarrow is isolated to `data/store/`. mypy stays strict on our code, tolerating the untyped SDKs via targeted overrides.
- **Raw immutability + adjusted-as-derived.** The raw Parquet layer refuses silent overwrite (atomic writes; re-write raises); corrections flow to the regenerable adjusted layer. This is what makes the backfill safely resumable (skip already-stored days).
