# PROGRESS — Intraday Strategy Research Lab

*The authoritative, session-by-session build log. `MASTER_BLUEPRINT.md` Part VI mirrors the top-level status of this file. Update this at the end of **every** session, before the phase PR is opened.*

*Status: Phase 2 harness built and `gate-2-harness` tagged. A **2026-07-08 call-site audit** found four Phase-3-blocking issues; **all four (B-1..B-4) plus the approved criteria-1/4 stub-guard extension are now fixed in the working tree** (224 tests green; ruff + black + mypy strict clean) — see "Phase 3 readiness" below. Changes are **not yet committed**. **Phase 3.1 opens once the batch is reviewed and committed.** The gate tags (`gate-1-data`, `gate-2-harness`) predate their deliverables — treat them as never being deliverable snapshots; work from `HEAD`.*

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

**Non-blocking cleanups (sweep whenever):**
- Session log below stops at Phase 2; record PRs #6–#11 (Kite setup kit, feature-library completion, harness-gap fixes, ingestion/hygiene hardening, universe provenance, run_study CLI). **Note the tag-before-deliverable ordering:** `gate-1-data` and `gate-2-harness` were tagged *before* their deliverables were complete — feature-library completion (PR #7), the CPCV purge + orchestrator (PR #8 `d7b7c65`), and hygiene hardening (PR #9) all landed **post-tag**.
- Backtester squares off at the day's last bar, not the configured `square_off` 15:20 (`backtester.py:148-151` vs `default.yaml:68` / `nse_calendar.py:106-114`).
- Dead `pandas` / `pandas-stubs` dependency (`pyproject.toml:21,36` — no `import pandas` anywhere in `src`).
- Vestigial empty `config/universe.yaml` (superseded by `config/universe/nifty50.yaml`).
- Orphaned `PurgedKFold` (`splitter.py:46` — no call site in `src`/`scripts`; CPCV purges inline via the primitive). Delete or wire.

---

## Session log

*One row per working session. A study-phase session records the category batch completed, not just "Phase 3."*

| Date | Phase / batch | Work done | Tests | PR / commit / tag | Notes |
|---|---|---|---|---|---|
| — | — | *(pre-Phase-0 scaffold: docs + directory skeleton only)* | — | — | Blueprint, README, findings scaffold, deep-dive outlines in place |
| 2026-07-05 | Phase 0 — Foundation | P0.1 tooling + CI + package skeleton · P0.2 layered config + secrets · P0.3 structured logging (IST, correlation IDs, redaction) · P0.4 NSE calendar · P0.5 domain types + interface Protocols | 61 unit tests; ruff + black + mypy (strict) + pytest green; pre-commit clean | branch `feat/p0-foundation` → PR #1 → `main`; tag `gate-0-foundation` | **Gate 0 passed.** Runtime deps kept minimal (pandas/numpy/TA-Lib/pyarrow deferred to the phases that use them). Effective-N DSR spec fix folded in. |
| 2026-07-05 | Phase 1 — Data & Feature Layer | P1.1 Kite historical adapter + daily auth · P1.2 Parquet archive (immutable raw + adjusted) · P1.3 resumable backfill + script · P1.4 hygiene (corp-actions, survivorship, bad-ticks, gaps, liquidity, ESM/T2T) · P1.5 point-in-time indicator library + dual-path skew harness · P1.6 leakage/skew suite in CI | 114 tests (incl. dual-path skew + adversarial leakage); ruff + black + mypy (strict) + pytest green; pre-commit clean | branch `feat/p1-data-layer` → PR to `main`; tag `gate-1-data` | **Gate 1 passed.** TA-Lib installs via prebuilt wheels (no C build) — CI green. Kite SDK isolated to `data/brokers/` (enforced by a test). |
| 2026-07-05 | Phase 2 — Research & Validation Harness | P2.1 purged CV + embargo, event-driven backtester (next-bar-open, square-off), full cost model · P2.2 CPCV path distribution + DSR/PSR + PBO via CSCV · P2.3 honest effective-N trial ledger (correlation participation ratio) · P2.4 StrategySpec adapter + reference spec · P2.5 robustness battery + two-engine reconciliation · P2.6 seven-point kill-gate + report + paper updater | 173 tests (incl. hand-computed costs/DSR/PBO, effective-N clustering, two-engine reconciliation, Gate 2 end-to-end); ruff + black + mypy (strict) + pytest green; pre-commit clean | branch `feat/p2-validation-harness` → PR to `main`; tag `gate-2-harness` | **Gate 2 passed.** DSR auto-deflates from the ledger's effective trial count; kill-gate thresholds pinned in `config/killgate.yaml`; the reference spec KILLs (correct honest outcome). scipy added for the (D/P)SR/PBO math. |

---

## Cumulative trial ledger — checkpoint

*The per-trial return streams live in `research/trials/`; the DSR is deflated by the **effective** (cluster-adjusted) trial count automatically, not by a raw variant count. Record a human-readable checkpoint here at each gate so the deflation is auditable at a glance.*

| At gate | Variants evaluated → effective-N | Notes |
|---|---|---|
| Gate 2 | 0 → 0 | Ledger built (P2.3): persists per-trial return streams; effective-N via correlation participation ratio. No study trials logged yet — Phase 3 populates it. |

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
