# PROGRESS — Intraday Strategy Research Lab

*The authoritative, session-by-session build log. `MASTER_BLUEPRINT.md` Part VI mirrors the top-level status of this file. Update this at the end of **every** session, before the phase PR is opened.*

*Status: Phase 0 (Foundation & Scaffolding) complete — Gate 0 passed.*

---

## Gate status

| Gate | Meaning | State |
|---|---|---|
| Gate 0 | Foundation & scaffolding | ☑ |
| Gate 1 | Data & feature layer (Kite historical) | ☐ |
| Gate 2 | Research & validation harness | ☐ |
| Gate 3 | All 14 single-factor studies | ☐ |
| Gate 4 | All 6 multi-factor studies | ☐ |
| Gate 4.5 | Optional meta-labeling *(gated)* | ☐ |
| Gate 5 | Synthesis & findings | ☐ |

---

## Session log

*One row per working session. A study-phase session records the category batch completed, not just "Phase 3."*

| Date | Phase / batch | Work done | Tests | PR / commit / tag | Notes |
|---|---|---|---|---|---|
| — | — | *(pre-Phase-0 scaffold: docs + directory skeleton only)* | — | — | Blueprint, README, findings scaffold, deep-dive outlines in place |
| 2026-07-05 | Phase 0 — Foundation | P0.1 tooling + CI + package skeleton · P0.2 layered config + secrets · P0.3 structured logging (IST, correlation IDs, redaction) · P0.4 NSE calendar · P0.5 domain types + interface Protocols | 61 unit tests; ruff + black + mypy (strict) + pytest green; pre-commit clean | branch `feat/p0-foundation` → PR to `main`; tag `gate-0-foundation` | **Gate 0 passed.** Runtime deps kept minimal (pandas/numpy/TA-Lib/pyarrow deferred to the phases that use them). |

---

## Cumulative trial ledger — checkpoint

*The live count lives in `research/trials/` and feeds the DSR automatically. Record a human-readable checkpoint here at each gate so the deflation is auditable at a glance.*

| At gate | Cumulative trials evaluated | Notes |
|---|---|---|
| — | 0 | Ledger not yet initialized (created in Phase 2, P2.3) |

---

## Open decisions / surfaced ambiguities

*Per Inviolable Rule 8: when Part III is genuinely silent or self-contradictory on a decision, STOP and log it here with the options considered and the resolution. Nothing in Phase 0 was genuinely blocking; the decisions below were resolved within the Part I ground rules and Part III, and are recorded here for transparency.*

- **Config-driven NSE calendar (not a third-party calendar library).** The timezone, session boundaries, square-off time, and holiday list live in `config/default.yaml` and load into a typed `CalendarSettings`. Grounded in Part I §2 — "all parameters in versioned configuration; one source of truth; every run reproducible from its config." A vendored calendar package would bury exchange data outside our versioned source of truth and couple the calendar to a package's release cadence. The 2024–2025 holiday set is seeded from NSE circulars; it **must be verified and extended against the official NSE circular for the full backfill date range in Phase 1 (P1.4)** — Deep Dive 01 notes intraday depth caps CPCV/DSR sample power, so the exact range matters.
- **Square-off default 15:20 IST.** Zerodha MIS intraday auto-square-off policy, pinned as configuration. Part III Layer 2 mandates "intraday square-off at the configured session end"; the exact time is broker policy, so it is config, not a code literal.
- **Phase-0 runtime dependencies kept minimal (`pyyaml`, `structlog`, `tzdata`).** pandas, numpy, pyarrow, and TA-Lib are deferred to the phases that use them (P1.x/P2.x), keeping CI fast and green and avoiding an early pin of an unused native dependency. Grounded in Part I §2 and the "simpler, more robust, more testable" tie-breaker (Rules of Engagement).
- **black formats, ruff lints.** Per Part I §7 ("ruff + black"). Line length (100) is owned by black; ruff's `E501` is disabled so the two tools do not conflict. structlog was chosen for logging (clean processor pipeline for redaction, contextvars for correlation IDs) over hand-rolled stdlib logging.
