# Deep Dive 01 — Data & Feature Layer (Layer 1)

> **Status: authored outline — expand on demand.** Part III of `MASTER_BLUEPRINT.md` is the self-contained, authoritative distillation of this layer and is sufficient to build Phases 0–1 from. This file exists so the reference resolves and so deeper detail can be added where a task needs it. Nothing here overrides Part III; if the two ever diverge, Part III governs and the divergence is logged in `docs/PROGRESS.md`.

## Scope
Everything from raw Kite historical candles to a versioned, point-in-time, leakage-proof feature/indicator vector behind swappable interfaces.

## The three silent killers (design them out structurally, not by discipline)
1. **Lookahead leakage** — any feature or signal touching data unavailable at `asof`. Defended by pure point-in-time functions, trailing/expanding normalization only, and the CI leakage suite (P1.6).
2. **Survivorship bias** — testing only names that exist today. Defended by point-in-time index constituents, including delisted/renamed symbols.
3. **Train/serve skew** — vectorized backfill computing a feature differently from the incremental path. Defended by the dual-path harness: vectorized output must equal bar-by-bar incremental output (the skew tripwire, P1.5).

## Connectivity (Kite historical) — points to expand
- REST historical-candle endpoint; ~3 requests/sec data limit; paginate long ranges.
- **Intraday history depth is shorter than daily** and is served in bounded windows (chunk the backfill accordingly); verify the actual minute-bar depth available under the paid Connect plan before committing to a date range — it caps CPCV/DSR sample power for every study.
- Daily access-token flow (token expires daily; SEBI-mandated 2FA/TOTP). Manual daily auth is the compliant path; the token artifact must land in a git-ignored location.
- All access behind the `BrokerAdapter` protocol; nothing else imports the Kite SDK.

## Storage — points to expand
- Parquet on local disk, partitioned by symbol/date; immutable raw layer + a derived/adjusted layer; all behind `Repository`.
- Corrections become new versions, never silent mutations.

## Hygiene jobs (each idempotent, tested, logged)
NSE calendar + IST timestamps + session tagging; corporate-action adjustment (store raw *and* adjusted); point-in-time constituents/survivorship; bad-tick filtering (log every correction); gap detection; liquidity screen (high ADV, tight spread); ESM/T2T exclusion.

## Feature & indicator families (all point-in-time pure functions, OHLCV-derived)
Price/return transforms; realized vol, ATR, Parkinson/Garman-Klass ranges; VWAP + deviation; pivots (classic/Fib/Camarilla); Donchian; SMA/EMA + adaptive MAs (KAMA/AMA); ADX/DMI; RSI; MACD; Bollinger + %B; ATR bands/stops; opening range; gap measures; volume surge / relative volume; momentum & pullback measures; candlestick/pattern primitives; time-of-day encodings; cross-sectional ranks; regime features. Prefer TA-Lib for standards; hand-roll only what it lacks, with tests.

## Feature-store contract
`compute_features(symbol, asof) → versioned vector`, used identically in backfill and (hypothetical) serving; CI asserts vectorized == incremental and runs the leakage battery.

## To expand later (as tasks demand)
- Exact corp-action adjustment math and the raw-vs-adjusted read paths.
- The point-in-time constituents source and how delisted names are reintroduced.
- Per-indicator point-in-time definitions and their TA-Lib bindings.
- The full leakage/skew test list and the injected-leakage fixtures.
