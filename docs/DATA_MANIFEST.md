# Data Manifest — as-built Kite historical cache (survivor-only)

*Generated from `data/raw/` partition filenames + a first/last-session timestamp sample per symbol. The data dir is gitignored; this manifest makes the `RESEARCH_FINDINGS.md` §2 claim checkable without shipping the data. It is a point-in-time snapshot — regenerate with `scripts` after any backfill.*

## Summary

- **Symbols:** 49 (survivor-only; see `config/universe/nifty50.yaml` backfill block)
- **Interval:** 5minute (raw layer)
- **Partitions (files):** 137,137 (one per symbol/IST trading day)
- **Date range:** 2015-02-02 → 2026-07-03
- **Intraday grid (sampled, every symbol's first+last session):** 09:15 … 15:25 (5-min bars)

## Square-off check (backtester vs configured MIS cutoff)

- Configured `calendar.session.square_off` = **15:20** (`config/default.yaml`).
- Observed **last bar timestamp = 15:25** — i.e. **> 15:20**.
- **Consequence:** the event-driven backtester squares off at the day's *last bar* (`backtester.py`), so with this data it holds MIS positions through the 15:25 bar (~2 bars / ~10 min past the 15:20 cutoff). The 'harmless iff data ends ≤ 15:20' caveat is **not** satisfied — the square-off cleanup item (backtester should honor `square_off`, or the data be trimmed to ≤ 15:20) is a **live** correction to make before real studies, not hypothetical.

## Candle count

- Pinned in `RESEARCH_FINDINGS.md` §2 / `nifty50.yaml`: **8,958,811** candles.
- That implies **~65.3 rows/partition** over 137,137 partitions — consistent with 75-bar max sessions (09:15–15:30) minus holidays, half-days, and bad-tick/blank drops at ingest. Plausible, not a discrepancy.
- **Exact per-symbol row counts were skipped** (a full footer scan of 137,137 files is ~14 min — non-trivial). Regenerate exactly with a `pyarrow.dataset(...).count_rows()` pass per symbol if the precise figure is needed.

## Per-symbol partition spans

| Symbol | First session | Last session | Partitions |
|---|---|---|---|
| ADANIENT | 2015-02-02 | 2026-07-03 | 2,827 |
| ADANIPORTS | 2015-02-02 | 2026-07-03 | 2,827 |
| APOLLOHOSP | 2015-02-02 | 2026-07-03 | 2,827 |
| ASIANPAINT | 2015-02-02 | 2026-07-03 | 2,827 |
| AXISBANK | 2015-02-02 | 2026-07-03 | 2,827 |
| BAJAJ-AUTO | 2015-02-02 | 2026-07-03 | 2,827 |
| BAJAJFINSV | 2015-02-02 | 2026-07-03 | 2,827 |
| BAJFINANCE | 2015-02-02 | 2026-07-03 | 2,827 |
| BEL | 2015-02-02 | 2026-07-03 | 2,827 |
| BHARTIARTL | 2015-02-02 | 2026-07-03 | 2,827 |
| BPCL | 2015-02-02 | 2026-07-03 | 2,827 |
| BRITANNIA | 2015-02-02 | 2026-07-03 | 2,827 |
| CIPLA | 2015-02-02 | 2026-07-03 | 2,827 |
| COALINDIA | 2015-02-02 | 2026-07-03 | 2,827 |
| DRREDDY | 2015-02-02 | 2026-07-03 | 2,827 |
| EICHERMOT | 2015-02-02 | 2026-07-03 | 2,827 |
| GRASIM | 2015-02-02 | 2026-07-03 | 2,827 |
| HCLTECH | 2015-02-02 | 2026-07-03 | 2,827 |
| HDFCBANK | 2015-02-02 | 2026-07-03 | 2,827 |
| HDFCLIFE | 2015-02-11 | 2026-07-03 | 2,137 |
| HEROMOTOCO | 2015-02-02 | 2026-07-03 | 2,827 |
| HINDALCO | 2015-02-02 | 2026-07-03 | 2,827 |
| HINDUNILVR | 2015-02-02 | 2026-07-03 | 2,827 |
| ICICIBANK | 2015-02-02 | 2026-07-03 | 2,815 |
| INDUSINDBK | 2015-02-02 | 2026-07-03 | 2,827 |
| INFY | 2015-02-02 | 2026-07-03 | 2,815 |
| ITC | 2015-02-02 | 2026-07-03 | 2,815 |
| JSWSTEEL | 2015-02-02 | 2026-07-03 | 2,827 |
| KOTAKBANK | 2015-02-02 | 2026-07-03 | 2,827 |
| LT | 2015-02-02 | 2026-07-03 | 2,827 |
| M&M | 2015-02-02 | 2026-07-03 | 2,827 |
| MARUTI | 2015-02-02 | 2026-07-03 | 2,827 |
| NESTLEIND | 2015-02-02 | 2026-07-03 | 2,827 |
| NTPC | 2015-02-02 | 2026-07-03 | 2,827 |
| ONGC | 2015-02-02 | 2026-07-03 | 2,827 |
| POWERGRID | 2015-02-02 | 2026-07-03 | 2,827 |
| RELIANCE | 2015-02-02 | 2026-07-03 | 2,826 |
| SBILIFE | 2015-02-02 | 2026-07-03 | 2,176 |
| SBIN | 2015-02-02 | 2026-07-03 | 2,824 |
| SHRIRAMFIN | 2015-02-02 | 2026-07-03 | 2,827 |
| SUNPHARMA | 2015-02-02 | 2026-07-03 | 2,827 |
| TATACONSUM | 2015-02-02 | 2026-07-03 | 2,827 |
| TATASTEEL | 2015-02-02 | 2026-07-03 | 2,827 |
| TCS | 2015-02-02 | 2026-07-03 | 2,826 |
| TECHM | 2015-02-02 | 2026-07-03 | 2,827 |
| TITAN | 2015-02-02 | 2026-07-03 | 2,827 |
| TRENT | 2015-02-02 | 2026-07-03 | 2,823 |
| ULTRACEMCO | 2015-02-02 | 2026-07-03 | 2,827 |
| WIPRO | 2015-02-02 | 2026-07-03 | 2,827 |
