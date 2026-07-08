# Data Manifest — as-built Kite historical cache (survivor-only)

*Generated from `data/raw/` partition filenames + a first/last-session timestamp sample per symbol. The data dir is gitignored; this manifest makes the `RESEARCH_FINDINGS.md` §2 claim checkable without shipping the data. It is a point-in-time snapshot — regenerate with `scripts` after any backfill.*

## Summary

- **Symbols:** 49 (survivor-only; see `config/universe/nifty50.yaml` backfill block)
- **Interval:** 5minute (raw layer)
- **Partitions (files):** 137,137 (one per symbol/IST trading day)
- **Date range:** 2015-02-02 → 2026-07-03
- **Intraday grid:** regular-session bars **09:15 … 15:25** (5-min, within the 09:15–15:30 regular session). The raw archive **also contains Diwali Muhurat evening sessions out to ~19:15 IST** (one per year; see the Muhurat note below). The earlier "09:15 … 15:25" figure sampled only each symbol's first+last session, so it missed these mid-history evening bars.

## Square-off check (backtester vs configured MIS cutoff)

- Configured `calendar.session.square_off` = **15:20** (`config/default.yaml`).
- Observed **last bar timestamp = 15:25** — i.e. **> 15:20**.
- **Resolved.** The event-driven backtester now **honors the configured 15:20 cutoff** (merged in `fix/square-off-honor-cutoff`): with the grid running to 15:25, positions are forced flat at the last bar `< 15:20` and no entry opens at/after it — verified on live RELIANCE data (latest trade entry/exit 15:15). The 'harmless iff data ends ≤ 15:20' caveat no longer applies. Separately, the regular-session filter (below) drops any bar outside 09:15–15:30 before it reaches the backtester. Both boundaries are fixed downstream; the raw store is kept whole.

## Muhurat / out-of-session bars (regular-session filter)

- **Finding:** the raw 5-min archive contains **Diwali Muhurat evening-session** bars (~18:15–19:15 IST) — one session per year: 2015-11-11, 2016-10-30, 2017-10-19, 2018-11-07, 2019-10-27, 2020-11-14, 2021-11-04, 2022-10-24, 2023-11-12, 2024-11-01. For RELIANCE: **118 bars over 2015–2024 (0.056%)**. Genuine in the source parquet (correct IST, not a TZ/ingest bug) — Kite returns the Muhurat session; `config/default.yaml` treats these dates as regular-session holidays.
- **Actual grid:** 09:15 … **19:15** IST (raw); **regular session used for studies: 09:15–15:30.**
- **Handling:** the raw store is kept **whole** (never trimmed). Out-of-session bars (Muhurat evenings, pre-open, post-close) are filtered **downstream at the ingest boundary** — `regular_session_candles(...)` via `NseCalendar.is_regular_session_time`, wired in `scripts/run_study.py` — so they cannot enter feature or backtest computation. Same principle as the square-off cutoff (fix the boundary, keep the data).

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
