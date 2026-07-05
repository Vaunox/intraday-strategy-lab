# Operator runbook — Kite data setup & backfill

How to go from nothing to a populated local `data/` cache the strategy studies
(Phase 3) run on. **Everything here is run by you** — the credentials never leave
your machine. Historical candle data needs the **paid** Kite Connect plan (the
free Personal API has no market data).

> **Security:** secrets live ONLY in `.env` (git-ignored) and the git-ignored
> token file. Never paste them into chat, code, config, tickets, or git. If a
> secret is exposed, regenerate it in the Kite console immediately.

---

## 1. One-time setup

1. **Zerodha account** with an active demat/trading account.
2. **Kite Connect app** — at <https://developers.kite.trade>, create an app on the
   paid Connect plan. Note the **API key** and **API secret**. Set the app's
   **redirect URL** to anything you control (e.g. `https://127.0.0.1`); you only
   need to read the `request_token` off the redirected URL.
3. **Create `.env`** at the repo root from the template and fill in your values:
   ```bash
   cp .env.example .env
   # edit .env: KITE_API_KEY=... , KITE_API_SECRET=... , leave KITE_ACCESS_TOKEN blank
   ```
   `.env` is git-ignored — confirm with `git check-ignore .env`.
4. **Install deps:** `uv sync`.

## 2. Daily — mint the access token

Kite access tokens expire every day and require a manual 2FA/TOTP login.

```bash
uv run python scripts/kite_login.py              # prints the login URL
# open the URL, log in; you are redirected to <redirect_url>?request_token=XXXX&...
uv run python scripts/kite_login.py --request-token XXXX
```

The token is minted and saved to a git-ignored file (`secrets/kite_access_token.json`);
it is never printed. The instrument dump and backfill read it automatically.

## 3. Build the instrument-token map (once, or when the universe changes)

```bash
uv run python scripts/dump_instrument_tokens.py \
    --universe RELIANCE,TCS,INFY,HDFCBANK \
    --output secrets/instrument_tokens.json
```

Omit `--universe` to dump every NSE cash-equity (EQ) symbol. `--universe` also
accepts a path to a file with one symbol per line.

## 4. Backfill the historical candles

```bash
uv run python scripts/run_backfill.py \
    --symbols RELIANCE,TCS,INFY,HDFCBANK \
    --interval 5minute \
    --start 2022-01-01 --end 2024-12-31 \
    --instruments secrets/instrument_tokens.json \
    --data-root data
```

- **Intervals:** `minute`, `3minute`, `5minute`, `15minute`, `60minute`, `day`.
- **Resumable & idempotent:** re-running skips already-stored days — safe to
  interrupt and resume, and to extend the date range later.
- **Rate limit:** Kite historical is ~3 requests/second; the backfill paginates
  by trading-day windows. Minute-interval history is served in bounded windows
  and has shorter depth than daily — verify the depth available on your plan
  before committing to a long range.
- Data lands under `data/raw/symbol=.../interval=.../<date>.parquet` (git-ignored).

## 5. Then

Once `data/` is populated, the Phase 3 studies run against it through the frozen
validation harness (Gate 2). Verify the holiday calendar covers your date range
first — regenerate `config/default.yaml` holidays via
`scripts/generate_nse_holidays.py` if you backfill outside 2018–2026.

## Notes

- **Regenerating a token mid-day** is fine — re-run `scripts/kite_login.py`.
- **Corporate actions:** the raw archive stores as-fetched candles; the adjusted
  layer and corp-action adjustment are applied by the hygiene jobs (P1.4).
