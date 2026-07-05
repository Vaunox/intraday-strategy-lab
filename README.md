# Intraday Strategy Research Lab

A systematic, honest research apparatus for testing classic intraday trading strategies on liquid Indian (NSE) cash equities, using **Zerodha Kite Connect historical candle data** as the only data source.

The goal is not profit — it is an honest, cost-inclusive, overfitting-resistant **verdict per strategy**: does it hold a small, real, cost-surviving edge, or not? Most strategies are expected to fail that test, and an honest negative is treated as a complete, valuable result.

## What this repo does
1. Ingests and cleans Kite historical OHLCV data (corp-action adjusted, survivorship-correct, leakage-tested).
2. Builds a point-in-time feature & technical-indicator library.
3. Expresses each strategy as a deterministic rule-based `StrategySpec`.
4. Runs every strategy through a rigorous validation harness — purged CV, Combinatorial Purged CV, Deflated Sharpe (against an honest cumulative trial count), PBO, full Indian cost model, next-bar-open fills, robustness battery — and a **seven-point kill-gate**.
5. Records every verdict in a living research paper: [`docs/RESEARCH_FINDINGS.md`](docs/RESEARCH_FINDINGS.md).

**Research-only. No live trading, no order placement, no capital at risk.**

## Build order — research apparatus first, strategies second
The apparatus that judges a strategy is built **before** any strategy is tested. Phases 0–2 build the honest research machine (foundation → data & features → validation harness); only then, in Phases 3–4, are the strategies run through it. A strategy is never evaluated by a harness that does not yet enforce point-in-time correctness, full costs, and the kill-gate.

## Strategy slate
14 single-factor strategies (VWAP, breakout, mean-reversion, reversal, pivots, Donchian, adaptive MAs, volatility filters, momentum pullback, gap-and-go, ORB, bull flag, scalping, MA crossovers) and 6 multi-factor combinations. Full catalog and status in [`MASTER_BLUEPRINT.md`](MASTER_BLUEPRINT.md) (Part V).

## How to work this repo
Read [`MASTER_BLUEPRINT.md`](MASTER_BLUEPRINT.md) top to bottom and build it **one full phase per session**, respecting the phase gates. A phase is the session unit from start to gate — including the strategy phases, where a whole batch of studies sharing one harness is completed together (Phase 3 = all 14 single-factor studies; Phase 4 = all 6 multi-factor studies). Phases: 0 Foundation → 1 Data & Features → 2 Research Harness → 3 Single-Factor Studies → 4 Multi-Factor Studies → 5 Synthesis & Findings.

## Setup
> **Note:** this repo currently sits at its *pre-Phase-0* scaffold state. The toolchain files these commands need — `pyproject.toml`, `.pre-commit-config.yaml`, `.github/workflows/ci.yml`, and the initial test suite — are created in **Phase 0 (P0.1)**. The commands below work once Gate 0 is tagged; on a fresh scaffold clone they will not yet run.
```bash
# toolchain: uv (or your preferred venv manager) — available after Phase 0
uv sync
pre-commit install
pytest -q
```
Kite Connect credentials are supplied via the secrets interface / environment variables (never committed). See `docs/operator_runbooks/` for signup, daily auth, and backfill.

---

*Not financial advice. Trading carries substantial risk of loss; SEBI studies find over 90% of retail F&O traders lose money — a humility anchor on retail active trading, though this program trades cash-equity intraday, a distinct population.*
