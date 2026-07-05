"""Full Indian intraday (MIS) cost model (Phase 2, P2.1).

Every simulation is net of costs — no gross-only backtests (Inviolable Rule 3).
A round trip (buy + sell) is charged brokerage (lower of a rate or a per-order
cap), STT (sell side), exchange transaction, SEBI turnover, GST (on
brokerage+exchange+SEBI), and stamp duty (buy side), plus size/liquidity-aware
slippage. NO DP charge (that is delivery, not MIS). All rates come from
``config/costs.yaml`` — never hard-coded here.

Slippage is **size- and liquidity-aware**: the price concession grows with the
order's participation in the bar's traded value under a square-root market-impact
law (Almgren et al.), ``rate = base + impact_coefficient * sqrt(participation)``,
with ``participation`` capped so an illiquid/zero-volume bar cannot produce an
unbounded cost. A flat cost (``participation = 0``) recovers the base rate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class CostModel:
    """Itemized Indian intraday cost model (all rates are fractions of turnover)."""

    brokerage_rate: float
    brokerage_cap: float
    stt_sell_rate: float
    exchange_rate: float
    sebi_rate: float
    stamp_buy_rate: float
    gst_rate: float
    slippage_base_rate: float
    slippage_stress_multiplier: float
    #: Square-root market-impact coefficient (0.0 => flat, size-independent slippage).
    slippage_impact_coefficient: float = 0.0
    #: Cap on participation (order / bar traded value) fed to the impact law, so an
    #: illiquid or zero-volume bar cannot produce an unbounded slippage.
    slippage_participation_cap: float = 0.10

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> CostModel:
        """Build a cost model from the parsed ``costs.yaml`` mapping."""
        slippage = mapping["slippage"]
        return cls(
            brokerage_rate=float(mapping["brokerage"]["rate"]),
            brokerage_cap=float(mapping["brokerage"]["cap"]),
            stt_sell_rate=float(mapping["stt"]["sell_rate"]),
            exchange_rate=float(mapping["exchange_txn"]["rate"]),
            sebi_rate=float(mapping["sebi_turnover"]["rate"]),
            stamp_buy_rate=float(mapping["stamp"]["buy_rate"]),
            gst_rate=float(mapping["gst"]["rate"]),
            slippage_base_rate=float(slippage["base_rate"]),
            slippage_stress_multiplier=float(slippage["stress_multiplier"]),
            slippage_impact_coefficient=float(slippage.get("impact_coefficient", 0.0)),
            slippage_participation_cap=float(slippage.get("participation_cap", 0.10)),
        )

    def _brokerage(self, order_value: float) -> float:
        return min(self.brokerage_rate * order_value, self.brokerage_cap)

    def slippage_rate(self, participation: float, *, stressed: bool = False) -> float:
        """Per-side slippage rate under the square-root impact law.

        ``participation`` is the order value as a fraction of the bar's traded
        value; it is floored at 0 and capped at ``slippage_participation_cap`` so
        a thin or zero-volume bar yields a bounded (worst-case) concession.
        """
        capped = min(max(participation, 0.0), self.slippage_participation_cap)
        rate = self.slippage_base_rate + self.slippage_impact_coefficient * math.sqrt(capped)
        return rate * (self.slippage_stress_multiplier if stressed else 1.0)

    def round_trip_cost(
        self,
        buy_value: float,
        sell_value: float,
        *,
        participation: float = 0.0,
        stressed: bool = False,
    ) -> float:
        """Total round-trip cost in currency for a buy and a sell of given values.

        ``participation`` (order / bar traded value) drives the size/liquidity-aware
        slippage; the default of 0.0 recovers the flat base-rate slippage.
        """
        brokerage = self._brokerage(buy_value) + self._brokerage(sell_value)
        stt = self.stt_sell_rate * sell_value
        exchange = self.exchange_rate * (buy_value + sell_value)
        sebi = self.sebi_rate * (buy_value + sell_value)
        gst = self.gst_rate * (brokerage + exchange + sebi)
        stamp = self.stamp_buy_rate * buy_value
        slip_rate = self.slippage_rate(participation, stressed=stressed)
        slippage = slip_rate * (buy_value + sell_value)
        return brokerage + stt + exchange + sebi + gst + stamp + slippage

    def round_trip_cost_fraction(
        self, notional: float, *, participation: float = 0.0, stressed: bool = False
    ) -> float:
        """Round-trip cost as a fraction of ``notional`` (buy and sell at ~one price)."""
        if notional <= 0:
            raise ValueError("notional must be positive")
        cost = self.round_trip_cost(
            notional, notional, participation=participation, stressed=stressed
        )
        return cost / notional


def trade_cost_fraction(
    cost_model: CostModel,
    notional: float,
    entry_price: float,
    entry_volume: float,
    *,
    stressed: bool = False,
) -> float:
    """Per-trade round-trip cost fraction, liquidity-aware via the entry bar.

    Participation is the order ``notional`` over the entry bar's traded value
    (``entry_price * entry_volume``); a non-positive traded value (a halt/thin
    bar) yields the maximum (capped) participation, i.e. the worst-case impact.
    Both backtest engines route through this one function so their per-trade
    costs — and therefore the two-engine reconciliation — stay identical.
    """
    traded_value = entry_price * entry_volume
    participation = notional / traded_value if traded_value > 0 else float("inf")
    return cost_model.round_trip_cost_fraction(
        notional, participation=participation, stressed=stressed
    )


def load_cost_model(config_dir: Path) -> CostModel:
    """Load the cost model from ``config_dir/costs.yaml``."""
    path = config_dir / "costs.yaml"
    data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return CostModel.from_mapping(data)
