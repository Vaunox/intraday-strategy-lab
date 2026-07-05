"""Full Indian intraday (MIS) cost model (Phase 2, P2.1).

Every simulation is net of costs — no gross-only backtests (Inviolable Rule 3).
A round trip (buy + sell) is charged brokerage (lower of a rate or a per-order
cap), STT (sell side), exchange transaction, SEBI turnover, GST (on
brokerage+exchange+SEBI), and stamp duty (buy side), plus size/liquidity-aware
slippage. NO DP charge (that is delivery, not MIS). All rates come from
``config/costs.yaml`` — never hard-coded here.
"""

from __future__ import annotations

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

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> CostModel:
        """Build a cost model from the parsed ``costs.yaml`` mapping."""
        return cls(
            brokerage_rate=float(mapping["brokerage"]["rate"]),
            brokerage_cap=float(mapping["brokerage"]["cap"]),
            stt_sell_rate=float(mapping["stt"]["sell_rate"]),
            exchange_rate=float(mapping["exchange_txn"]["rate"]),
            sebi_rate=float(mapping["sebi_turnover"]["rate"]),
            stamp_buy_rate=float(mapping["stamp"]["buy_rate"]),
            gst_rate=float(mapping["gst"]["rate"]),
            slippage_base_rate=float(mapping["slippage"]["base_rate"]),
            slippage_stress_multiplier=float(mapping["slippage"]["stress_multiplier"]),
        )

    def _brokerage(self, order_value: float) -> float:
        return min(self.brokerage_rate * order_value, self.brokerage_cap)

    def round_trip_cost(
        self, buy_value: float, sell_value: float, *, stressed: bool = False
    ) -> float:
        """Total round-trip cost in currency for a buy and a sell of given values."""
        brokerage = self._brokerage(buy_value) + self._brokerage(sell_value)
        stt = self.stt_sell_rate * sell_value
        exchange = self.exchange_rate * (buy_value + sell_value)
        sebi = self.sebi_rate * (buy_value + sell_value)
        gst = self.gst_rate * (brokerage + exchange + sebi)
        stamp = self.stamp_buy_rate * buy_value
        slip_rate = self.slippage_base_rate * (self.slippage_stress_multiplier if stressed else 1.0)
        slippage = slip_rate * (buy_value + sell_value)
        return brokerage + stt + exchange + sebi + gst + stamp + slippage

    def round_trip_cost_fraction(self, notional: float, *, stressed: bool = False) -> float:
        """Round-trip cost as a fraction of ``notional`` (buy and sell at ~one price)."""
        if notional <= 0:
            raise ValueError("notional must be positive")
        return self.round_trip_cost(notional, notional, stressed=stressed) / notional


def load_cost_model(config_dir: Path) -> CostModel:
    """Load the cost model from ``config_dir/costs.yaml``."""
    path = config_dir / "costs.yaml"
    data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return CostModel.from_mapping(data)
