from dataclasses import dataclass
from typing import Optional


@dataclass
class SymbolState:
    """
    Snapshot of everything the strategy needs to know about a symbol
    for one decision cycle.
    """
    symbol: str
    bid: Optional[float]
    ask: Optional[float]
    position_qty: float
    avg_entry_price: Optional[float]
    has_open_orders: bool

    def pnl_percent(self) -> Optional[float]:
        """
        Computes unrealized PnL% based on the current bid and avg_entry_price.
        Returns None if PnL cannot be computed (e.g., no position, missing data).
        """
        if self.position_qty == 0:
            return None
        if self.avg_entry_price is None:
            return None
        if self.bid is None or self.bid <= 0:
            return None

        return (self.bid - self.avg_entry_price) / self.avg_entry_price * 100.0


@dataclass
class TargetPosition:
    """
    Strategy output: how many shares we *want* to hold after this cycle,
    plus a human-readable reason.
    """
    symbol: str
    target_qty: float
    reason: str
