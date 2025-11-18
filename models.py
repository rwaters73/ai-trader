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
        Compute unrealized PnL% based on bid and avg_entry_price.
        Returns None if not computable.
        """
        if self.position_qty == 0.0:
            return None
        if self.avg_entry_price is None or self.avg_entry_price <= 0:
            return None
        if self.bid is None or self.bid <= 0:
            return None

        return (self.bid - self.avg_entry_price) / self.avg_entry_price * 100.0


@dataclass
class TargetPosition:
    """
    Strategy output: how many shares we *want* to hold after this cycle,
    plus a human-readable reason.

    entry_type / entry_limit_price are for *entry* orders (flat -> long, etc.).
    take_profit_price / stop_loss_price are suggested exit levels based on the
    entry price.
    """
    symbol: str
    target_qty: float
    reason: str
    entry_type: str = "market"                 # "market" or "limit" or future "bracket"
    entry_limit_price: Optional[float] = None  # price for LIMIT entry, if any
    take_profit_price: Optional[float] = None  # suggested TP level for this entry
    stop_loss_price: Optional[float] = None    # suggested SL level for this entry
