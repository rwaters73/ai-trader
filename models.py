from dataclasses import dataclass
from typing import Optional


@dataclass
class SymbolState:
    """
    Snapshot of everything the strategy needs to know about a symbol
    for one decision cycle.
    """
    symbol: str
    bid: Optional[float]             # latest bid price (may be None)
    ask: Optional[float]             # latest ask price (may be None)
    position_qty: float              # current position size (shares), can be 0.0
    avg_entry_price: Optional[float] # average entry price for the open position
    has_open_orders: bool            # True if any open orders exist for this symbol

    def pnl_percent(self) -> Optional[float]:
        """
        Compute unrealized PnL% based on current bid and average entry price.

        Returns:
          - float percentage if computable, e.g. 5.0 means +5%
          - None if there is no position or we lack valid pricing data.
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
    Strategy output for a symbol for a single decision cycle.

    Fields:
      - symbol: which instrument this decision refers to.
      - target_qty: how many shares we *want* to hold after this cycle.
      - reason: human-readable explanation for logging/debugging.

      - entry_type:
          "market" → broker should use a market order for entries.
          "limit"  → broker should use a limit order at entry_limit_price.
          (in future we might support "bracket" or other types.)

      - entry_limit_price:
          Price for LIMIT entry orders (only meaningful if entry_type == "limit").

      - take_profit_price / stop_loss_price:
          Optional suggested exit levels associated with this entry.
          Right now we’re mostly using separate EOD/exit logic, but these
          fields give us a place to store bracket-style intentions.
    """
    symbol: str
    target_qty: float
    reason: str

    entry_type: str = "market"                 # "market" or "limit" (future: "bracket")
    entry_limit_price: Optional[float] = None  # price for LIMIT entry, if any

    take_profit_price: Optional[float] = None  # suggested TP level for this entry
    stop_loss_price: Optional[float] = None    # suggested SL level for this entry
