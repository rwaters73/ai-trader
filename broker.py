from typing import Optional
from datetime import datetime

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus

from logger import log_order  # CSV order log
from data import get_latest_quote
from config import ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY
from models import SymbolState, TargetPosition
from db import log_order_to_db, log_order_event_to_db

# One shared trading client for the process
_trading_client = TradingClient(
    ALPACA_API_KEY_ID,
    ALPACA_API_SECRET_KEY,
    paper=True,
)


def build_symbol_state(symbol: str) -> SymbolState:
    """
    Gather quote, position, and open-order info into a SymbolState snapshot.
    """
    quote = get_latest_quote(symbol)
    bid = quote.bid_price
    ask = quote.ask_price

    qty, avg_entry_price = get_position_info(symbol)
    open_orders_exist = has_open_orders(symbol)

    print(f"\n=== {datetime.now().isoformat(timespec='seconds')} | Symbol: {symbol} ===")
    print(f"Bid: {bid}, Ask: {ask}")
    print(f"Current open quantity in {symbol}: {qty}")
    if avg_entry_price is not None:
        print(f"Average entry price: {avg_entry_price}")
    print(f"Has open orders in {symbol}: {open_orders_exist}")

    return SymbolState(
        symbol=symbol,
        bid=bid,
        ask=ask,
        position_qty=qty,
        avg_entry_price=avg_entry_price,
        has_open_orders=open_orders_exist,
    )


def get_position_info(symbol: str) -> tuple[float, Optional[float]]:
    """
    Return (qty, avg_entry_price) for the symbol in the paper account.
    If no open position, return (0.0, None).
    """
    positions = _trading_client.get_all_positions()

    for pos in positions:
        if pos.symbol == symbol:
            qty = float(pos.qty)  # fractional support
            avg_price = float(pos.avg_entry_price)
            return qty, avg_price

    return 0.0, None


def has_open_orders(symbol: str) -> bool:
    """
    Return True if there are any OPEN orders for this symbol.
    We treat that as 'in flight' and avoid stacking new orders.
    """
    request_params = GetOrdersRequest(
        status=QueryOrderStatus.OPEN,
        symbols=[symbol],
        nested=False,
        limit=50,
    )

    open_orders = _trading_client.get_orders(filter=request_params)
    return len(open_orders) > 0


def submit_market_order(symbol: str, qty: float, side: OrderSide, extended: bool = False):
    """
    Submit a market order. Returns the Alpaca order object or None
    if qty is non-positive.
    """
    if qty <= 0:
        print(f"[WARN] Not placing order for {symbol}: non-positive qty={qty}")
        return None

    order_request = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=side,
        time_in_force=TimeInForce.DAY,
        extended_hours=extended,
    )

    order = _trading_client.submit_order(order_request)

    # Log order submission to both CSV and SQLite
    if order is not None:
        # CSV log
        log_order(order)
        # DB log
        log_order_to_db(order)

        # If the order is already filled (typical for paper market orders)
        if getattr(order, "filled_at", None) is not None:
            log_order_event_to_db(
                alpaca_order_id=str(order.id),
                event_type="filled",
                filled_qty=float(order.filled_qty or 0),
                remaining_qty=float(order.qty) - float(order.filled_qty or 0),
                status=str(order.status),
            )

    return order

def submit_limit_order(symbol: str, qty: float, side: OrderSide, limit_price: float, extended: bool = False):
    return

def submit_bracket_order(
    symbol: str,
    qty: float,
    side: OrderSide,
    entry_type: str,
    entry_price: Optional[float],
    take_profit_price: float,
    stop_loss_price: float,
    extended: bool = False
):
    return


def reconcile_position(state: SymbolState, target: TargetPosition, extended: bool = False):
    """
    Given current state and target position, submit whatever order is needed
    to move from current position_qty to target_qty.
    """
    current = state.position_qty
    desired = target.target_qty
    delta = desired - current

    print(f"{state.symbol}: current={current}, target={desired} | {target.reason}")

    # No change needed
    if abs(delta) < 1e-6:
        print(f"{state.symbol}: No change in position. No order placed.")
        return

    side = OrderSide.BUY if delta > 0 else OrderSide.SELL
    qty = abs(delta)

    print(f"{state.symbol}: Submitting market order {side.name} qty={qty} to reach target.")
    order = submit_market_order(state.symbol, qty=qty, side=side, extended=extended)

    if order is not None:
        print(f"{state.symbol}: Order submitted:")
        print(order)


def flatten_symbol(symbol: str):
    """
    Flatten a single symbol (close long or cover short).
    """
    qty, _ = get_position_info(symbol)

    if abs(qty) < 1e-6:
        print(f"{symbol}: Already flat. No flatten order placed.")
        return

    side = OrderSide.SELL if qty > 0 else OrderSide.BUY
    print(f"{symbol}: Flattening position qty={qty} with {side.name} order.")
    submit_market_order(symbol, qty=abs(qty), side=side, extended=False)


def flatten_all(symbols: list[str]):
    """
    Flatten positions for all given symbols.
    """
    for symbol in symbols:
        flatten_symbol(symbol)


def cancel_all_open_orders() -> None:
    """
    Cancel all open orders across all symbols.

    This is intended to be called during the EOD window before applying
    EOD position policies, so we don't have stray orders filling after
    we've decided to be flat or hold.
    """
    request_params = GetOrdersRequest(
        status=QueryOrderStatus.OPEN,
        nested=True,
    )

    open_orders = _trading_client.get_orders(filter=request_params)

    if not open_orders:
        print("EOD: No open orders to cancel.")
        return

    print(f"EOD: Cancelling {len(open_orders)} open orders...")
    for order in open_orders:
        try:
            print(f" - Cancelling order {order.id} ({order.symbol}, {order.side}, qty={order.qty})")
            _trading_client.cancel_order_by_id(order.id)

            # Log cancel event to DB
            log_order_event_to_db(
                alpaca_order_id=str(order.id),
                event_type="canceled",
                filled_qty=float(order.filled_qty or 0),
                remaining_qty=float(order.qty) - float(order.filled_qty or 0),
                status="canceled",
            )
        except Exception as e:
            print(f"[WARN] Failed to cancel order {order.id}: {e}")
