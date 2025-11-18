from typing import Optional
from datetime import datetime

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    GetOrdersRequest,
    #BracketOrderRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus

# Try to import StopOrderRequest; if not available in this alpaca-py version,
# we'll skip placing real SL orders and just log a warning.
try:
    from alpaca.trading.requests import StopOrderRequest  # type: ignore
except ImportError:
    StopOrderRequest = None  # type: ignore

from logger import log_order  # CSV order log
from data import get_latest_quote
from models import SymbolState, TargetPosition
from db import log_order_to_db, log_order_event_to_db
from config import (
    ALPACA_API_KEY_ID,
    ALPACA_API_SECRET_KEY,
    BRACKET_TP_PERCENT_BY_SYMBOL,
    DEFAULT_BRACKET_TP_PERCENT,
    BRACKET_SL_PERCENT_BY_SYMBOL,
    DEFAULT_BRACKET_SL_PERCENT,
)

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

def get_open_orders_for_symbol(symbol: str):
    """
    Return a list of OPEN orders for the given symbol.
    """
    request_params = GetOrdersRequest(
        status=QueryOrderStatus.OPEN,
        symbols=[symbol],
        nested=False,
        limit=50,
    )
    return _trading_client.get_orders(filter=request_params)


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

    # ---- NEW: log order submission ----
    if order is not None:
        log_order_to_db(order)

        # If the order is already filled (typical for paper market orders)
        if order.filled_at is not None:
            log_order_event_to_db(
                alpaca_order_id=str(order.id),
                event_type="filled",
                filled_qty=float(order.filled_qty or 0),
                remaining_qty=float(order.qty) - float(order.filled_qty or 0),
                status=str(order.status)
            )

    return order

def submit_limit_order(
    symbol: str,
    qty: float,
    side: OrderSide,
    limit_price: float,
    extended: bool = False,
):
    """
    Submit a limit order. Returns the Alpaca order object or None
    if qty is non-positive or limit_price is non-positive.
    """
    if qty <= 0:
        print(f"[WARN] Not placing LIMIT order for {symbol}: non-positive qty={qty}")
        return None

    if limit_price <= 0:
        print(f"[WARN] Not placing LIMIT order for {symbol}: non-positive limit_price={limit_price}")
        return None

    order_request = LimitOrderRequest(
        symbol=symbol,
        qty=qty,
        side=side,
        time_in_force=TimeInForce.DAY,
        limit_price=limit_price,
        extended_hours=extended,
    )

    order = _trading_client.submit_order(order_request)

    if order is not None:
        log_order_to_db(order)

        # If it's immediately filled (paper often does this)
        if order.filled_at is not None:
            log_order_event_to_db(
                alpaca_order_id=str(order.id),
                event_type="filled",
                filled_qty=float(order.filled_qty or 0),
                remaining_qty=float(order.qty) - float(order.filled_qty or 0),
                status=str(order.status),
            )

    return order

def submit_stop_loss_order(
    symbol: str,
    qty: float,
    stop_price: float,
    extended: bool = False,
):
    """
    Submit a SELL stop-loss order for an existing LONG position.

    This assumes:
      - We are long the symbol (qty > 0).
      - We want a STOP-MARKET SELL at stop_price.

    If StopOrderRequest is not available in this alpaca-py version, this
    function will log a warning and do nothing.
    """
    if qty <= 0:
        print(f"[WARN] Not placing STOP order for {symbol}: non-positive qty={qty}")
        return None

    if stop_price <= 0:
        print(f"[WARN] Not placing STOP order for {symbol}: non-positive stop_price={stop_price}")
        return None

    if StopOrderRequest is None:
        print(
            f"[WARN] alpaca.trading.requests.StopOrderRequest not available in this "
            f"alpaca-py version; skipping SL order for {symbol} at {stop_price:.2f}."
        )
        return None

    order_request = StopOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
        stop_price=stop_price,
        extended_hours=extended,
    )

    order = _trading_client.submit_order(order_request)

    if order is not None:
        log_order_to_db(order)

        if order.filled_at is not None:
            log_order_event_to_db(
                alpaca_order_id=str(order.id),
                event_type="filled",
                filled_qty=float(order.filled_qty or 0),
                remaining_qty=float(order.qty) - float(order.filled_qty or 0),
                status=str(order.status),
            )

    return order



# def submit_bracket_order(
#     symbol: str,
#     qty: float,
#     entry_limit_price: float,
#     take_profit_price: float,
#     stop_loss_price: float,
#     extended: bool = False,
# ):
#     """
#     Submit a bracket order:
#        ENTRY:     LIMIT at entry_limit_price
#        TAKE PROFIT: LIMIT at take_profit_price
#        STOP LOSS:   STOP at stop_loss_price

#     Returns the main order object from Alpaca.
#     """

#     if qty <= 0:
#         print(f"[WARN] Not placing bracket order for {symbol}: qty={qty}")
#         return None

#     if entry_limit_price <= 0:
#         print(f"[WARN] Invalid entry_limit_price for bracket order: {entry_limit_price}")
#         return None

#     if take_profit_price <= 0 or stop_loss_price <= 0:
#         print(f"[WARN] TP or SL price invalid for bracket order: TP={take_profit_price}, SL={stop_loss_price}")
#         return None

#     order_request = BracketOrderRequest(
#         symbol=symbol,
#         qty=qty,
#         side=OrderSide.BUY,
#         time_in_force=TimeInForce.DAY,
#         extended_hours=extended,
#         limit_price=entry_limit_price,
#         take_profit={"limit_price": take_profit_price},
#         stop_loss={"stop_price": stop_loss_price},
#     )

#     order = _trading_client.submit_order(order_request)

#     if order is not None:
#         log_order_to_db(order)
#         # In case the entry leg fills immediately (paper trading often does)
#         if order.filled_at is not None:
#             log_order_event_to_db(
#                 alpaca_order_id=str(order.id),
#                 event_type="filled",
#                 filled_qty=float(order.filled_qty or 0),
#                 remaining_qty=float(order.qty) - float(order.filled_qty or 0),
#                 status=str(order.status),
#             )

#     return order


def reconcile_position(state: SymbolState, target: TargetPosition, extended: bool = False):
    """
    Given current state and target position, submit whatever order is needed
    to move from current position_qty to target_qty.

    Behavior:
      - If no change in position is needed → do nothing.
      - If moving from flat -> non-flat and target.entry_type == "limit"
        and target.entry_limit_price is provided → submit a LIMIT order.
      - Otherwise → submit a MARKET order.
      - Exits (reducing position) are currently always MARKET.
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

    # Determine if this is an ENTRY (flat -> non-flat) or an ADJUST/EXIT
    is_entry_from_flat = (abs(current) < 1e-6) and (desired > 0)

    if is_entry_from_flat and target.entry_type.lower() == "limit" and target.entry_limit_price is not None:
        print(
            f"{state.symbol}: Submitting LIMIT {side.name} qty={qty} at "
            f"{target.entry_limit_price:.2f} to reach target."
        )
        order = submit_limit_order(
            symbol=state.symbol,
            qty=qty,
            side=side,
            limit_price=target.entry_limit_price,
            extended=extended,
        )
    else:
        # For exits, adjustments, or entries without a valid limit price, use MARKET
        print(f"{state.symbol}: Submitting MARKET {side.name} qty={qty} to reach target.")
        order = submit_market_order(
            symbol=state.symbol,
            qty=qty,
            side=side,
            extended=extended,
        )

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

def ensure_exit_orders_for_symbol(state: SymbolState, extended: bool = False) -> None:
    """
    If we hold a long position in this symbol and there are no SELL exit orders
    already open, create a take-profit LIMIT order (and later, a stop-loss order).

    For now:
      - Only handles LONG positions (qty > 0).
      - Only creates a TP LIMIT SELL order, using per-symbol TP%.
      - SIgnals where we would put a SL, but leaves SL as a TODO
        until we verify the exact stop-order API in this alpaca-py version.
    """
    symbol = state.symbol
    qty = state.position_qty
    avg_entry = state.avg_entry_price

    # Only manage exits for LONG positions
    if qty <= 0 or avg_entry is None:
        return

    # If there are already SELL orders open for this symbol, assume exits exist
    open_orders = get_open_orders_for_symbol(symbol)
    for o in open_orders:
        try:
            if str(o.side).lower().endswith("sell"):
                # We already have at least one SELL order; assume TP/SL (or some exit) exists.
                print(f"{symbol}: SELL exit order(s) already present; not creating new TP/SL.")
                return
        except Exception:
            # If for some reason side isn't readable, be safe and do nothing.
            print(f"{symbol}: Could not inspect side on open order; skipping exit creation.")
            return

    # Compute TP/SL percentages for this symbol
    tp_pct = BRACKET_TP_PERCENT_BY_SYMBOL.get(symbol, DEFAULT_BRACKET_TP_PERCENT)
    sl_pct = BRACKET_SL_PERCENT_BY_SYMBOL.get(symbol, DEFAULT_BRACKET_SL_PERCENT)

    take_profit_price = avg_entry * (1.0 + tp_pct / 100.0)
    stop_loss_price = avg_entry * (1.0 - sl_pct / 100.0)

    print(
        f"{symbol}: No SELL exits found. Creating TP LIMIT at {take_profit_price:.2f} "
        f"(+{tp_pct}%) for qty={qty}. Planned SL at {stop_loss_price:.2f} (-{sl_pct}%) [TODO]."
    )

    # --- Create TP LIMIT SELL order ---
    tp_order = submit_limit_order(
        symbol=symbol,
        qty=qty,
        side=OrderSide.SELL,
        limit_price=take_profit_price,
        extended=extended,
    )

    if tp_order is not None:
        print(f"{symbol}: TP LIMIT order submitted:")
        print(tp_order)

    # --- Stop-loss order (if supported) ---
    sl_order = submit_stop_loss_order(
        symbol=symbol,
        qty=qty,
        stop_price=stop_loss_price,
        extended=extended,
    )

    if sl_order is not None:
        print(f"{symbol}: SL STOP order submitted:")
        print(sl_order)
    else:
        # This will fire if StopOrderRequest is missing or validation failed.
        print(
            f"{symbol}: SL order not placed (see warnings above). "
            f"Intended stop_price was {stop_loss_price:.2f}."
        )
