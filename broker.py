from typing import Optional
from datetime import datetime

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    GetOrdersRequest,
    # BracketOrderRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus

# Try to import StopOrderRequest; if not available in this alpaca-py version,
# we'll skip placing real SL orders and just log a warning.
try:
    from alpaca.trading.requests import StopOrderRequest  # type: ignore
except ImportError:
    StopOrderRequest = None  # type: ignore

from logger import log_order  # CSV order log (still available if we want it)
from data import get_latest_quote
from models import SymbolState, TargetPosition
from db import log_order_to_db, log_order_event_to_db, log_risk_event_to_db
from config import (
    ALPACA_API_KEY_ID,
    ALPACA_API_SECRET_KEY,
    BRACKET_TP_PERCENT_BY_SYMBOL,
    DEFAULT_BRACKET_TP_PERCENT,
    BRACKET_SL_PERCENT_BY_SYMBOL,
    DEFAULT_BRACKET_SL_PERCENT,
)

from risk_limits import build_risk_context, can_open_new_position

# One shared trading client for the process
_trading_client = TradingClient(
    ALPACA_API_KEY_ID,
    ALPACA_API_SECRET_KEY,
    paper=True,
)


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def build_symbol_state(symbol: str) -> SymbolState:
    """
    Gather quote, position, and open-order info into a SymbolState snapshot.
    """
    quote = get_latest_quote(symbol)
    bid_price = quote.bid_price
    ask_price = quote.ask_price

    position_quantity, average_entry_price = get_position_info(symbol)
    open_orders_exist = has_open_orders(symbol)

    print(f"\n=== {datetime.now().isoformat(timespec='seconds')} | Symbol: {symbol} ===")
    print(f"Bid: {bid_price}, Ask: {ask_price}")
    print(f"Current open quantity in {symbol}: {position_quantity}")
    if average_entry_price is not None:
        print(f"Average entry price: {average_entry_price}")
    print(f"Has open orders in {symbol}: {open_orders_exist}")

    return SymbolState(
        symbol=symbol,
        bid=bid_price,
        ask=ask_price,
        position_qty=position_quantity,
        avg_entry_price=average_entry_price,
        has_open_orders=open_orders_exist,
    )


def get_position_info(symbol: str) -> tuple[float, Optional[float]]:
    """
    Return (position_quantity, average_entry_price) for the symbol in the paper account.
    If no open position, return (0.0, None).
    """
    positions = _trading_client.get_all_positions()

    for position in positions:
        if position.symbol == symbol:
            position_quantity = float(position.qty)  # fractional support
            average_price = float(position.avg_entry_price)
            return position_quantity, average_price

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


# ---------------------------------------------------------------------------
# Order submission helpers
# ---------------------------------------------------------------------------

def submit_market_order(
    symbol: str,
    quantity: float,
    side: OrderSide,
    extended: bool = False,
):
    """
    Submit a MARKET order. Returns the Alpaca order object or None
    if quantity is non-positive.
    """
    if quantity <= 0:
        print(f"[WARN] Not placing MARKET order for {symbol}: non-positive quantity={quantity}")
        return None

    order_request = MarketOrderRequest(
        symbol=symbol,
        qty=quantity,  # Alpaca field name is 'qty'
        side=side,
        time_in_force=TimeInForce.DAY,
        extended_hours=extended,
    )

    order = _trading_client.submit_order(order_request)

    # Log order submission to DB
    if order is not None:
        log_order_to_db(order)

        # If the order is already filled (typical for paper market orders)
        if order.filled_at is not None:
            log_order_event_to_db(
                alpaca_order_id=str(order.id),
                event_type="filled",
                filled_qty=float(order.filled_qty or 0),
                remaining_qty=float(order.qty) - float(order.filled_qty or 0),
                status=str(order.status),
            )

    return order


def submit_limit_order(
    symbol: str,
    quantity: float,
    side: OrderSide,
    limit_price: float,
    extended: bool = False,
):
    """
    Submit a LIMIT order. Returns the Alpaca order object or None
    if quantity is non-positive.

    We normalize limit_price to two decimal places to satisfy US equity
    tick-size rules (no sub-penny prices).
    """
    if quantity <= 0:
        print(f"[WARN] Not placing LIMIT order for {symbol}: non-positive quantity={quantity}")
        return None

    # Normalize limit price to cents (2 decimal places)
    normalized_limit_price = round(float(limit_price), 2)

    order_request = LimitOrderRequest(
        symbol=symbol,
        qty=quantity,  # Alpaca field name is 'qty'
        side=side,
        limit_price=normalized_limit_price,
        time_in_force=TimeInForce.DAY,
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


def submit_stop_loss_order(
    symbol: str,
    quantity: float,
    stop_price: float,
    extended: bool = False,
):
    """
    Submit a SELL stop-loss order for an existing LONG position.

    This assumes:
      - We are long the symbol (quantity > 0).
      - We want a STOP-MARKET SELL at stop_price.

    If StopOrderRequest is not available in this alpaca-py version, this
    function will log a warning and do nothing.
    """
    if quantity <= 0:
        print(f"[WARN] Not placing STOP order for {symbol}: non-positive quantity={quantity}")
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
        qty=quantity,  # Alpaca field name is 'qty'
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


# ---------------------------------------------------------------------------
# Position reconciliation and flatten helpers
# ---------------------------------------------------------------------------

def reconcile_position(state: SymbolState, target: TargetPosition, extended: bool = False):
    """
    Given current state and target position, submit whatever order is needed
    to move from current position_qty to target_qty.

    Behavior:
      - If no change in position is needed -> do nothing.
      - If moving from flat -> non-flat and target.entry_type == "limit"
        and target.entry_limit_price is provided -> submit a LIMIT order.
      - Otherwise -> submit a MARKET order.
      - Exits (reducing position) are currently always MARKET.

    New: risk_limits integration
      - For BUY orders that increase net long exposure, check risk_limits
        before submitting the order. If the order would violate the
        risk-limited account rules, we log a message and skip the order.
    """
    current_quantity = state.position_qty
    desired_quantity = target.target_qty
    quantity_delta = desired_quantity - current_quantity

    print(f"{state.symbol}: current={current_quantity}, target={desired_quantity} | {target.reason}")

    # No change needed
    if abs(quantity_delta) < 1e-6:
        print(f"{state.symbol}: No change in position. No order placed.")
        return

    order_side = OrderSide.BUY if quantity_delta > 0 else OrderSide.SELL
    order_quantity = abs(quantity_delta)

    # Determine if this is an ENTRY (flat -> non-flat)
    is_entry_from_flat = (abs(current_quantity) < 1e-6) and (desired_quantity > 0)

    # ------------------------------------------------------------------
    # Risk check for new BUY exposure
    # ------------------------------------------------------------------
    is_new_long_exposure = is_entry_from_flat and (order_side is OrderSide.BUY)

    if is_new_long_exposure:
        # Estimate the entry price so we can estimate order cost.
        # For a limit entry, use the proposed limit price.
        # For a market entry, use the current ask, or bid as a fallback.
        if target.entry_type.lower() == "limit" and target.entry_limit_price is not None:
            estimated_price = float(target.entry_limit_price)
        else:
            if state.ask is not None and state.ask > 0:
                estimated_price = float(state.ask)
            elif state.bid is not None and state.bid > 0:
                estimated_price = float(state.bid)
            else:
                estimated_price = 0.0

        if estimated_price <= 0:
            print(
                f"[RISK] {state.symbol}: Could not estimate a positive entry price "
                f"for risk check. Skipping risk_limits check and not placing order."
            )
            return

        estimated_order_cost = order_quantity * estimated_price

        # Build risk context and ask whether we are allowed to open this position.
        risk_context = build_risk_context(_trading_client)
        risk_decision = can_open_new_position(risk_context, estimated_order_cost)

        # risk_decision might be a bool or (bool, message)
        if isinstance(risk_decision, tuple):
            is_allowed = bool(risk_decision[0])
            risk_message = str(risk_decision[1]) if len(risk_decision) > 1 else ""
        else:
            is_allowed = bool(risk_decision)
            risk_message = ""

        if not is_allowed:
            detail = f" Reason: {risk_message}" if risk_message else ""
            print(
                f"[RISK] {state.symbol}: Blocked new BUY of approx cost "
                f"{estimated_order_cost:.2f} by risk_limits.{detail}"
            )

            # NEW: Log blocked decision
            log_risk_event_to_db(
                symbol=state.symbol,
                action="BLOCK_BUY",
                cost=estimated_order_cost,
                allowed=False,
                message=risk_message
            )

            return
        
        print(
            f"[RISK] {state.symbol}: New BUY of approx cost {estimated_order_cost:.2f} "
            f"approved by risk_limits."
        )
        # NEW: Log approval
        log_risk_event_to_db(
            symbol=state.symbol,
            action="ALLOW_BUY",
            cost=estimated_order_cost,
            allowed=True,
            message=risk_message
        )


    # ------------------------------------------------------------------
    # Submit the actual order
    # ------------------------------------------------------------------
    if (
        is_entry_from_flat
        and target.entry_type.lower() == "limit"
        and target.entry_limit_price is not None
    ):
        print(
            f"{state.symbol}: Submitting LIMIT {order_side.name} quantity={order_quantity} at "
            f"{target.entry_limit_price:.2f} to reach target."
        )
        order = submit_limit_order(
            symbol=state.symbol,
            quantity=order_quantity,
            side=order_side,
            limit_price=target.entry_limit_price,
            extended=extended,
        )
    else:
        # For exits, adjustments, or entries without a valid limit price, use MARKET
        print(f"{state.symbol}: Submitting MARKET {order_side.name} quantity={order_quantity} to reach target.")
        order = submit_market_order(
            symbol=state.symbol,
            quantity=order_quantity,
            side=order_side,
            extended=extended,
        )

    if order is not None:
        print(f"{state.symbol}: Order submitted:")
        print(order)


def flatten_symbol(symbol: str):
    """
    Flatten a single symbol (close long or cover short).
    """
    position_quantity, _ = get_position_info(symbol)

    if abs(position_quantity) < 1e-6:
        print(f"{symbol}: Already flat. No flatten order placed.")
        return

    order_side = OrderSide.SELL if position_quantity > 0 else OrderSide.BUY
    print(f"{symbol}: Flattening position quantity={position_quantity} with {order_side.name} order.")
    submit_market_order(
        symbol=symbol,
        quantity=abs(position_quantity),
        side=order_side,
        extended=False,
    )


def flatten_all(symbol_list: list[str]):
    """
    Flatten positions for all given symbols.
    """
    for symbol in symbol_list:
        flatten_symbol(symbol)


# ---------------------------------------------------------------------------
# Open-order management + automatic TP/SL creation
# ---------------------------------------------------------------------------

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
        except Exception as exception:
            print(f"[WARN] Failed to cancel order {order.id}: {exception}")


def ensure_exit_orders_for_symbol(state: SymbolState, extended: bool = False) -> None:
    """
    If we hold a long position in this symbol and there are no SELL exit orders
    already open, create a take-profit LIMIT order (and a stop-loss order if supported).

    For now:
      - Only handles LONG positions (position_qty > 0).
      - Creates a TP LIMIT SELL order, using per-symbol TP%.
      - Attempts a SL STOP order, but may skip if StopOrderRequest is unavailable.
    """
    symbol = state.symbol
    position_quantity = state.position_qty
    average_entry_price = state.avg_entry_price

    # Only manage exits for LONG positions
    if position_quantity <= 0 or average_entry_price is None:
        return

    # If there are already SELL orders open for this symbol, assume exits exist
    open_orders = get_open_orders_for_symbol(symbol)
    for open_order in open_orders:
        try:
            if str(open_order.side).lower().endswith("sell"):
                # We already have at least one SELL order; assume TP/SL (or some exit) exists.
                print(f"{symbol}: SELL exit order(s) already present; not creating new TP/SL.")
                return
        except Exception:
            # If for some reason side isn't readable, be safe and do nothing.
            print(f"{symbol}: Could not inspect side on open order; skipping exit creation.")
            return

    # Compute TP/SL percentages for this symbol
    take_profit_percent = BRACKET_TP_PERCENT_BY_SYMBOL.get(symbol, DEFAULT_BRACKET_TP_PERCENT)
    stop_loss_percent = BRACKET_SL_PERCENT_BY_SYMBOL.get(symbol, DEFAULT_BRACKET_SL_PERCENT)

    raw_take_profit_price = average_entry_price * (1.0 + take_profit_percent / 100.0)
    take_profit_price = round(raw_take_profit_price, 2)

    stop_loss_price = average_entry_price * (1.0 - stop_loss_percent / 100.0)

    print(
        f"{symbol}: No SELL exits found. Creating TP LIMIT at {take_profit_price:.2f} "
        f"(+{take_profit_percent}%) for quantity={position_quantity}. "
        f"Planned SL at {stop_loss_price:.2f} (-{stop_loss_percent}%) [TODO if unsupported]."
    )

    # --- Create TP LIMIT SELL order ---
    take_profit_order = submit_limit_order(
        symbol=symbol,
        quantity=position_quantity,
        side=OrderSide.SELL,
        limit_price=take_profit_price,
        extended=extended,
    )

    if take_profit_order is not None:
        print(f"{symbol}: TP LIMIT order submitted:")
        print(take_profit_order)

    # --- Stop-loss order (if supported) ---
    stop_loss_order = submit_stop_loss_order(
        symbol=symbol,
        quantity=position_quantity,
        stop_price=stop_loss_price,
        extended=extended,
    )

    if stop_loss_order is not None:
        print(f"{symbol}: SL STOP order submitted:")
        print(stop_loss_order)
    else:
        # This will fire if StopOrderRequest is missing or validation failed.
        print(
            f"{symbol}: SL order not placed (see warnings above). "
            f"Intended stop_price was {stop_loss_price:.2f}."
        )
