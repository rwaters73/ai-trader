import pandas as pd

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

from daily_risk import can_open_new_trade, register_new_trade

# Try to import StopOrderRequest; if not available in this alpaca-py version,
# we'll skip placing real SL orders and just log a warning.
try:
    from alpaca.trading.requests import StopOrderRequest  # type: ignore
except ImportError:
    StopOrderRequest = None  # type: ignore

from logger import log_order  # CSV order log (still available if we want it)
from data import get_latest_quote, get_recent_history
from models import SymbolState, TargetPosition
from db import log_order_to_db, log_order_event_to_db, log_risk_event_to_db

from config import (
    ALPACA_API_KEY_ID,
    ALPACA_API_SECRET_KEY,
    BRACKET_TP_PERCENT_BY_SYMBOL,
    DEFAULT_BRACKET_TP_PERCENT,
    BRACKET_SL_PERCENT_BY_SYMBOL,
    DEFAULT_BRACKET_SL_PERCENT,
    ATR_PERIOD_DEFAULT,
    ATR_MULTIPLIER_SL_DEFAULT,
    ATR_TP_MULTIPLIER_DEFAULT,
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
    Submit a market order. Returns the Alpaca order object or None
    if quantity is non-positive.

    For live trading, Alpaca requires whole-share quantities for
    non-fractionable assets. Here we floor to an integer number of
    shares before submitting.
    """
    if quantity <= 0:
        print(f"[WARN] Not placing MARKET order for {symbol}: non-positive quantity={quantity}")
        return None

    # Coerce to whole shares for safety
    normalized_quantity = int(quantity)
    if normalized_quantity <= 0:
        print(
            f"[WARN] Not placing MARKET order for {symbol}: "
            f"computed whole-share quantity={normalized_quantity} from {quantity}"
        )
        return None

    if normalized_quantity != quantity:
        print(
            f"[INFO] Adjusting MARKET order quantity for {symbol} from "
            f"{quantity:.4f} to whole shares={normalized_quantity}"
        )

    order_request = MarketOrderRequest(
        symbol=symbol,
        qty=normalized_quantity,
        side=side,
        time_in_force=TimeInForce.DAY,
        extended_hours=extended,
    )

    order = _trading_client.submit_order(order_request)

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

    We normalize limit_price to two decimal places and coerce the
    quantity to whole shares so non-fractionable assets do not error.
    """
    if quantity <= 0:
        print(f"[WARN] Not placing LIMIT order for {symbol}: non-positive quantity={quantity}")
        return None

    # Coerce to whole shares for safety
    normalized_quantity = int(quantity)
    if normalized_quantity <= 0:
        print(
            f"[WARN] Not placing LIMIT order for {symbol}: "
            f"computed whole-share quantity={normalized_quantity} from {quantity}"
        )
        return None

    if normalized_quantity != quantity:
        print(
            f"[INFO] Adjusting LIMIT order quantity for {symbol} from "
            f"{quantity:.4f} to whole shares={normalized_quantity}"
        )

    # Normalize limit price to cents (2 decimal places)
    normalized_limit_price = round(float(limit_price), 2)

    order_request = LimitOrderRequest(
        symbol=symbol,
        qty=normalized_quantity,
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
      - If no change in position is needed → do nothing.
      - If moving from flat -> non-flat and target.entry_type == "limit"
        and target.entry_limit_price is provided → submit a LIMIT order.
      - Otherwise → submit a MARKET order.
      - Exits (reducing position) are currently always MARKET.

    Now also integrates a simple daily circuit breaker via daily_risk:
      - Limits the number of *new positions* opened per calendar day.
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
    quantity = abs(delta)

    # Is this an ENTRY (flat -> non-flat, long only)?
    is_entry_from_flat = (abs(current) < 1e-6) and (desired > 0)

    # --------------------------------------------------
    # Daily risk circuit breaker for new entries
    # --------------------------------------------------
    if is_entry_from_flat:
        allowed, reason = can_open_new_trade()
        if not allowed:
            print(
                f"{state.symbol}: DAILY RISK HALT - {reason} "
                f"Not opening new position."
            )
            return

    # --------------------------------------------------
    # Place the appropriate order
    # --------------------------------------------------
    if (
        is_entry_from_flat
        and target.entry_type.lower() == "limit"
        and target.entry_limit_price is not None
    ):
        print(
            f"{state.symbol}: Submitting LIMIT {side.name} quantity={quantity} at "
            f"{target.entry_limit_price:.2f} to reach target."
        )
        order = submit_limit_order(
            symbol=state.symbol,
            quantity=quantity,
            side=side,
            limit_price=target.entry_limit_price,
            extended=extended,
        )
    else:
        # For exits, adjustments, or entries without a valid limit price, use MARKET
        print(f"{state.symbol}: Submitting MARKET {side.name} quantity={quantity} to reach target.")
        order = submit_market_order(
            symbol=state.symbol,
            quantity=quantity,
            side=side,
            extended=extended,
        )

    # If this was a new entry and the order was successfully submitted,
    # increment the daily trade count.
    if is_entry_from_flat and order is not None:
        register_new_trade()
        print(f"{state.symbol}: Registered new trade for daily risk tracking.")

    if order is not None:
        print(f"{state.symbol}: Order submitted:")
        print(order)


def compute_atr_for_symbol(
    symbol: str,
    atr_period: int = ATR_PERIOD_DEFAULT,
) -> Optional[float]:
    """
    Compute the latest ATR value for `symbol` using recent daily bars.

    Returns:
        Most recent ATR value (float) or None if ATR cannot be computed.
    """
    # Fetch slightly more than atr_period in case of missing days
    lookback_days = atr_period + 10
    daily_bars_dataframe = get_recent_history(symbol, lookback_days=lookback_days)

    if daily_bars_dataframe is None or daily_bars_dataframe.empty:
        print(f"[ATR] No daily bars available for {symbol}; cannot compute ATR.")
        return None

    # We expect columns: high, low, close
    required_columns = {"high", "low", "close"}
    if not required_columns.issubset(set(daily_bars_dataframe.columns)):
        print(
            f"[ATR] Missing required columns for {symbol}. "
            f"Expected {required_columns}, got {set(daily_bars_dataframe.columns)}"
        )
        return None

    # True Range (TR) per bar
    high_series = daily_bars_dataframe["high"].astype(float)
    low_series = daily_bars_dataframe["low"].astype(float)
    close_series = daily_bars_dataframe["close"].astype(float)

    previous_close_series = close_series.shift(1)

    range_high_low = high_series - low_series
    range_high_prev_close = (high_series - previous_close_series).abs()
    range_low_prev_close = (low_series - previous_close_series).abs()

    true_range_series = pd.concat(
        [range_high_low, range_high_prev_close, range_low_prev_close],
        axis=1,
    ).max(axis=1)

    # Simple moving average of TR for ATR
    atr_series = true_range_series.rolling(window=atr_period).mean()

    latest_atr = atr_series.iloc[-1]

    if pd.isna(latest_atr):
        print(f"[ATR] Computed ATR for {symbol} is NaN; not using ATR.")
        return None

    return float(latest_atr)


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
    already open, create ATR-based take-profit and stop-loss exits.

    Behavior:
      - Only handles LONG positions (position_qty > 0).
      - Uses ATR to compute SL and TP:
            SL  = avg_entry_price - ATR_SL_MULTIPLIER_DEFAULT * ATR
            TP  = avg_entry_price + ATR_TP_MULTIPLIER_DEFAULT * ATR
      - If ATR cannot be computed for any reason, falls back to
        percentage-based exits from BRACKET_TP_PERCENT_BY_SYMBOL and
        BRACKET_SL_PERCENT_BY_SYMBOL.
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
            # If for some reason side is not readable, be safe and do nothing.
            print(f"{symbol}: Could not inspect side on open order; skipping exit creation.")
            return

    # Try to compute ATR-based SL/TP
    atr_value = compute_atr_for_symbol(symbol)

    if atr_value is not None:
        take_profit_price = average_entry_price + ATR_TP_MULTIPLIER_DEFAULT * atr_value
        stop_loss_price = average_entry_price - ATR_MULTIPLIER_SL_DEFAULT * atr_value
        print(
            f"{symbol}: Using ATR-based exits. ATR={atr_value:.2f}, "
            f"TP={take_profit_price:.2f}, SL={stop_loss_price:.2f} "
            f"(TP_mult={ATR_TP_MULTIPLIER_DEFAULT}, SL_mult={ATR_MULTIPLIER_SL_DEFAULT})."
        )
    else:
        # Fallback: percentage-based exits from config
        take_profit_percent = BRACKET_TP_PERCENT_BY_SYMBOL.get(
            symbol,
            DEFAULT_BRACKET_TP_PERCENT,
        )
        stop_loss_percent = BRACKET_SL_PERCENT_BY_SYMBOL.get(
            symbol,
            DEFAULT_BRACKET_SL_PERCENT,
        )

        take_profit_price = average_entry_price * (1.0 + take_profit_percent / 100.0)
        stop_loss_price = average_entry_price * (1.0 - stop_loss_percent / 100.0)

        print(
            f"{symbol}: ATR not available. Falling back to percent-based exits. "
            f"TP={take_profit_price:.2f} (+{take_profit_percent}%), "
            f"SL={stop_loss_price:.2f} (-{stop_loss_percent}%)."
        )

    print(
        f"{symbol}: Creating TP LIMIT at {take_profit_price:.2f} "
        f"and SL STOP at {stop_loss_price:.2f} for quantity={position_quantity}."
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
