import pandas as pd
import math

from typing import Optional
from datetime import datetime, timezone

import requests
from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    GetOrdersRequest,
    # BracketOrderRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus

from daily_risk import (
    can_open_new_trade, 
    register_new_trade, 
    can_open_new_trade_for_symbol, 
    record_new_trade_for_symbol,
)

# Try to import StopOrderRequest; if not available in this alpaca-py version,
# we'll skip placing real SL orders and just log a warning.
try:
    from alpaca.trading.requests import StopOrderRequest  # type: ignore
except ImportError:
    StopOrderRequest = None  # type: ignore

from logger import log_order  # CSV order log (still available if we want it)
from data import get_latest_quote, get_recent_history, get_intraday_history
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
    MAX_ENTRY_ORDER_AGE_MINUTES,
    RISK_LIMITED_STARTING_CASH,
    LIVE_TRADING_ENABLED,
    ENTRY_ORDER_TTL_SECONDS,
)

from risk_limits import (
    build_risk_context, 
    can_open_new_position,
    build_portfolio_exposure_context, 
    can_open_new_position_with_portfolio_caps,
)

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

    This function tolerates missing quote data (e.g., when Alpaca returns a
    502/503). Instead of raising, it sets bid/ask to None and continues so the
    trading loop can make a conservative decision for the symbol.
    """
    bid_price: Optional[float] = None
    ask_price: Optional[float] = None

    try:
        quote = get_latest_quote(symbol)
        if quote is not None:
            # Some SDK versions use different attribute names
            bid_price = getattr(quote, "bid_price", getattr(quote, "bid", None))
            ask_price = getattr(quote, "ask_price", getattr(quote, "ask", None))
        else:
            print(f"[WARN] No quote available for {symbol} (Alpaca may be returning 5xx).")

    except Exception as e:
        print(f"[WARN] Failed to fetch quote for {symbol}: {e}. Continuing with no quote.")
        bid_price = None
        ask_price = None

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

    # If live trading is disabled, log and skip placing the order
    if not LIVE_TRADING_ENABLED:
        print(
            f"[SIM] MARKET order (skipped): symbol={symbol}, side={side}, qty={quantity}, extended={extended}"
        )
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

    # Alpaca does not allow MARKET orders during extended hours (pre/post). In
    # that case we convert the request into a DAY LIMIT order priced at the
    # current quote (ask for BUY, bid for SELL) to preserve the user's intent
    # without crashing the program due to API 422 errors.
    if extended:
        print(
            f"[WARN] MARKET orders not allowed in extended hours for {symbol}; "
            "converting to DAY LIMIT order using current quote."
        )
        try:
            quote = get_latest_quote(symbol)
            if quote is None:
                print(
                    f"[ERROR] Cannot place converted LIMIT for {symbol}: no quote available. Skipping order."
                )
                return None

            if side == OrderSide.BUY:
                limit_price = getattr(quote, "ask_price", getattr(quote, "ask", None))
            else:
                limit_price = getattr(quote, "bid_price", getattr(quote, "bid", None))

            if limit_price is None:
                print(
                    f"[ERROR] Cannot place converted LIMIT for {symbol}: quote missing bid/ask. Skipping order."
                )
                return None

            # Use the existing limit-order helper which already normalizes prices
            return submit_limit_order(
                symbol=symbol,
                quantity=normalized_quantity,
                side=side,
                limit_price=limit_price,
                extended=True,
            )

        except Exception as exc:
            print(f"[ERROR] Failed to convert MARKET->LIMIT for {symbol}: {exc}")
            return None

    order_request = MarketOrderRequest(
        symbol=symbol,
        qty=normalized_quantity,
        side=side,
        time_in_force=TimeInForce.DAY,
        extended_hours=extended,
    )

    try:
        order = _trading_client.submit_order(order_request)
    except APIError as api_err:
        print(f"[ERROR] Alpaca APIError during MARKET order for {symbol}: {api_err}")
        return None
    except requests.exceptions.HTTPError as http_err:
        print(f"[ERROR] HTTPError during MARKET order for {symbol}: {http_err}")
        return None
    except Exception as exc:
        print(f"[ERROR] Unexpected error submitting MARKET order for {symbol}: {exc}")
        return None

    if order is not None:
        log_order_to_db(order)

        # If the order is already filled (typical for paper market orders)
        if getattr(order, "filled_at", None) is not None:
            log_order_event_to_db(
                alpaca_order_id=str(order.id),
                event_type="filled",
                filled_qty=float(getattr(order, "filled_qty", 0) or 0),
                remaining_qty=float(getattr(order, "qty", 0)) - float(getattr(order, "filled_qty", 0) or 0),
                status=str(getattr(order, "status", "")),
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

    # If live trading is disabled, log and skip placing the order
    if not LIVE_TRADING_ENABLED:
        print(
            f"[SIM] LIMIT order (skipped): symbol={symbol}, side={side}, qty={quantity}, limit_price={limit_price}, extended={extended}"
        )
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

    try:
        order = _trading_client.submit_order(order_request)
    except APIError as api_err:
        print(f"[ERROR] Alpaca APIError during LIMIT order for {symbol}: {api_err}")
        return None
    except requests.exceptions.HTTPError as http_err:
        print(f"[ERROR] HTTPError during LIMIT order for {symbol}: {http_err}")
        return None
    except Exception as exc:
        print(f"[ERROR] Unexpected error submitting LIMIT order for {symbol}: {exc}")
        return None

    if order is not None:
        log_order_to_db(order)

        if getattr(order, "filled_at", None) is not None:
            log_order_event_to_db(
                alpaca_order_id=str(order.id),
                event_type="filled",
                filled_qty=float(getattr(order, "filled_qty", 0) or 0),
                remaining_qty=float(getattr(order, "qty", 0)) - float(getattr(order, "filled_qty", 0) or 0),
                status=str(getattr(order, "status", "")),
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

    # If live trading is disabled, log and skip placing the stop order
    if not LIVE_TRADING_ENABLED:
        print(
            f"[SIM] STOP order (skipped): symbol={symbol}, qty={quantity}, stop_price={stop_price}, extended={extended}"
        )
        return None

    if StopOrderRequest is None:
        print(
            "[WARN] alpaca.trading.requests.StopOrderRequest not available in this "
            f"alpaca-py version; skipping SL order for {symbol} at {stop_price:.4f}."
        )
        return None

    # Coerce to whole shares for safety (consistent with other submit_* helpers)
    normalized_quantity = int(quantity)
    if normalized_quantity <= 0:
        print(
            f"[WARN] Not placing STOP order for {symbol}: "
            f"computed whole-share quantity={normalized_quantity} from {quantity}"
        )
        return None

    if normalized_quantity != quantity:
        print(
            f"[INFO] Adjusting STOP order quantity for {symbol} from "
            f"{quantity:.4f} to whole shares={normalized_quantity}"
        )

    # Normalize stop price to cents (2 decimal places) to satisfy tick size rules
    normalized_stop_price = round(float(stop_price), 2)

    order_request = StopOrderRequest(
        symbol=symbol,
        qty=normalized_quantity,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
        stop_price=normalized_stop_price,
        extended_hours=extended,
    )

    try:
        order = _trading_client.submit_order(order_request)
    except Exception as exc:
        print(f"[ERROR] Failed to submit STOP order for {symbol}: {exc}")
        return None

    if order is not None:
        log_order_to_db(order)

        if getattr(order, "filled_at", None) is not None:
            log_order_event_to_db(
                alpaca_order_id=str(order.id),
                event_type="filled",
                filled_qty=float(getattr(order, "filled_qty", 0) or 0),
                remaining_qty=float(getattr(order, "qty", 0)) - float(getattr(order, "filled_qty", 0) or 0),
                status=str(getattr(order, "status", "")),
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
    # Daily risk circuit breaker for new entries (global and per-symbol)
    # --------------------------------------------------
    if is_entry_from_flat:
        # Global daily limit
        try:
            allowed, reason = can_open_new_trade()
        except Exception:
            # Some implementations may return a single bool
            allowed = bool(can_open_new_trade())
            reason = ""

        if not allowed:
            print(
                f"{state.symbol}: DAILY RISK HALT - {reason} "
                f"Not opening new position."
            )
            return

        # Per-symbol cap: avoid repeated re-entries for same symbol
        if not can_open_new_trade_for_symbol(state.symbol):
            print(
                f"[RISK] {state.symbol}: Per-symbol trade cap reached for today; "
                f"skipping new entry."
            )
            return

    order = None  # will hold whatever order we submit

    # --------------------------------------------------
    # Place the appropriate order (respect portfolio caps for new limit entries)
    # --------------------------------------------------
    if is_entry_from_flat and target.entry_type.lower() == "limit" and target.entry_limit_price is not None:
        # --- Portfolio exposure gate ---
        # Approximate order notional: quantity * entry limit price
        order_notional = float(quantity) * float(target.entry_limit_price)

        portfolio_ctx = build_portfolio_exposure_context(_trading_client)

        if not can_open_new_position_with_portfolio_caps(
            symbol=state.symbol,
            order_notional=order_notional,
            portfolio_ctx=portfolio_ctx,
        ):
            print(f"{state.symbol}: Portfolio caps blocked new entry. No order placed.")
            return

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
    # increment the daily trade count and record per-symbol usage.
    if is_entry_from_flat and order is not None:
        try:
            register_new_trade()
        except Exception:
            # ignore if register not present or fails
            pass

        try:
            record_new_trade_for_symbol(state.symbol)
        except Exception:
            # ignore if per-symbol recorder not present
            pass

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

def compute_intraday_atr_for_symbol(
    symbol: str,
    atr_period: Optional[int] = None,
) -> Optional[float]:
    """
    Compute ATR using recent INTRADAY bars for the given symbol.

    Uses the same ATR math as `compute_atr_for_symbol`, but pulls
    minute bars via get_intraday_history and applies the configured
    intraday lookback and bar size.

    Returns the latest ATR value, or None if not enough data.
    """
    from config import (
        ATR_PERIOD_DEFAULT,
        INTRADAY_LOOKBACK_MINUTES,
        INTRADAY_BAR_SIZE_MINUTES,
    )

    if atr_period is None:
        atr_period = ATR_PERIOD_DEFAULT

    intraday_df = get_intraday_history(
        symbol=symbol,
        lookback_minutes=INTRADAY_LOOKBACK_MINUTES,
        bar_size_minutes=INTRADAY_BAR_SIZE_MINUTES,
    )

    if intraday_df is None or intraday_df.empty:
        print(f"[ATR] Intraday: no intraday bars for {symbol}; cannot compute ATR.")
        return None

    if len(intraday_df) < atr_period + 1:
        print(
            f"[ATR] Intraday: not enough bars for {symbol}: have {len(intraday_df)}, need at least {atr_period + 1}.")
        return None

    # Make sure we are working with floats
    df = intraday_df.copy()
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)

    # Previous close per bar
    df["prev_close"] = df["close"].shift(1)

    # True Range components
    df["tr1"] = df["high"] - df["low"]
    df["tr2"] = (df["high"] - df["prev_close"]).abs()
    df["tr3"] = (df["low"] - df["prev_close"]).abs()

    df["true_range"] = df[["tr1", "tr2", "tr3"]].max(axis=1)

    # Simple moving average ATR
    df["atr"] = df["true_range"].rolling(
        window=atr_period,
        min_periods=atr_period,
    ).mean()

    latest_atr = float(df["atr"].iloc[-1])

    if not math.isfinite(latest_atr) or latest_atr <= 0:
        print(f"[ATR] Intraday: last ATR value invalid for {symbol}: {latest_atr}")
        return None

    print(
        f"[ATR DEBUG] Latest ATR for {symbol} over {atr_period} bars (lookback {INTRADAY_LOOKBACK_MINUTES} min, bar {INTRADAY_BAR_SIZE_MINUTES} min) is {latest_atr:.4f}"
    )

    return latest_atr


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

def cancel_stale_entry_orders_for_symbol(
    state: SymbolState,
    ttl_seconds: int = ENTRY_ORDER_TTL_SECONDS,
) -> None:
    """
    Cancel stale LIMIT BUY entry orders if:
      - We are currently FLAT in this symbol (position_qty == 0).
      - There are open BUY entry orders.
      - The order's submitted_at timestamp is older than `ttl_seconds`.

    This prevents us from chasing an entry after price moves away. Exit orders
    (SELL TP/SL) are not affected.
    """
    symbol = state.symbol

    # Only check for stale ENTRY orders if we are flat and there are open orders
    if abs(state.position_qty) > 1e-6:
        return

    if not state.has_open_orders:
        # Fast path: nothing to do when we know there are no open orders
        return

    try:
        open_orders = get_open_orders_for_symbol(symbol)
    except Exception as exc:
        print(f"[WARN] {symbol}: Could not fetch open orders for stale-check: {exc}")
        return

    if not open_orders:
        return

    now_utc = datetime.now(timezone.utc)

    for order in open_orders:
        try:
            # Ignore non BUY orders here (we do not want to cancel TP/SL SELLs)
            side_str = str(order.side).lower()
            if not side_str.endswith("buy"):
                continue

            # Prefer submitted_at (when the order was actually sent); fallback to created_at
            submitted_at = getattr(order, "submitted_at", None) or getattr(order, "created_at", None)
            if submitted_at is None:
                # No reliable timestamp; skip this order
                print(f"[WARN] {symbol}: Open BUY order {order.id} has no submitted_at/created_at; skipping TTL check.")
                continue

            # Ensure timezone-aware
            if submitted_at.tzinfo is None:
                submitted_at = submitted_at.replace(tzinfo=timezone.utc)

            age_seconds = (now_utc - submitted_at).total_seconds()

            if age_seconds > ttl_seconds:
                print(
                    f"{symbol}: Cancelling ENTRY order {order.id} (BUY) due to TTL exceeded: age={age_seconds:.1f}s > {ttl_seconds}s."
                )
                try:
                    _trading_client.cancel_order_by_id(order.id)
                except Exception as cancel_exc:
                    print(f"[WARN] Failed to cancel order {order.id}: {cancel_exc}")

                # Log cancel to DB
                try:
                    log_order_event_to_db(
                        alpaca_order_id=str(order.id),
                        event_type="canceled_stale_entry",
                        filled_qty=float(getattr(order, "filled_qty", 0) or 0),
                        remaining_qty=float(getattr(order, "qty", 0)) - float(getattr(order, "filled_qty", 0) or 0),
                        status="canceled",
                    )
                except Exception as log_exc:
                    print(f"[WARN] Failed to log stale-cancel event for order {order.id}: {log_exc}")

        except Exception as e:
            print(f"[WARN] {symbol}: Error while checking stale entry order {getattr(order, 'id', 'unknown')}: {e}")


def ensure_exit_orders_for_symbol(state: SymbolState, extended: bool = False) -> None:
    """
    If we hold a long position in this symbol and there are no SELL exit orders
    already open, create an ATR-based stop-loss (and log the intended TP).

    Behavior:
      - Only handles LONG positions (position_qty > 0).
      - Uses ATR to compute SL and TP:
            SL  = avg_entry_price - ATR_MULTIPLIER_SL_DEFAULT * ATR
            TP  = avg_entry_price + ATR_TP_MULTIPLIER_DEFAULT * ATR
      - If ATR cannot be computed for any reason, falls back to
        percentage-based exits from BRACKET_TP_PERCENT_BY_SYMBOL and
        BRACKET_SL_PERCENT_BY_SYMBOL.
      - Only a single SELL stop-loss order is submitted via submit_stop_loss_order.
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

    # Compute ATR: prefer intraday, fall back to daily
    atr_value = compute_intraday_atr_for_symbol(symbol)
    if atr_value is None:
        atr_value = compute_atr_for_symbol(symbol)
    
    if atr_value is None:
        print(f"[ATR] {symbol}: ATR (intraday and daily) not available; skipping exit order creation.")
        return

    # Use bid or ask as the current price
    last_price = state.bid if state.bid and state.bid > 0 else state.ask
    if last_price is None or last_price <= 0:
        print(f"[ATR] {symbol}: No valid bid/ask price; skipping exit order creation.")
        return

    # Compute stop-loss price only
    stop_loss_price = last_price - atr_value * ATR_MULTIPLIER_SL_DEFAULT
    print(
        f"{symbol}: Using ATR-based stop-loss. Last price={last_price:.2f}, ATR={atr_value:.2f}, "
        f"SL={stop_loss_price:.2f} (SL_mult={ATR_MULTIPLIER_SL_DEFAULT})."
    )

    # Submit only the STOP order
    print(
        f"{symbol}: Submitting SL STOP at {stop_loss_price:.2f} "
        f"for quantity={position_quantity}."
    )

    # Submit only the STOP order via the shared helper.
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