import pandas as pd
import math

from typing import Optional, Tuple
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
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus, OrderType

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
    ATR_STOP_MULTIPLIER_DEFAULT,
    ATR_TP_MULTIPLIER_DEFAULT,
    MAX_ENTRY_ORDER_AGE_MINUTES,
    RISK_LIMITED_STARTING_CASH,
    LIVE_TRADING_ENABLED,
    ENTRY_ORDER_TTL_SECONDS,
)

# Maximum spread percentage allowed during extended hours before skipping entry
MAX_SPREAD_PCT_EXTENDED = 2.0  # 2% max spread during extended hours

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
# Entry order lifecycle tracking (in-memory, per symbol per trading day)
# ---------------------------------------------------------------------------
_entry_order_tracking: dict[str, dict] = {}
# Structure: {symbol: {"date": "YYYY-MM-DD", "submitted": True, "filled": False}}

def _get_trading_date() -> str:
    """Return current trading date as YYYY-MM-DD string (UTC-based for simplicity)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

def can_submit_entry_for_symbol(symbol: str) -> tuple[bool, str]:
    """Check if we can submit a new entry order for this symbol today.
    
    Returns (allowed, reason).
    Rules:
      - If no entry submitted today: allowed.
      - If entry submitted but not filled: blocked (one-entry-per-day guard).
      - If entry submitted and filled: allowed (prior entry completed).
    """
    today = _get_trading_date()
    if symbol not in _entry_order_tracking:
        return True, "No prior entry today"
    
    record = _entry_order_tracking[symbol]
    if record.get("date") != today:
        # Stale record from previous day; reset
        _entry_order_tracking[symbol] = {"date": today, "submitted": False, "filled": False}
        return True, "Prior entry from different day; reset"
    
    if record.get("submitted") and not record.get("filled"):
        return False, "Entry order already submitted today and not yet filled (one-entry-per-day guard)"
    
    if record.get("filled"):
        # Prior entry filled; allow another entry
        return True, "Prior entry filled; allowing new entry"
    
    return True, "No blocking condition"

def record_entry_submission(symbol: str) -> None:
    """Record that we submitted an entry order for this symbol today."""
    today = _get_trading_date()
    _entry_order_tracking[symbol] = {"date": today, "submitted": True, "filled": False}
    print(f"[ENTRY_GUARD] {symbol}: recorded entry submission for {today}")

def record_entry_fill(symbol: str) -> None:
    """Record that an entry order for this symbol filled (allows new entries)."""
    today = _get_trading_date()
    if symbol in _entry_order_tracking and _entry_order_tracking[symbol].get("date") == today:
        _entry_order_tracking[symbol]["filled"] = True
        print(f"[ENTRY_GUARD] {symbol}: recorded entry fill for {today}")


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

from typing import Optional, Tuple

def _safe_int_qty(q) -> int:
    """
    Alpaca often returns qty as a string. Coerce to int safely.
    """
    try:
        if q is None:
            return 0
        return int(float(q))
    except Exception:
        return 0

def _is_sell_order(order_obj) -> bool:
    try:
        return str(getattr(order_obj, "side", "")).lower().endswith("sell")
    except Exception:
        return False

def _classify_exit_orders(open_orders) -> Tuple[Optional[object], Optional[object]]:
    """
    Return (tp_limit_order, sl_stop_order) from a list of open orders.
    We classify by order_type + presence of limit_price/stop_price.
    """
    tp = None
    sl = None

    for o in open_orders:
        if not _is_sell_order(o):
            continue

        try:
            otype = str(getattr(o, "order_type", "")).lower()
        except Exception:
            otype = ""

        limit_price = getattr(o, "limit_price", None)
        stop_price = getattr(o, "stop_price", None)

        # TP: a SELL limit order
        if tp is None and ("limit" in otype) and (limit_price is not None):
            tp = o
            continue

        # SL: a SELL stop order (stop market)
        if sl is None and ("stop" in otype) and (stop_price is not None):
            sl = o
            continue

    return tp, sl

def _cancel_order_by_id(order_id: str) -> bool:
    """
    Cancel using alpaca-py TradingClient. Returns True if request was sent.
    """
    try:
        _trading_client.cancel_order_by_id(order_id)
        return True
    except Exception as exc:
        print(f"[WARN] Failed to cancel order {order_id}: {exc}")
        return False

def _recreate_tp_sl_orders_if_needed(
    symbol: str,
    desired_tp_qty: int,
    desired_sl_qty: int,
    tp_price: float,
    sl_price: float,
    extended: bool,
) -> None:
    """
    Ensure TP and SL exist with the desired quantities.
    If an order exists with the wrong qty, cancel and recreate.
    """
    open_orders = get_open_orders_for_symbol(symbol)
    tp_order, sl_order = _classify_exit_orders(open_orders)

    # --- TP order ---
    if desired_tp_qty > 0:
        if tp_order is None:
            print(f"{symbol}: Watcher creating missing TP LIMIT qty={desired_tp_qty} at {tp_price:.2f}")
            submit_limit_order(
                symbol=symbol,
                quantity=desired_tp_qty,
                side=OrderSide.SELL,
                limit_price=tp_price,
                extended=extended,
            )
        else:
            existing_tp_qty = _safe_int_qty(getattr(tp_order, "qty", None))
            if existing_tp_qty != desired_tp_qty:
                print(
                    f"{symbol}: Watcher TP qty mismatch: existing={existing_tp_qty}, desired={desired_tp_qty}. "
                    f"Cancelling/recreating TP."
                )
                _cancel_order_by_id(str(getattr(tp_order, "id", "")))
                submit_limit_order(
                    symbol=symbol,
                    quantity=desired_tp_qty,
                    side=OrderSide.SELL,
                    limit_price=tp_price,
                    extended=extended,
                )
    else:
        # desired_tp_qty == 0, cancel any existing TP
        if tp_order is not None:
            print(f"{symbol}: Watcher cancelling TP because desired_tp_qty=0")
            _cancel_order_by_id(str(getattr(tp_order, "id", "")))

    # --- SL order ---
    if desired_sl_qty > 0:
        if sl_order is None:
            print(f"{symbol}: Watcher creating missing SL STOP qty={desired_sl_qty} at {sl_price:.2f}")
            submit_stop_loss_order(
                symbol=symbol,
                quantity=desired_sl_qty,
                stop_price=sl_price,
                extended=extended,
            )
        else:
            existing_sl_qty = _safe_int_qty(getattr(sl_order, "qty", None))
            if existing_sl_qty != desired_sl_qty:
                print(
                    f"{symbol}: Watcher SL qty mismatch: existing={existing_sl_qty}, desired={desired_sl_qty}. "
                    f"Cancelling/recreating SL."
                )
                _cancel_order_by_id(str(getattr(sl_order, "id", "")))
                submit_stop_loss_order(
                    symbol=symbol,
                    quantity=desired_sl_qty,
                    stop_price=sl_price,
                    extended=extended,
                )
    else:
        # desired_sl_qty == 0, cancel any existing SL
        if sl_order is not None:
            print(f"{symbol}: Watcher cancelling SL because desired_sl_qty=0")
            _cancel_order_by_id(str(getattr(sl_order, "id", "")))

def watch_and_sync_exit_orders(state: SymbolState, extended: bool = False) -> None:
    """
    Lightweight order-event watcher (polling-based).

    Each cycle, it:
      - Reads current position quantity
      - Computes intended TP/SL split (33% TP, remainder SL), whole shares only
      - Computes ATR-based TP/SL prices (intraday preferred, fallback daily)
      - Cancels/recreates TP/SL orders if missing or quantities are wrong
      - Cancels all SELL exits if position is flat
    """
    symbol = state.symbol
    pos_qty_raw = state.position_qty
    avg_entry = state.avg_entry_price

    # If flat, cancel any stray SELL exits
    if pos_qty_raw <= 0 or avg_entry is None:
        open_orders = get_open_orders_for_symbol(symbol)
        for o in open_orders:
            if _is_sell_order(o):
                oid = str(getattr(o, "id", ""))
                if oid:
                    print(f"{symbol}: Watcher cancelling SELL exit because position is flat. order_id={oid}")
                    _cancel_order_by_id(oid)
        return

    # Whole shares only
    pos_qty = int(pos_qty_raw)
    if pos_qty <= 0:
        return

    # ATR
    atr_value = compute_intraday_atr_for_symbol(symbol)
    if atr_value is None:
        atr_value = compute_atr_for_symbol(symbol)
    if atr_value is None:
        print(f"[ATR] {symbol}: Watcher cannot compute ATR; skipping sync.")
        return

    # Use current market price (bid preferred)
    last_price = state.bid if state.bid and state.bid > 0 else state.ask
    if last_price is None or last_price <= 0:
        print(f"[ATR] {symbol}: Watcher no valid bid/ask; skipping sync.")
        return

    tp_price = round(float(last_price + atr_value * ATR_TP_MULTIPLIER_DEFAULT), 2)
    sl_price = round(float(last_price - atr_value * ATR_STOP_MULTIPLIER_DEFAULT), 2)

    if sl_price <= 0:
        print(f"[ATR] {symbol}: Watcher computed SL<=0; skipping sync.")
        return

    # Quantity split: 33% TP, remainder SL; no fractional shares.
    tp_qty = int(pos_qty * 0.33)
    if pos_qty >= 2 and tp_qty < 1:
        tp_qty = 1

    sl_qty = pos_qty - tp_qty

    # If only 1 share, protect it with SL; TP disabled
    if pos_qty == 1:
        tp_qty = 0
        sl_qty = 1

    if sl_qty <= 0:
        sl_qty = pos_qty
        tp_qty = 0

    print(
        f"{symbol}: Watcher sync exits. pos={pos_qty}, TP_qty={tp_qty}, SL_qty={sl_qty}, "
        f"TP={tp_price:.2f}, SL={sl_price:.2f}"
    )

    _recreate_tp_sl_orders_if_needed(
        symbol=symbol,
        desired_tp_qty=tp_qty,
        desired_sl_qty=sl_qty,
        tp_price=tp_price,
        sl_price=sl_price,
        extended=extended,
    )

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
    # Entry order lifecycle management (before risk checks)
    # --------------------------------------------------
    if is_entry_from_flat:
        # 1) Quote validation: ensure bid/ask are valid
        if state.ask is None or state.ask <= 0:
            print(f"{state.symbol}: [ENTRY_GUARD] Invalid ask price ({state.ask}); skipping entry.")
            return
        
        # 2) Spread check during extended hours
        if extended and state.bid and state.bid > 0:
            spread_pct = 100.0 * (state.ask - state.bid) / state.bid
            if spread_pct > MAX_SPREAD_PCT_EXTENDED:
                print(
                    f"{state.symbol}: [ENTRY_GUARD] Spread too wide during extended hours: "
                    f"{spread_pct:.2f}% > {MAX_SPREAD_PCT_EXTENDED}%; skipping entry."
                )
                return
        
        # 3) Cancel any stale entry orders (TTL enforcement)
        try:
            canceled_count = cancel_stale_entry_buy_limits_for_symbol(
                symbol=state.symbol,
                ttl_seconds=ENTRY_ORDER_TTL_SECONDS,
            )
            if canceled_count > 0:
                print(f"{state.symbol}: [ENTRY_GUARD] Canceled {canceled_count} stale entry order(s).")
        except Exception as e:
            print(f"{state.symbol}: [ENTRY_GUARD] TTL cancel failed: {e}")
        
        # 4) One-entry-per-day guard: block if prior entry not filled
        allowed, reason = can_submit_entry_for_symbol(state.symbol)
        if not allowed:
            print(f"{state.symbol}: [ENTRY_GUARD] {reason}; skipping entry.")
            return

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

        # Record entry submission for one-entry-per-day guard
        record_entry_submission(state.symbol)

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

def _parse_order_time(order) -> Optional[datetime]:
    """
    Alpaca order timestamps may be datetime or ISO strings depending on SDK/version.
    Prefer submitted_at, else created_at.
    """
    ts = getattr(order, "submitted_at", None) or getattr(order, "created_at", None)
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    # string fallback
    try:
        # alpaca tends to use ISO-8601
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except Exception:
        return None


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def cancel_stale_entry_orders_for_symbol(
    symbol: str,
    ttl_seconds: int,
    replace: bool,
    replacement_qty: Optional[float],
    replacement_limit_price: Optional[float],
    extended: bool,
) -> Tuple[bool, str]:
    """
    TTL cancel + (optional) replace for ENTRY orders.

    We define "entry order" as:
      - side == BUY
      - type == LIMIT
      - status == OPEN

    Returns:
      (did_action, reason)
    """
    # Fetch open orders for symbol
    request_params = GetOrdersRequest(
        status=QueryOrderStatus.OPEN,
        symbols=[symbol],
        nested=False,
        limit=50,
    )
    open_orders = _trading_client.get_orders(filter=request_params)

    if not open_orders:
        return False, "No open orders."

    now = datetime.now(timezone.utc)

    # Find open BUY LIMIT orders (entry candidates)
    entry_orders = []
    for o in open_orders:
        side = getattr(o, "side", None)
        otype = getattr(o, "order_type", None) or getattr(o, "type", None)

        if side == OrderSide.BUY and otype in (OrderType.LIMIT, "limit"):
            entry_orders.append(o)

    if not entry_orders:
        return False, "No open BUY LIMIT orders."

    # Evaluate staleness
    stale_orders = []
    for o in entry_orders:
        ts = _parse_order_time(o)
        if ts is None:
            continue
        age_seconds = (now - ts).total_seconds()
        if age_seconds >= float(ttl_seconds):
            stale_orders.append((o, age_seconds))

    if not stale_orders:
        return False, "No stale entry orders."

    # Cancel all stale entry orders (usually you will have 1)
    for o, age_seconds in stale_orders:
        oid = getattr(o, "id", None)
        try:
            _trading_client.cancel_order_by_id(oid)
            print(
                f"[TTL] Cancelled stale ENTRY order for {symbol}: "
                f"id={oid}, age={age_seconds:.0f}s"
            )
        except Exception as e:
            return True, f"Attempted cancel but failed: {e}"

    # Optional replace
    if not replace:
        return True, "Cancelled stale entry order(s); replace disabled."

    if replacement_qty is None or replacement_qty <= 0:
        return True, "Cancelled stale entry order(s); not replacing because qty invalid."

    if replacement_limit_price is None or replacement_limit_price <= 0:
        return True, "Cancelled stale entry order(s); not replacing because limit_price invalid."

    # Whole-share safety (non-fractionable assets)
    replacement_qty_int = int(replacement_qty)
    if replacement_qty_int <= 0:
        return True, "Cancelled stale entry order(s); not replacing because qty floors to 0."

    # Round price to 2 decimals to avoid sub-penny rejection
    replacement_limit_price_rounded = round(float(replacement_limit_price), 2)

    order = submit_limit_order(
        symbol=symbol,
        quantity=replacement_qty_int,
        limit_price=replacement_limit_price_rounded,
        side="buy",
        extended=extended,
    )

    if order is None:
        return True, "Cancelled stale entry order(s); replacement submit returned None."

    return True, (
        f"Cancelled stale entry order(s) and replaced with BUY LIMIT "
        f"qty={replacement_qty_int} @ {replacement_limit_price_rounded}."
    )

def ensure_exit_orders_for_symbol(state: SymbolState, extended: bool = False) -> None:
    """
    If we hold a long position in this symbol, ensure we have:
      - A partial take-profit LIMIT sell for 33% of shares
      - A stop-loss STOP sell for the remaining shares

    This is a safe non-OCO approximation that avoids "insufficient qty available"
    by splitting the position quantity across the two exit orders.

    Notes:
      - No fractional shares: quantities are coerced to whole shares.
      - ATR is preferred intraday, with a fallback to daily ATR.
      - TP/SL prices are computed off last_price (bid preferred, else ask).
    """
    symbol = state.symbol
    position_qty_raw = state.position_qty
    avg_entry = state.avg_entry_price

    # Only manage exits for LONG positions
    if position_qty_raw <= 0 or avg_entry is None:
        return

    # Whole shares only
    position_qty = int(position_qty_raw)
    if position_qty <= 0:
        print(f"{symbol}: Position qty rounds to 0 shares; skipping exit order creation.")
        return

    # Compute ATR: prefer intraday, fall back to daily
    atr_value = compute_intraday_atr_for_symbol(symbol)
    if atr_value is None:
        atr_value = compute_atr_for_symbol(symbol)

    if atr_value is None:
        print(f"[ATR] {symbol}: ATR (intraday and daily) not available; skipping exit creation.")
        return

    # Use bid or ask as the current price
    last_price = state.bid if state.bid and state.bid > 0 else state.ask
    if last_price is None or last_price <= 0:
        print(f"[ATR] {symbol}: No valid bid/ask price; skipping exit creation.")
        return

    # Compute TP/SL prices from last_price
    tp_price = last_price + (atr_value * ATR_TP_MULTIPLIER_DEFAULT)
    sl_price = last_price - (atr_value * ATR_STOP_MULTIPLIER_DEFAULT)

    if sl_price <= 0:
        print(f"[ATR] {symbol}: Computed SL <= 0 (SL={sl_price:.4f}); skipping exit creation.")
        return

    # Normalize to cents (submit_* also rounds, but doing it here keeps logs clean)
    tp_price = round(float(tp_price), 2)
    sl_price = round(float(sl_price), 2)

    # ------------------------------------------------------
    # Split quantities to avoid "insufficient qty available"
    # ------------------------------------------------------
    # Partial TP: 33% of shares, at least 1 share if position >= 2
    tp_qty = int(position_qty * 0.33)
    if position_qty >= 2 and tp_qty < 1:
        tp_qty = 1

    # Remaining shares go to SL
    sl_qty = position_qty - tp_qty

    # If position is 1 share, prioritize stop-loss protection
    if position_qty == 1:
        tp_qty = 0
        sl_qty = 1

    if sl_qty <= 0:
        # Safety: ensure SL always protects something
        sl_qty = position_qty
        tp_qty = 0

    print(
        f"{symbol}: Using ATR exits. Last={last_price:.2f}, ATR={atr_value:.2f}, "
        f"TP={tp_price:.2f} (TP_mult={ATR_TP_MULTIPLIER_DEFAULT}), "
        f"SL={sl_price:.2f} (SL_mult={ATR_STOP_MULTIPLIER_DEFAULT}). "
        f"Split qty: TP={tp_qty}, SL={sl_qty}, pos={position_qty}."
    )

    # ------------------------------------------------------
    # Inspect existing open SELL orders so we don't duplicate
    # ------------------------------------------------------
    open_orders = get_open_orders_for_symbol(symbol)

    def _is_sell(order_obj) -> bool:
        try:
            return str(getattr(order_obj, "side", "")).lower().endswith("sell")
        except Exception:
            return False

    def _is_tp_limit(order_obj) -> bool:
        try:
            otype = str(getattr(order_obj, "order_type", "")).lower()
            lim = getattr(order_obj, "limit_price", None)
            return ("limit" in otype) and (lim is not None)
        except Exception:
            return False

    def _is_sl_stop(order_obj) -> bool:
        try:
            otype = str(getattr(order_obj, "order_type", "")).lower()
            stop = getattr(order_obj, "stop_price", None)
            return ("stop" in otype) and (stop is not None)
        except Exception:
            return False

    has_existing_tp = any(_is_sell(o) and _is_tp_limit(o) for o in open_orders)
    has_existing_sl = any(_is_sell(o) and _is_sl_stop(o) for o in open_orders)

    if has_existing_tp:
        print(f"{symbol}: Existing TP LIMIT sell already present; not creating a new TP.")
    if has_existing_sl:
        print(f"{symbol}: Existing SL STOP sell already present; not creating a new SL.")

    # ------------------------------------------------------
    # Create missing exit orders
    # ------------------------------------------------------
    # TP LIMIT (partial)
    if (not has_existing_tp) and tp_qty > 0:
        print(f"{symbol}: Submitting TP LIMIT SELL qty={tp_qty} at {tp_price:.2f}.")
        tp_order = submit_limit_order(
            symbol=symbol,
            quantity=tp_qty,
            side=OrderSide.SELL,
            limit_price=tp_price,
            extended=extended,
        )
        if tp_order is not None:
            print(f"{symbol}: TP LIMIT order submitted:")
            print(tp_order)
        else:
            print(f"{symbol}: TP LIMIT order not placed (see warnings above). Intended TP was {tp_price:.2f}.")

    # SL STOP (remainder)
    if (not has_existing_sl) and sl_qty > 0:
        print(f"{symbol}: Submitting SL STOP SELL qty={sl_qty} at {sl_price:.2f}.")
        sl_order = submit_stop_loss_order(
            symbol=symbol,
            quantity=sl_qty,
            stop_price=sl_price,
            extended=extended,
        )
        if sl_order is not None:
            print(f"{symbol}: SL STOP order submitted:")
            print(sl_order)
        else:
            print(f"{symbol}: SL STOP order not placed (see warnings above). Intended SL was {sl_price:.2f}.")


def _order_timestamp_utc(order) -> Optional[datetime]:
    """
    Return the best available order timestamp in UTC.
    alpaca-py order objects commonly have submitted_at, created_at.
    """
    ts = getattr(order, "submitted_at", None) or getattr(order, "created_at", None)
    if ts is None:
        return None
    # Ensure tz-aware UTC
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)

def cancel_order_by_id_safe(order_id: str) -> bool:
    try:
        _trading_client.cancel_order_by_id(order_id)
        return True
    except Exception as e:
        print(f"[WARN] Failed to cancel order {order_id}: {e}")
        return False

def cancel_stale_entry_buy_limits_for_symbol(symbol: str, ttl_seconds: int) -> int:
    """
    Cancels stale OPEN BUY LIMIT orders for this symbol.
    Only intended for entry orders. Does not touch SELL exits.
    Returns number canceled.
    """
    open_orders = get_open_orders_for_symbol(symbol)
    now_utc = datetime.now(timezone.utc)

    canceled = 0
    for o in open_orders:
        try:
            side = str(o.side).lower()
            order_type = str(getattr(o, "order_type", getattr(o, "type", ""))).lower()
            status = str(getattr(o, "status", "")).lower()

            if "buy" not in side:
                continue
            if "limit" not in order_type:
                continue
            if "open" not in status and "new" not in status and "pending" not in status:
                # Keep it conservative: only handle clearly-open-ish orders
                continue

            submitted_at = _order_timestamp_utc(o)
            if submitted_at is None:
                continue

            age_seconds = (now_utc - submitted_at).total_seconds()
            if age_seconds < ttl_seconds:
                continue

            print(f"[ENTRY] {symbol}: Canceling stale BUY LIMIT {o.id} age={age_seconds:.0f}s")
            if cancel_order_by_id_safe(str(o.id)):
                canceled += 1
                # Log cancel to DB if you already do elsewhere
                try:
                    log_order_event_to_db(
                        alpaca_order_id=str(o.id),
                        event_type="canceled",
                        filled_qty=float(getattr(o, "filled_qty", 0) or 0),
                        remaining_qty=float(float(getattr(o, "qty", 0) or 0) - float(getattr(o, "filled_qty", 0) or 0)),
                        status="canceled",
                    )
                except Exception:
                    pass

        except Exception as e:
            print(f"[WARN] cancel_stale_entry_buy_limits_for_symbol: {symbol}: {e}")

    return canceled

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Tuple, List

@dataclass
class EntryReplaceState:
    replaces: int = 0
    cooldown_until_epoch: float = 0.0


def _is_openish(status: str) -> bool:
    s = (status or "").lower()
    return ("open" in s) or ("new" in s) or ("pending" in s) or ("accepted" in s)


def _open_entry_buy_limit_orders(symbol: str):
    """Return open-ish BUY LIMIT orders for this symbol."""
    orders = get_open_orders_for_symbol(symbol)
    out = []
    for o in orders:
        side = str(getattr(o, "side", "")).lower()
        order_type = str(getattr(o, "order_type", getattr(o, "type", ""))).lower()
        status = str(getattr(o, "status", ""))
        if "buy" not in side:
            continue
        if "limit" not in order_type:
            continue
        if not _is_openish(status):
            continue
        out.append(o)
    return out


def replace_stale_entry_buy_limit_if_needed(
    symbol: str,
    desired_qty: float,
    desired_limit_price: float,
    state: EntryReplaceState,
    ttl_seconds: int,
    max_replaces: int,
    chase_pct: float,
    extended: bool = False,
) -> Tuple[bool, EntryReplaceState]:
    """
    If there is an existing open BUY LIMIT entry order older than TTL, cancel it and
    re-submit at a slightly higher price (chasing), up to max_replaces.

    Returns (did_replace, updated_state).
    """
    now_utc = datetime.now(timezone.utc)

    open_orders = _open_entry_buy_limit_orders(symbol)
    if not open_orders:
        return False, state

    # pick the oldest open entry
    def order_ts(o):
        return _order_timestamp_utc(o) or now_utc

    open_orders.sort(key=order_ts)
    oldest = open_orders[0]

    ts = _order_timestamp_utc(oldest)
    if ts is None:
        return False, state

    age_seconds = (now_utc - ts).total_seconds()
    if age_seconds < ttl_seconds:
        return False, state

    if state.replaces >= max_replaces:
        print(f"[ENTRY] {symbol}: stale entry detected but max replaces reached ({state.replaces}/{max_replaces}).")
        return False, state

    old_limit = float(getattr(oldest, "limit_price", 0) or 0)
    # chase upward from the *old* price to avoid oscillating if strategy recomputes
    chased = old_limit * (1.0 + float(chase_pct))
    new_limit = max(float(desired_limit_price), float(chased))

    print(
        f"[ENTRY] {symbol}: replacing stale BUY LIMIT. age={age_seconds:.0f}s "
        f"old_limit={old_limit:.4f} -> new_limit={new_limit:.4f} "
        f"replace={state.replaces+1}/{max_replaces}"
    )

    if not cancel_order_by_id_safe(str(oldest.id)):
        print(f"[WARN] {symbol}: could not cancel stale entry order {oldest.id}; not replacing.")
        return False, state

    # Submit the replacement order
    submit_limit_order(
        symbol=symbol,
        quantity=desired_qty,
        side=OrderSide.BUY,
        limit_price=new_limit,
        extended=extended,
    )

    # update state
    state.replaces += 1
    return True, state
