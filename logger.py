import csv
import os
from datetime import datetime
from typing import Optional

from models import SymbolState, TargetPosition
from config import (
    LOG_DECISIONS,
    LOG_FILE_PATH,
    LOG_ORDERS,
    ORDER_LOG_FILE_PATH,
)


def _ensure_dir_for(path: str):
    """
    Ensure the directory for the given file path exists.
    """
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


# ---------------------------------------------------------------------------
# Decision log
# ---------------------------------------------------------------------------

def init_decision_log():
    """
    Initialize the decision log file with a header row if it doesn't exist.
    Safe to call multiple times; it won't overwrite an existing file.
    """
    if not LOG_DECISIONS:
        return

    _ensure_dir_for(LOG_FILE_PATH)

    if not os.path.exists(LOG_FILE_PATH):
        with open(LOG_FILE_PATH, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "session",
                "in_eod_window",
                "symbol",
                "bid",
                "ask",
                "position_qty",
                "avg_entry_price",
                "pnl_pct",
                "target_qty",
                "reason",
            ])


def _pnl_percent(state: SymbolState) -> Optional[float]:
    """
    Wrapper around SymbolState.pnl_percent() so we keep CSV decision logging
    consistent with the rest of the system.
    """
    return state.pnl_percent()


def log_decision(
    state: SymbolState,
    target: TargetPosition,
    session_label: str,
    in_eod_window: bool,
    now: Optional[datetime] = None,
):
    """
    Log one decision row for a given symbol and cycle.
    """
    if not LOG_DECISIONS:
        return

    if now is None:
        now = datetime.now()

    _ensure_dir_for(LOG_FILE_PATH)

    pnl_pct = _pnl_percent(state)

    with open(LOG_FILE_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            now.isoformat(timespec="seconds"),
            session_label,
            int(in_eod_window),  # 1 or 0
            state.symbol,
            state.bid if state.bid is not None else "",
            state.ask if state.ask is not None else "",
            state.position_qty,
            state.avg_entry_price if state.avg_entry_price is not None else "",
            f"{pnl_pct:.4f}" if pnl_pct is not None else "",
            target.target_qty,
            target.reason,
        ])


# ---------------------------------------------------------------------------
# Order log
# ---------------------------------------------------------------------------

def init_order_log():
    """
    Initialize the order log file with a header row if it doesn't exist.
    """
    if not LOG_ORDERS:
        return

    _ensure_dir_for(ORDER_LOG_FILE_PATH)

    if not os.path.exists(ORDER_LOG_FILE_PATH):
        with open(ORDER_LOG_FILE_PATH, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "order_id",
                "client_order_id",
                "symbol",
                "side",
                "type",
                "time_in_force",
                "extended_hours",
                "qty",
                "filled_qty",
                "status",
                "submitted_at",
                "filled_at",
                "canceled_at",
                "expired_at",
                "failed_at",
            ])


def log_order(order, now: Optional[datetime] = None):
    """
    Log a single Alpaca order object at the moment we submit it.

    NOTE: This captures the order's status as returned by submit_order,
    which for market orders in paper trading is often already 'filled'.
    For longer-lived orders, this won't capture later status transitions
    unless we later add a polling/refresh logger.
    """
    if not LOG_ORDERS:
        return
    if order is None:
        return

    if now is None:
        now = datetime.now()

    _ensure_dir_for(ORDER_LOG_FILE_PATH)

    # Safely extract attributes (order is typically alpaca.trading.models.Order)
    def get(attr, default=""):
        return getattr(order, attr, default)

    # Some fields are enums/objects; convert to string
    side = str(get("side", ""))
    order_type = str(get("order_type", get("type", "")))
    tif = str(get("time_in_force", ""))
    status = str(get("status", ""))

    # Extended hours may or may not exist depending on order type
    extended_hours = getattr(order, "extended_hours", False)

    with open(ORDER_LOG_FILE_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            now.isoformat(timespec="seconds"),
            str(get("id", "")),
            get("client_order_id", ""),
            get("symbol", ""),
            side,
            order_type,
            tif,
            int(bool(extended_hours)),
            get("qty", ""),
            get("filled_qty", ""),
            status,
            # These stamped attributes may be datetime or None
            str(get("submitted_at", "")),
            str(get("filled_at", "")),
            str(get("canceled_at", "")),
            str(get("expired_at", "")),
            str(get("failed_at", "")),
        ])
