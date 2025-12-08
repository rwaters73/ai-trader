"""
report_live_performance.py

Summarize live trading performance using the SQLite orders table.

- Reconstructs closed round-trip trades (BUY -> SELL) per symbol using FIFO.
- Uses limit_price as the executed price (approximation).
- Computes basic PnL stats and prints a summary.

Run:
    python report_live_performance.py
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

from db import DB_PATH  # we already defined this in db.py


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class ClosedTrade:
    symbol: str
    quantity: float
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    side: str          # "long" for now (we only do long)
    pnl_dollars: float
    pnl_percent: float
    holding_minutes: float


# -----------------------------
# Helpers
# -----------------------------

def parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    """
    Parse an ISO8601 string (e.g., from Alpaca / our DB) into a datetime.
    Returns None if the value is None or invalid.
    """
    if not value:
        return None
    try:
        # Handles '2025-12-04T15:24:13+00:00' style strings
        return datetime.fromisoformat(value)
    except Exception:
        return None


def load_filled_orders_from_db() -> List[sqlite3.Row]:
    """
    Load all filled (or partially filled) orders from the `orders` table.

    Assumes `orders` schema has at least:
      - alpaca_order_id TEXT
      - symbol TEXT
      - side TEXT  ("buy" / "sell")
      - qty REAL
      - limit_price REAL
      - status TEXT
      - submitted_at TEXT (ISO string)
      - filled_at TEXT (ISO string, may be NULL for some statuses)
    """
    db_path = Path(DB_PATH)
    if not db_path.exists():
        print(f"[report] DB does not exist at {db_path}. No data to report.")
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                alpaca_order_id,
                symbol,
                side,
                qty,
                limit_price,
                status,
                submitted_at,
                filled_at
            FROM orders
            WHERE status IN ('filled', 'partially_filled')
            ORDER BY
                COALESCE(filled_at, submitted_at)
            """
        )
        rows = cursor.fetchall()
        print(f"[report] Loaded {len(rows)} filled/partial orders from DB.")
        return rows
    finally:
        conn.close()


def reconstruct_closed_trades(orders: List[sqlite3.Row]) -> List[ClosedTrade]:
    """
    Reconstruct closed trades per symbol using FIFO matching:

      - BUY orders add to an open position (as lots in a queue).
      - SELL orders reduce the open position:
          * We match the sell quantity against earliest open BUY lots (FIFO).
          * Each matched chunk becomes a ClosedTrade with its own PnL.

    This is a standard way to approximate round-trip trades from order history.
    """
    closed_trades: List[ClosedTrade] = []

    # For each symbol, we maintain a FIFO list of open lots:
    #   lot = dict(qty_remaining, price, time)
    open_lots_by_symbol: Dict[str, List[dict]] = {}

    for row in orders:
        symbol = row["symbol"]
        side_raw = str(row["side"]).lower()
        quantity = float(row["qty"] or 0.0)
        limit_price = float(row["limit_price"] or 0.0)

        submitted_time = parse_iso_datetime(row["submitted_at"])
        filled_time = parse_iso_datetime(row["filled_at"]) or submitted_time

        if quantity <= 0 or limit_price <= 0:
            # Skip nonsensical rows
            continue

        # Initialize symbol bucket
        if symbol not in open_lots_by_symbol:
            open_lots_by_symbol[symbol] = []

        if side_raw == "buy":
            # Open (or add to) a long position
            open_lots_by_symbol[symbol].append(
                {
                    "qty_remaining": quantity,
                    "price": limit_price,
                    "time": filled_time,
                }
            )

        elif side_raw == "sell":
            # Close or reduce an existing long position
            lots = open_lots_by_symbol[symbol]
            qty_to_close = quantity

            while qty_to_close > 0 and lots:
                lot = lots[0]
                lot_qty = lot["qty_remaining"]
                match_qty = min(qty_to_close, lot_qty)

                entry_price = lot["price"]
                exit_price = limit_price
                entry_time = lot["time"]
                exit_time = filled_time

                pnl_dollars = (exit_price - entry_price) * match_qty
                pnl_percent = (
                    (exit_price - entry_price) / entry_price * 100.0
                    if entry_price > 0
                    else 0.0
                )
                holding_minutes = (
                    (exit_time - entry_time).total_seconds() / 60.0
                    if entry_time and exit_time
                    else 0.0
                )

                closed_trades.append(
                    ClosedTrade(
                        symbol=symbol,
                        quantity=match_qty,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        entry_time=entry_time,
                        exit_time=exit_time,
                        side="long",
                        pnl_dollars=pnl_dollars,
                        pnl_percent=pnl_percent,
                        holding_minutes=holding_minutes,
                    )
                )

                # Update lot / remaining qty
                lot["qty_remaining"] -= match_qty
                qty_to_close -= match_qty

                if lot["qty_remaining"] <= 1e-8:
                    lots.pop(0)

            # If qty_to_close > 0 and no lots remain, that is an over-close;
            # we ignore the unmatched portion for now.

        else:
            # Unknown side; ignore
            continue

    return closed_trades


def compute_trade_statistics(trades: List[ClosedTrade]) -> dict:
    """
    Compute global summary stats from a list of ClosedTrade objects.
    """
    if not trades:
        return {
            "trade_count": 0,
            "winner_count": 0,
            "loser_count": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "total_pnl_pct": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
        }

    total_pnl = sum(t.pnl_dollars for t in trades)
    winner_pnls = [t.pnl_dollars for t in trades if t.pnl_dollars > 0]
    loser_pnls = [t.pnl_dollars for t in trades if t.pnl_dollars < 0]

    trade_count = len(trades)
    winner_count = len(winner_pnls)
    loser_count = len(loser_pnls)
    win_rate = (winner_count / trade_count * 100.0) if trade_count > 0 else 0.0

    def safe_avg(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    stats = {
        "trade_count": trade_count,
        "winner_count": winner_count,
        "loser_count": loser_count,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "total_pnl_pct": 0.0,  # we do not know starting capital here
        "average_win": safe_avg(winner_pnls),
        "average_loss": safe_avg(loser_pnls),
        "best_trade": max(winner_pnls) if winner_pnls else 0.0,
        "worst_trade": min(loser_pnls) if loser_pnls else 0.0,
    }
    return stats


def print_summary(trades: List[ClosedTrade]) -> None:
    stats = compute_trade_statistics(trades)

    print("==================== LIVE PERFORMANCE SUMMARY ====================")
    print(f"Closed trades:     {stats['trade_count']}")
    print(f"Winners:           {stats['winner_count']}")
    print(f"Losers:            {stats['loser_count']}")
    print(f"Win rate:          {stats['win_rate']:.2f}%")
    print()
    print(f"Total PnL:         {stats['total_pnl']:,.2f}")
    print(f"Average win:       {stats['average_win']:,.2f}")
    print(f"Average loss:      {stats['average_loss']:,.2f}")
    print(f"Best trade:        {stats['best_trade']:,.2f}")
    print(f"Worst trade:       {stats['worst_trade']:,.2f}")
    print("===================================================================")

    # Optional: per-symbol breakdown
    pnl_by_symbol: Dict[str, float] = {}
    for t in trades:
        pnl_by_symbol.setdefault(t.symbol, 0.0)
        pnl_by_symbol[t.symbol] += t.pnl_dollars

    if pnl_by_symbol:
        print("\nPer-symbol PnL:")
        for symbol, pnl in sorted(pnl_by_symbol.items(), key=lambda x: x[0]):
            print(f"  {symbol}: {pnl:,.2f}")


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    orders = load_filled_orders_from_db()
    closed_trades = reconstruct_closed_trades(orders)
    print_summary(closed_trades)
