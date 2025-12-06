"""
report_live_performance.py

Reads live trading data from the SQLite DB and prints a summary similar
to backtest.py: total PnL, win rate, average win/loss, holding duration, etc.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
from datetime import datetime

from db import DB_PATH

@dataclass
class LiveTrade:
    symbol: str
    entry_time: datetime
    exit_time: datetime
    qty: float
    entry_price: float
    exit_price: float
    pnl_dollars: float
    pnl_percent: float
    holding_minutes: float
# ...existing code...
from statistics import median
from typing import Dict

import sqlite3
from datetime import timezone

# ...existing code...

@dataclass
class LiveTrade:
    symbol: str
    entry_time: datetime
    exit_time: datetime
    qty: float
    entry_price: float
    exit_price: float
    pnl_dollars: float
    pnl_percent: float
    holding_minutes: float

# ------------------------------------------------------------------
# DB helpers: read filled order events and pair BUY->SELL to infer round-trips
# ------------------------------------------------------------------
def _parse_iso_timestamp(ts: str) -> datetime:
    # Handle common ISO formats, strip trailing Z if present
    if ts is None:
        raise ValueError("timestamp is None")
    if ts.endswith("Z"):
        ts = ts[:-1]
    # sqlite may store as "YYYY-MM-DD HH:MM:SS" or ISO; try fromisoformat then fallback to strptime
    try:
        return datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
    except Exception:
        try:
            return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        except Exception:
            # last resort: naive parse (may raise)
            return datetime.fromisoformat(ts)

def load_closed_trades_from_db() -> List[LiveTrade]:
    """
    Open the SQLite DB at DB_PATH, read filled order events (joined with orders),
    then pair BUY and SELL fills for the same symbol in chronological order
    to produce a list of closed LiveTrade objects.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Try a query that matches the schema implied by db.py: orders + order_events.
    # We select only "filled" events and include symbol/side from orders.
    try:
        cursor.execute(
            """
            SELECT oe.order_id,
                   o.symbol AS symbol,
                   o.side   AS side,
                   oe.timestamp    AS timestamp,
                   COALESCE(oe.filled_qty, oe.qty, oe.quantity)    AS qty,
                   COALESCE(oe.filled_price, oe.price, oe.avg_price) AS price
            FROM order_events oe
            JOIN orders o ON oe.order_id = o.id
            WHERE oe.event_type = 'filled'
            ORDER BY oe.timestamp ASC
            """
        )
        rows = cursor.fetchall()
    except Exception:
        # Fallback: try reading filled rows from orders table directly
        cursor.execute(
            """
            SELECT id AS order_id,
                   symbol,
                   side,
                   COALESCE(filled_at, timestamp, created_at) AS timestamp,
                   COALESCE(filled_qty, qty, quantity) AS qty,
                   COALESCE(filled_price, price, avg_price) AS price
            FROM orders
            WHERE status = 'filled' OR COALESCE(filled_qty,0) > 0
            ORDER BY COALESCE(filled_at, timestamp, created_at) ASC
            """
        )
        rows = cursor.fetchall()

    fills = []
    for r in rows:
        try:
            symbol = r["symbol"]
            side = r["side"]
            ts_raw = r["timestamp"]
            qty = float(r["qty"]) if r["qty"] is not None else 0.0
            price = float(r["price"]) if r["price"] is not None else 0.0
        except Exception:
            # skip malformed row
            continue

        if qty <= 0:
            continue

        try:
            ts = _parse_iso_timestamp(str(ts_raw))
        except Exception:
            # skip if timestamp cannot be parsed
            continue

        fills.append(
            {
                "symbol": symbol,
                "side": side.upper() if isinstance(side, str) else str(side).upper(),
                "timestamp": ts,
                "qty": qty,
                "price": price,
            }
        )

    conn.close()

    # Now pair BUY fills -> SELL fills (FIFO matching) per symbol
    trades: List[LiveTrade] = []
    buys_by_symbol: Dict[str, List[Dict]] = {}

    for f in fills:
        symbol = f["symbol"]
        side = f["side"]
        ts = f["timestamp"]
        qty = f["qty"]
        price = f["price"]

        if side.startswith("B"):  # BUY
            buys_by_symbol.setdefault(symbol, []).append(
                {"qty": qty, "price": price, "timestamp": ts}
            )
            continue

        if side.startswith("S"):  # SELL
            remaining_sell = qty
            buy_queue = buys_by_symbol.get(symbol, [])
            # match FIFO buys
            while remaining_sell > 0 and buy_queue:
                buy_leg = buy_queue[0]
                matched = min(remaining_sell, buy_leg["qty"])

                entry_time = buy_leg["timestamp"]
                exit_time = ts
                entry_price = float(buy_leg["price"])
                exit_price = float(price)
                pnl = (exit_price - entry_price) * matched
                pnl_pct = (exit_price - entry_price) / entry_price * 100.0 if entry_price != 0 else 0.0
                holding_minutes = (exit_time - entry_time).total_seconds() / 60.0

                trades.append(
                    LiveTrade(
                        symbol=symbol,
                        entry_time=entry_time,
                        exit_time=exit_time,
                        qty=matched,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        pnl_dollars=pnl,
                        pnl_percent=pnl_pct,
                        holding_minutes=holding_minutes,
                    )
                )

                # decrement quantities
                buy_leg["qty"] -= matched
                remaining_sell -= matched
                if buy_leg["qty"] <= 0:
                    buy_queue.pop(0)

            # if sells remain unmatched (shorts), skip them for now
            # (could be extended to support short-trades)
            continue

    return trades

# ------------------------------------------------------------------
# Stats + pretty print
# ------------------------------------------------------------------
def compute_live_trade_statistics(trades: List[LiveTrade]) -> dict:
    if not trades:
        return {
            "total_pnl": 0.0,
            "trade_count": 0,
            "winner_count": 0,
            "loser_count": 0,
            "win_rate": 0.0,
            "average_win": 0.0,
            "median_win": 0.0,
            "average_loss": 0.0,
            "median_loss": 0.0,
            "average_holding_minutes": 0.0,
            "median_holding_minutes": 0.0,
        }

    pnl_values = [t.pnl_dollars for t in trades]
    total_pnl = sum(pnl_values)
    winner_pnls = [p for p in pnl_values if p > 0]
    loser_pnls = [p for p in pnl_values if p < 0]
    holding_minutes = [t.holding_minutes for t in trades]

    def safe_avg(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    def safe_median(xs: List[float]) -> float:
        return median(xs) if xs else 0.0

    winner_count = len(winner_pnls)
    loser_count = len(loser_pnls)
    total_trades = len(trades)
    win_rate = (winner_count / total_trades * 100.0) if total_trades > 0 else 0.0

    return {
        "total_pnl": total_pnl,
        "trade_count": total_trades,
        "winner_count": winner_count,
        "loser_count": loser_count,
        "win_rate": win_rate,
        "average_win": safe_avg(winner_pnls),
        "median_win": safe_median(winner_pnls),
        "average_loss": safe_avg(loser_pnls),
        "median_loss": safe_median(loser_pnls),
        "average_holding_minutes": safe_avg(holding_minutes),
        "median_holding_minutes": safe_median(holding_minutes),
    }

def print_live_performance_summary(trades: List[LiveTrade]):
    stats = compute_live_trade_statistics(trades)
    print("=============== LIVE PERFORMANCE SUMMARY ===============")
    print(f"Trades:            {stats['trade_count']}")
    print(f"Total PnL:         {stats['total_pnl']:.2f}")
    print(f"Winners:           {stats['winner_count']}")
    print(f"Losers:            {stats['loser_count']}")
    print(f"Win rate:          {stats['win_rate']:.2f}%")
    print()
    print(f"Average win:       {stats['average_win']:.2f}")
    print(f"Median win:        {stats['median_win']:.2f}")
    print(f"Average loss:      {stats['average_loss']:.2f}")
    print(f"Median loss:       {stats['median_loss']:.2f}")
    print()
    print(f"Avg holding (min): {stats['average_holding_minutes']:.2f}")
    print(f"Median holding(min):{stats['median_holding_minutes']:.2f}")
    print("=======================================================")

if __name__ == "__main__":
    trades = load_closed_trades_from_db()
    print_live_performance_summary(trades)
#```// filepath: c:\Users\rickw\ai-trader\report_live_performance.py
# ...existing code...
from statistics import median
from typing import Dict

import sqlite3
from datetime import timezone

# ...existing code...

@dataclass
class LiveTrade:
    symbol: str
    entry_time: datetime
    exit_time: datetime
    qty: float
    entry_price: float
    exit_price: float
    pnl_dollars: float
    pnl_percent: float
    holding_minutes: float

# ------------------------------------------------------------------
# DB helpers: read filled order events and pair BUY->SELL to infer round-trips
# ------------------------------------------------------------------
def _parse_iso_timestamp(ts: str) -> datetime:
    # Handle common ISO formats, strip trailing Z if present
    if ts is None:
        raise ValueError("timestamp is None")
    if ts.endswith("Z"):
        ts = ts[:-1]
    # sqlite may store as "YYYY-MM-DD HH:MM:SS" or ISO; try fromisoformat then fallback to strptime
    try:
        return datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
    except Exception:
        try:
            return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        except Exception:
            # last resort: naive parse (may raise)
            return datetime.fromisoformat(ts)

def load_closed_trades_from_db() -> List[LiveTrade]:
    """
    Open the SQLite DB at DB_PATH, read filled order events (joined with orders),
    then pair BUY and SELL fills for the same symbol in chronological order
    to produce a list of closed LiveTrade objects.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Try a query that matches the schema implied by db.py: orders + order_events.
    # We select only "filled" events and include symbol/side from orders.
    try:
        cursor.execute(
            """
            SELECT oe.order_id,
                   o.symbol AS symbol,
                   o.side   AS side,
                   oe.timestamp    AS timestamp,
                   COALESCE(oe.filled_qty, oe.qty, oe.quantity)    AS qty,
                   COALESCE(oe.filled_price, oe.price, oe.avg_price) AS price
            FROM order_events oe
            JOIN orders o ON oe.order_id = o.id
            WHERE oe.event_type = 'filled'
            ORDER BY oe.timestamp ASC
            """
        )
        rows = cursor.fetchall()
    except Exception:
        # Fallback: try reading filled rows from orders table directly
        cursor.execute(
            """
            SELECT id AS order_id,
                   symbol,
                   side,
                   COALESCE(filled_at, timestamp, created_at) AS timestamp,
                   COALESCE(filled_qty, qty, quantity) AS qty,
                   COALESCE(filled_price, price, avg_price) AS price
            FROM orders
            WHERE status = 'filled' OR COALESCE(filled_qty,0) > 0
            ORDER BY COALESCE(filled_at, timestamp, created_at) ASC
            """
        )
        rows = cursor.fetchall()

    fills = []
    for r in rows:
        try:
            symbol = r["symbol"]
            side = r["side"]
            ts_raw = r["timestamp"]
            qty = float(r["qty"]) if r["qty"] is not None else 0.0
            price = float(r["price"]) if r["price"] is not None else 0.0
        except Exception:
            # skip malformed row
            continue

        if qty <= 0:
            continue

        try:
            ts = _parse_iso_timestamp(str(ts_raw))
        except Exception:
            # skip if timestamp cannot be parsed
            continue

        fills.append(
            {
                "symbol": symbol,
                "side": side.upper() if isinstance(side, str) else str(side).upper(),
                "timestamp": ts,
                "qty": qty,
                "price": price,
            }
        )

    conn.close()

    # Now pair BUY fills -> SELL fills (FIFO matching) per symbol
    trades: List[LiveTrade] = []
    buys_by_symbol: Dict[str, List[Dict]] = {}

    for f in fills:
        symbol = f["symbol"]
        side = f["side"]
        ts = f["timestamp"]
        qty = f["qty"]
        price = f["price"]

        if side.startswith("B"):  # BUY
            buys_by_symbol.setdefault(symbol, []).append(
                {"qty": qty, "price": price, "timestamp": ts}
            )
            continue

        if side.startswith("S"):  # SELL
            remaining_sell = qty
            buy_queue = buys_by_symbol.get(symbol, [])
            # match FIFO buys
            while remaining_sell > 0 and buy_queue:
                buy_leg = buy_queue[0]
                matched = min(remaining_sell, buy_leg["qty"])

                entry_time = buy_leg["timestamp"]
                exit_time = ts
                entry_price = float(buy_leg["price"])
                exit_price = float(price)
                pnl = (exit_price - entry_price) * matched
                pnl_pct = (exit_price - entry_price) / entry_price * 100.0 if entry_price != 0 else 0.0
                holding_minutes = (exit_time - entry_time).total_seconds() / 60.0

                trades.append(
                    LiveTrade(
                        symbol=symbol,
                        entry_time=entry_time,
                        exit_time=exit_time,
                        qty=matched,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        pnl_dollars=pnl,
                        pnl_percent=pnl_pct,
                        holding_minutes=holding_minutes,
                    )
                )

                # decrement quantities
                buy_leg["qty"] -= matched
                remaining_sell -= matched
                if buy_leg["qty"] <= 0:
                    buy_queue.pop(0)

            # if sells remain unmatched (shorts), skip them for now
            # (could be extended to support short-trades)
            continue

    return trades

# ------------------------------------------------------------------
# Stats + pretty print
# ------------------------------------------------------------------
def compute_live_trade_statistics(trades: List[LiveTrade]) -> dict:
    if not trades:
        return {
            "total_pnl": 0.0,
            "trade_count": 0,
            "winner_count": 0,
            "loser_count": 0,
            "win_rate": 0.0,
            "average_win": 0.0,
            "median_win": 0.0,
            "average_loss": 0.0,
            "median_loss": 0.0,
            "average_holding_minutes": 0.0,
            "median_holding_minutes": 0.0,
        }

    pnl_values = [t.pnl_dollars for t in trades]
    total_pnl = sum(pnl_values)
    winner_pnls = [p for p in pnl_values if p > 0]
    loser_pnls = [p for p in pnl_values if p < 0]
    holding_minutes = [t.holding_minutes for t in trades]

    def safe_avg(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    def safe_median(xs: List[float]) -> float:
        return median(xs) if xs else 0.0

    winner_count = len(winner_pnls)
    loser_count = len(loser_pnls)
    total_trades = len(trades)
    win_rate = (winner_count / total_trades * 100.0) if total_trades > 0 else 0.0

    return {
        "total_pnl": total_pnl,
        "trade_count": total_trades,
        "winner_count": winner_count,
        "loser_count": loser_count,
        "win_rate": win_rate,
        "average_win": safe_avg(winner_pnls),
        "median_win": safe_median(winner_pnls),
        "average_loss": safe_avg(loser_pnls),
        "median_loss": safe_median(loser_pnls),
        "average_holding_minutes": safe_avg(holding_minutes),
        "median_holding_minutes": safe_median(holding_minutes),
    }

def print_live_performance_summary(trades: List[LiveTrade]):
    stats = compute_live_trade_statistics(trades)
    print("=============== LIVE PERFORMANCE SUMMARY ===============")
    print(f"Trades:            {stats['trade_count']}")
    print(f"Total PnL:         {stats['total_pnl']:.2f}")
    print(f"Winners:           {stats['winner_count']}")
    print(f"Losers:            {stats['loser_count']}")
    print(f"Win rate:          {stats['win_rate']:.2f}%")
    print()
    print(f"Average win:       {stats['average_win']:.2f}")
    print(f"Median win:        {stats['median_win']:.2f}")
    print(f"Average loss:      {stats['average_loss']:.2f}")
    print(f"Median loss:       {stats['median_loss']:.2f}")
    print()
    print(f"Avg holding (min): {stats['average_holding_minutes']:.2f}")
    print(f"Median holding(min):{stats['median_holding_minutes']:.2f}")
    print("=======================================================")

#from Chatgpt
def main() -> None:
    live_trades = load_live_trades_from_db()
    stats = compute_live_trade_statistics(live_trades)
    print_live_performance_summary(stats)

if __name__ == "__main__":
    main()
#end from chatgpt

#from copilot
if __name__ == "__main__":
    trades = load_closed_trades_from_db()
    print_live_performance_summary(trades)
#end from copilot    