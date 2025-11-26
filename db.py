import sqlite3
from datetime import datetime, timezone
from typing import Optional

from pathlib import Path

# Central location of the SQLite DB file
DB_PATH = Path("data/trading_log.db")

#DB_FILE = "trading_log.db"


# ----------- Connection Helper -----------

def get_connection():
    """
    Returns a SQLite connection.
    SQLite automatically creates the DB file if missing.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # dict-like results
    return conn


# ----------- Schema Initialization -----------

def init_db():
    """
    Initialize tables if they do not exist.
    Safe to call at program startup.
    """
    conn = get_connection()
    cur = conn.cursor()

    # 1. Table for strategy/eod decisions
    cur.execute("""
        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            session TEXT,
            in_eod_window INTEGER,
            symbol TEXT,
            bid REAL,
            ask REAL,
            position_qty REAL,
            avg_entry_price REAL,
            pnl_pct REAL,
            target_qty REAL,
            reason TEXT
        );
    """)

    # 2. Table for orders your bot sends to Alpaca
    cur.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            alpaca_order_id TEXT,
            symbol TEXT,
            side TEXT,
            qty REAL,
            order_type TEXT,
            time_in_force TEXT,
            status TEXT
        );
    """)

    # 3. Table for order lifecycle updates (fills, cancels, partial fills)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS order_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            alpaca_order_id TEXT,
            event_type TEXT,        -- e.g. "filled", "canceled", "partial_fill"
            filled_qty REAL,
            remaining_qty REAL,
            status TEXT
        );
    """)

    conn.commit()
    conn.close()

# ----------- Insert Helpers -----------

def log_decision_to_db(
    timestamp: str,
    session: str,
    in_eod_window: bool,
    symbol: str,
    bid: Optional[float],
    ask: Optional[float],
    position_qty: float,
    avg_entry_price: Optional[float],
    pnl_pct: Optional[float],
    target_qty: float,
    reason: str,
):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO decisions (
            timestamp, session, in_eod_window, symbol, bid, ask,
            position_qty, avg_entry_price, pnl_pct, target_qty, reason
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, (
        timestamp, session, int(in_eod_window), symbol, bid, ask,
        position_qty, avg_entry_price, pnl_pct, target_qty, reason
    ))
    conn.commit()
    conn.close()


def log_order_to_db(order):
    """
    Logs an Alpaca order object (order submission) into the SQLite 'orders' table.
    """
    if order is None:
        return

    conn = get_connection()
    cur = conn.cursor()

    # Be robust to SDK differences: try order_type then type
    order_type = str(getattr(order, "order_type", getattr(order, "type", "")))
    time_in_force = str(getattr(order, "time_in_force", ""))
    side = str(getattr(order, "side", ""))
    status = str(getattr(order, "status", ""))

    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    cur.execute("""
        INSERT INTO orders (
            timestamp, alpaca_order_id, symbol, side, qty,
            order_type, time_in_force, status
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?);
    """, (
        timestamp,
        str(getattr(order, "id", "")),
        getattr(order, "symbol", ""),
        side,
        float(getattr(order, "qty", 0) or 0),
        order_type,
        time_in_force,
        status,
    ))

    conn.commit()
    conn.close()

def log_risk_event_to_db(symbol: str, action: str, cost: float, allowed: bool, message: str = ""):
    """
    Log any risk-limits decision (allowed/blocked) into SQLite.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS risk_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        symbol TEXT NOT NULL,
        action TEXT NOT NULL,
        cost REAL NOT NULL,
        allowed INTEGER NOT NULL,
        message TEXT
    )
    """)

    cur.execute("""
    INSERT INTO risk_events (timestamp, symbol, action, cost, allowed, message)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(timespec="seconds"),
        symbol,
        action,
        float(cost),
        1 if allowed else 0,
        message,
    ))

    conn.commit()
    conn.close()


def log_order_event_to_db(
    alpaca_order_id: str,
    event_type: str,
    filled_qty: float,
    remaining_qty: float,
    status: str
):
    conn = get_connection()
    cur = conn.cursor()

    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    cur.execute("""
        INSERT INTO order_events (
            timestamp, alpaca_order_id, event_type, filled_qty,
            remaining_qty, status
        )
        VALUES (?, ?, ?, ?, ?, ?);
    """, (
        timestamp,
        alpaca_order_id,
        event_type,
        filled_qty,
        remaining_qty,
        status,
    ))

    conn.commit()
    conn.close()

