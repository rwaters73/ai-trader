from __future__ import annotations

import time
from datetime import datetime, time as dtime, timedelta
from pathlib import Path
from typing import List

import pytz

from broker import (
    build_symbol_state,
    reconcile_position,
    ensure_exit_orders_for_symbol,
    cancel_all_open_orders,
    flatten_all,
    cancel_stale_entry_orders_for_symbol,
    replace_stale_entry_buy_limit_if_needed,
    EntryReplaceState,
)

from strategy import decide_target_position
from config import (
    ENTRY_ORDER_TTL_SECONDS,
    ENTRY_MAX_REPLACES,
    ENTRY_REPLACE_CHASE_PCT,
    ENTRY_RETRY_COOLDOWN_SECONDS,
)

# Per-symbol replace state for TTL cancel + replace
_entry_replace_state: dict[str, EntryReplaceState] = {}

# Circuit breakers (optional)
try:
    from circuit_breakers import has_hit_daily_loss_limit
except Exception:
    has_hit_daily_loss_limit = None  # type: ignore


# -----------------------------
# Runtime configuration
# -----------------------------
WATCHLIST_PATH = Path("data/live_watchlist.txt")

ITERATIONS_DEFAULT = 5000
INTERVAL_SECONDS_DEFAULT = 1.0

CENTRAL_TZ = pytz.timezone("America/Chicago")

# Regular trading hours used by this bot (Central)
RTH_START = dtime(hour=8, minute=30)
RTH_END = dtime(hour=15, minute=0)

# EOD management window (last N minutes of RTH)
EOD_WINDOW_MINUTES = 15


# -----------------------------
# Symbol loading
# -----------------------------
def _looks_like_valid_symbol(token: str) -> bool:
    """
    Conservative validation for symbols.

    Accept typical US tickers:
      - Letters, numbers, dot (BRK.B), dash

    Reject anything that looks like a dataclass repr or object dump.
    """
    if not token:
        return False

    # Reject obviously bad tokens from accidental repr dumps
    if "(" in token or ")" in token or "," in token:
        return False

    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-")
    upper = token.upper()
    return all(ch in allowed for ch in upper)


def load_symbols_from_watchlist(path: Path = WATCHLIST_PATH) -> List[str]:
    """
    Reads one symbol per line from data/live_watchlist.txt.

    Defensive behavior:
      - strips whitespace
      - skips blank lines and comments (#)
      - takes only the first whitespace-delimited token
      - rejects tokens that do not look like symbols
      - de-duplicates while preserving order
    """
    if not path.exists():
        print(f"[WARN] Watchlist not found: {path}. No symbols loaded.")
        return []

    raw_lines = path.read_text(encoding="utf-8").splitlines()

    symbols: List[str] = []
    seen = set()

    skipped = 0
    for line in raw_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        token = stripped.split()[0].strip()
        if not _looks_like_valid_symbol(token):
            skipped += 1
            continue

        sym = token.upper()
        if sym not in seen:
            seen.add(sym)
            symbols.append(sym)

    if skipped:
        print(f"[WARN] Skipped {skipped} invalid watchlist line(s).")

    return symbols


# -----------------------------
# Session helpers
# -----------------------------
def _now_central() -> datetime:
    return datetime.now(tz=CENTRAL_TZ)


def is_weekday(dt: datetime) -> bool:
    return dt.weekday() < 5  # Mon=0 ... Fri=4


def is_in_rth(dt: datetime) -> bool:
    if not is_weekday(dt):
        return False
    t = dt.time()
    return RTH_START <= t <= RTH_END


def is_in_eod_window(dt: datetime) -> bool:
    """
    True during the last EOD_WINDOW_MINUTES minutes of RTH.
    Example: if RTH_END is 15:00 and window is 15 minutes,
    EOD window is 14:45 to 15:00.
    """
    if not is_in_rth(dt):
        return False
    end_dt = dt.replace(hour=RTH_END.hour, minute=RTH_END.minute, second=0, microsecond=0)
    window_start = end_dt - timedelta(minutes=EOD_WINDOW_MINUTES)
    return window_start <= dt <= end_dt


# -----------------------------
# Circuit breaker helpers
# -----------------------------
def _capture_session_start_equity() -> float:
    """Best-effort; returns 0.0 if unavailable."""
    equity = 0.0
    if has_hit_daily_loss_limit is not None:
        try:
            from circuit_breakers import get_current_equity

            equity = get_current_equity()
            print(f"Session start equity: ${equity:.2f}")
        except Exception as e:
            print(f"[WARN] Could not capture session start equity: {e}")
    return equity


def _apply_circuit_breaker(session_start_equity: float, symbols_to_trade: List[str]) -> bool:
    """Returns True if we should stop."""
    if has_hit_daily_loss_limit is None:
        return False

    try:
        if has_hit_daily_loss_limit(session_start_equity):
            print("[CIRCUIT] Daily loss limit hit. Cancelling orders and flattening.")
            try:
                cancel_all_open_orders()
            except Exception as e:
                print(f"[WARN] cancel_all_open_orders failed: {e}")
            try:
                flatten_all(symbols_to_trade)
            except Exception as e:
                print(f"[WARN] flatten_all failed: {e}")
            return True
    except Exception as e:
        print(f"[WARN] Circuit breaker check failed: {e}")
    return False


def _apply_eod_policies(symbols_to_trade: List[str]) -> bool:
    """Apply EOD actions; returns True to stop loop after handling."""
    print("EOD window active (last 15 minutes of RTH). Applying EOD policies.")
    print("EOD: Cancelling all open orders before applying EOD policies...")
    try:
        cancel_all_open_orders()
    except Exception as e:
        print(f"[WARN] Failed to cancel open orders: {e}")

    print("EOD: Flattening all symbols...")
    try:
        flatten_all(symbols_to_trade)
    except Exception as e:
        print(f"[WARN] Failed to flatten all: {e}")

    return True


# -----------------------------
# TTL cancel + replace helpers
# -----------------------------
def _set_entry_cooldown(symbol: str, seconds: float) -> None:
    """Sets a per-symbol cooldown for new entries after stale order cancellation."""
    st = _entry_replace_state.get(symbol, EntryReplaceState())
    st.cooldown_until_epoch = time.time() + float(seconds)
    _entry_replace_state[symbol] = st


def _process_symbol(symbol: str, in_ext: bool) -> None:
    """Single-symbol decision cycle."""
    try:
        state = build_symbol_state(symbol)
    except Exception as e:
        print(f"[WARN] Failed to build state for {symbol}: {e}")
        return

    # Per-symbol cooldown after stale entry cancellation or too many replaces
    st = _entry_replace_state.get(symbol)
    now_epoch = time.time()
    if st and now_epoch < getattr(st, "cooldown_until_epoch", 0):
        remaining = int(st.cooldown_until_epoch - now_epoch)
        print(f"[ENTRY] {symbol}: cooldown active ({remaining}s). Skipping.")
        return

    try:
        target = decide_target_position(state)
    except Exception as e:
        print(f"[WARN] Strategy error for {symbol}: {e}")
        return

    # Determine if this cycle is trying to place an ENTRY limit buy
    is_flat = abs(state.position_qty) < 1e-6
    wants_entry_limit_buy = (
        is_flat
        and target.target_qty > 0
        and (target.entry_type or "").lower() == "limit"
        and target.entry_limit_price is not None
    )

    # TTL cancel + replace is only relevant when:
    #  - we want to enter (flat -> long)
    #  - there are open orders (assumed to include that entry order)
    if wants_entry_limit_buy and state.has_open_orders:
        # 1) If the existing entry is stale, broker cancels it.
        #    If it cancels, we go into cooldown to avoid immediate re-submission.
        try:
            canceled = cancel_stale_entry_orders_for_symbol(
                symbol=symbol,
                ttl_seconds=ENTRY_ORDER_TTL_SECONDS,
            )
        except TypeError:
            # In case broker signature differs slightly, fall back to positional call
            try:
                canceled = cancel_stale_entry_orders_for_symbol(symbol, ENTRY_ORDER_TTL_SECONDS)
            except Exception as e:
                print(f"[WARN] cancel_stale_entry_orders_for_symbol failed for {symbol}: {e}")
                canceled = False
        except Exception as e:
            print(f"[WARN] cancel_stale_entry_orders_for_symbol failed for {symbol}: {e}")
            canceled = False

        if canceled:
            print(
                f"[ENTRY] {symbol}: canceled stale entry order(s). "
                f"Cooldown {ENTRY_RETRY_COOLDOWN_SECONDS}s."
            )
            _set_entry_cooldown(symbol, ENTRY_RETRY_COOLDOWN_SECONDS)
            return  # skip reconcile this iteration

        # 2) If not canceled, we may still want to replace (chase) if conditions met.
        st = _entry_replace_state.get(symbol, EntryReplaceState())
        try:
            did_replace, st = replace_stale_entry_buy_limit_if_needed(
                symbol=symbol,
                desired_qty=target.target_qty,  # state is flat so delta equals target
                desired_limit_price=target.entry_limit_price,
                state=st,
                ttl_seconds=ENTRY_ORDER_TTL_SECONDS,
                max_replaces=ENTRY_MAX_REPLACES,
                chase_pct=ENTRY_REPLACE_CHASE_PCT,
                extended=in_ext,
            )
            _entry_replace_state[symbol] = st
        except Exception as e:
            print(f"[WARN] replace_stale_entry_buy_limit_if_needed failed for {symbol}: {e}")
            did_replace = False

        if did_replace:
            # Important: skip reconcile this iteration to prevent double-submit
            return

        # 3) Safety: if we have exhausted replaces, cancel and cool down.
        #    We do not assume exact field name, but we try common ones.
        replace_count = getattr(st, "replace_count", None)
        if replace_count is None:
            replace_count = getattr(st, "num_replaces", 0)

        if int(replace_count or 0) >= int(ENTRY_MAX_REPLACES):
            print(
                f"[ENTRY] {symbol}: max replaces reached ({replace_count}). "
                f"Cancelling stale entry and cooling down."
            )
            try:
                cancel_all_open_orders_for_symbol = getattr(
                    __import__("broker"), "cancel_all_open_orders_for_symbol", None
                )
                if callable(cancel_all_open_orders_for_symbol):
                    cancel_all_open_orders_for_symbol(symbol)
                else:
                    cancel_all_open_orders()  # fallback
            except Exception as e:
                print(f"[WARN] Failed to cancel orders after max replaces for {symbol}: {e}")

            _set_entry_cooldown(symbol, ENTRY_RETRY_COOLDOWN_SECONDS)
            return

    # Normal reconciliation path
    try:
        reconcile_position(state, target, extended=in_ext)
    except Exception as e:
        print(f"[WARN] reconcile_position failed for {symbol}: {e}")
        return

    # Exit orders management (SL/TP logic lives in broker.py)
    try:
        ensure_exit_orders_for_symbol(state, extended=in_ext)
    except Exception as e:
        print(f"[WARN] ensure_exit_orders_for_symbol failed for {symbol}: {e}")
        return


# -----------------------------
# Main loop
# -----------------------------
def main(
    symbols_to_trade: List[str],
    iterations: int = ITERATIONS_DEFAULT,
    interval_seconds: float = INTERVAL_SECONDS_DEFAULT,
) -> None:
    if not symbols_to_trade:
        print("[WARN] No symbols provided. Exiting.")
        return

    print(f"Starting bounded loop for symbols: {', '.join(symbols_to_trade)}")
    print(f"Iterations: {iterations}, Interval: {interval_seconds}s")
    print("Trading-hours filter: 8:30–15:00 Central, Mon–Fri.")
    print()

    session_start_equity = _capture_session_start_equity()

    try:
        for i in range(iterations):
            loop_dt = _now_central()
            iso_ts = loop_dt.isoformat(timespec="seconds")

            print(f"\n--- Iteration {i + 1}/{iterations} at {iso_ts} ---")

            in_rth = is_in_rth(loop_dt)
            in_ext = not in_rth  # extended-hours allowed when not in RTH

            if in_rth:
                print("Within RTH session. Running decision cycles for all symbols...")
            else:
                print("Within Extended-hours session. Running decision cycles for all symbols...")

            # Circuit breaker: daily loss limit
            if _apply_circuit_breaker(session_start_equity, symbols_to_trade):
                break

            # EOD policies (still only during RTH EOD window)
            if is_in_eod_window(loop_dt):
                if _apply_eod_policies(symbols_to_trade):
                    break

            for symbol in symbols_to_trade:
                _process_symbol(symbol=symbol, in_ext=in_ext)

            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        print("\n\n[EXIT] Ctrl+C received. Exiting gracefully.")


if __name__ == "__main__":
    symbols = load_symbols_from_watchlist(WATCHLIST_PATH)
    print(f"✓ Loaded {len(symbols)} symbols from {WATCHLIST_PATH.as_posix()}")
    main(symbols_to_trade=symbols, iterations=ITERATIONS_DEFAULT, interval_seconds=INTERVAL_SECONDS_DEFAULT)
