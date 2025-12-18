from __future__ import annotations

import time
from datetime import datetime, time as dtime
from pathlib import Path
from typing import List, Optional

import pytz

from broker import (
    build_symbol_state,
    reconcile_position,
    ensure_exit_orders_for_symbol,
    cancel_all_open_orders,
    flatten_all,
    cancel_stale_entry_orders_for_symbol,
)
from strategy import decide_target_position
from config import ENTRY_RETRY_COOLDOWN_SECONDS

# Circuit breakers (safe if you added them, optional if you did not)
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

    We want to accept typical US tickers:
      - Letters, numbers, dot (BRK.B), dash (if it ever appears)
    We explicitly reject anything that looks like a dataclass repr or has
    punctuation that indicates a whole object got written to the file.
    """
    if not token:
        return False

    # Reject obviously bad tokens from accidental repr dumps
    if "(" in token or ")" in token or "," in token:
        return False

    # Very conservative allowed characters
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
        if not stripped:
            continue
        if stripped.startswith("#"):
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


# We avoid importing timedelta up top unless we need it
from datetime import timedelta  # noqa: E402  (kept here to avoid clutter above)


# -----------------------------
# Main loop
# -----------------------------
def main(symbols_to_trade: List[str],
         iterations: int = ITERATIONS_DEFAULT,
         interval_seconds: float = INTERVAL_SECONDS_DEFAULT) -> None:
    if not symbols_to_trade:
        print("[WARN] No symbols provided. Exiting.")
        return

    print(f"Starting bounded loop for symbols: {', '.join(symbols_to_trade)}")
    print(f"Iterations: {iterations}, Interval: {interval_seconds}s")
    print("Trading-hours filter: 8:30–15:00 Central, Mon–Fri.")
    print()

    # Track per-symbol cooldown after stale entry cancellation
    # Maps symbol -> timestamp when cooldown expires
    entry_retry_cooldown: dict[str, float] = {}

    try:
        for i in range(iterations):
            loop_dt = _now_central()
            iso_ts = loop_dt.isoformat(timespec="seconds")

            print(f"\n--- Iteration {i + 1}/{iterations} at {iso_ts} ---")

            in_rth = is_in_rth(loop_dt)
            in_ext = not in_rth  # your broker functions already accept extended flag

            if in_rth:
                print("Within RTH session. Running decision cycles for all symbols...")
            else:
                print("Within Extended-hours session. Running decision cycles for all symbols...")

            # Circuit breaker: daily loss limit
            if has_hit_daily_loss_limit is not None:
                try:
                    if has_hit_daily_loss_limit():
                        print("[CIRCUIT] Daily loss limit hit. Cancelling orders and flattening.")
                        try:
                            cancel_all_open_orders()
                        except Exception as e:
                            print(f"[WARN] cancel_all_open_orders failed: {e}")
                        try:
                            flatten_all(symbols_to_trade)
                        except Exception as e:
                            print(f"[WARN] flatten_all failed: {e}")
                        break
                except Exception as e:
                    print(f"[WARN] Circuit breaker check failed: {e}")

            # EOD policies
            if is_in_eod_window(loop_dt):
                print("EOD window active (last 15 minutes of RTH). Applying EOD policies.")
                print("EOD: Cancelling all open orders before applying EOD policies...")
                try:
                    cancel_all_open_orders()
                except Exception as e:
                    print(f"[WARN] Failed to cancel open orders: {e}")

                # Default policy: flatten everything into the close
                print("EOD: Flattening all symbols...")
                try:
                    flatten_all(symbols_to_trade)
                except Exception as e:
                    print(f"[WARN] Failed to flatten all: {e}")

                # Stop after EOD actions
                break

            # Per-symbol decision cycle
            for symbol in symbols_to_trade:
                try:
                    state = build_symbol_state(symbol)
                except Exception as e:
                    print(f"[WARN] Failed to build state for {symbol}: {e}")
                    continue

                # Check and cancel stale entry orders (BUY only, not exits)
                try:
                    canceled = cancel_stale_entry_orders_for_symbol(symbol)
                    if canceled > 0:
                        # Set cooldown: do not try to enter this symbol until cooldown expires
                        now_ts = time.time()
                        cooldown_until = now_ts + ENTRY_RETRY_COOLDOWN_SECONDS
                        entry_retry_cooldown[symbol] = cooldown_until
                        print(f"[cooldown] {symbol}: stale entry order(s) canceled; entry blocked for {ENTRY_RETRY_COOLDOWN_SECONDS}s.")
                except Exception as e:
                    print(f"[WARN] cancel_stale_entry_orders failed for {symbol}: {e}")

                try:
                    target = decide_target_position(state)
                except Exception as e:
                    print(f"[WARN] Strategy error for {symbol}: {e}")
                    continue

                # Check if we are in cooldown and block new entries
                now_ts = time.time()
                if symbol in entry_retry_cooldown and now_ts < entry_retry_cooldown[symbol]:
                    if target.target_qty > 0 and state.position_qty == 0:
                        # Block new entry while in cooldown
                        from models import TargetPosition
                        remaining_cooldown = entry_retry_cooldown[symbol] - now_ts
                        target = TargetPosition(
                            symbol=symbol,
                            target_qty=0.0,
                            reason=f"Entry retry cooldown active ({remaining_cooldown:.1f}s remaining)",
                            entry_type="market",
                            entry_limit_price=None,
                            take_profit_price=None,
                            stop_loss_price=None,
                        )
                else:
                    # Cooldown expired; clean up
                    entry_retry_cooldown.pop(symbol, None)

                try:
                    reconcile_position(state, target, extended=in_ext)
                except Exception as e:
                    print(f"[WARN] reconcile_position failed for {symbol}: {e}")
                    continue

                # Exit orders management (SL/TP logic lives in broker.py)
                try:
                    ensure_exit_orders_for_symbol(state, extended=in_ext)
                except Exception as e:
                    print(f"[WARN] ensure_exit_orders_for_symbol failed for {symbol}: {e}")
                    continue

            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        print("\n\n[EXIT] Ctrl+C received. Exiting gracefully.")


if __name__ == "__main__":
    symbols = load_symbols_from_watchlist(WATCHLIST_PATH)
    print(f"✓ Loaded {len(symbols)} symbols from {WATCHLIST_PATH.as_posix()}")
    main(symbols_to_trade=symbols, iterations=ITERATIONS_DEFAULT, interval_seconds=INTERVAL_SECONDS_DEFAULT)
