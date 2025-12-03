from datetime import datetime
import time
from logger import init_decision_log, init_order_log, log_decision
from db import init_db, log_decision_to_db
from config import SYMBOLS, ITERATIONS, INTERVAL_SECONDS
from broker import (
    build_symbol_state,
    reconcile_position,
    flatten_symbol,
    cancel_all_open_orders,
    get_position_info,
    ensure_exit_orders_for_symbol,
)
from models import SymbolState
from strategy import decide_target_position
from eod_policy import apply_eod_policy

from regime import get_market_regime
from models import TargetPosition  # you already import this somewhere, just make sure it is there


def is_rth(now: datetime | None = None) -> bool:
    """
    Regular Trading Hours: 8:30–15:00 Central, Monday–Friday.
    Assumes this script runs on a machine set to Central Time.
    """
    if now is None:
        now = datetime.now()

    # Monday=0, Sunday=6
    if now.weekday() > 4:  # 5=Saturday, 6=Sunday
        return False

    minutes = now.hour * 60 + now.minute

    # 8:30 am CT to 3:00 pm CT
    open_min = 8 * 60 + 30    # 8:30
    close_min = 15 * 60       # 15:00

    return open_min <= minutes <= close_min


def is_extended_session(now: datetime | None = None) -> bool:
    """
    Full extended-hours session for US equities:
    Pre-market:   5:00–8:30 CT
    After-hours:  15:00–19:00 CT
    """
    if now is None:
        now = datetime.now()

    if now.weekday() > 4:
        return False

    minutes = now.hour * 60 + now.minute

    pre_open_start = 5 * 60       # 5:00 CT
    pre_open_end   = 8 * 60 + 30  # 8:30 CT

    after_close_start = 15 * 60   # 15:00 CT
    after_close_end   = 19 * 60   # 19:00 CT

    in_pre  = pre_open_start <= minutes < pre_open_end
    in_after = after_close_start <= minutes < after_close_end

    return in_pre or in_after


def is_eod_window(now: datetime | None = None) -> bool:
    """
    End-of-day window: last 15 minutes of RTH.
    For RTH 8:30–15:00 CT, this is 14:45–15:00 CT, Mon–Fri.
    """
    if now is None:
        now = datetime.now()

    if now.weekday() > 4:  # Sat/Sun
        return False

    minutes = now.hour * 60 + now.minute
    start = 14 * 60 + 45   # 14:45
    end = 15 * 60          # 15:00

    return start <= minutes < end





def main():
    print(f"Starting bounded loop for symbols: {', '.join(SYMBOLS)}")
    print(f"Iterations: {ITERATIONS}, Interval: {INTERVAL_SECONDS}s")
    print("Trading-hours filter: 8:30–15:00 Central, Mon–Fri.\n")

    # Initialize logs
    init_decision_log()
    init_order_log()
    init_db()

    eod_orders_canceled = False  #cancel all open orders
    eod_completed = False   #break out after all eod instructions are executed

    try:
        for i in range(ITERATIONS):
            now = datetime.now()
            print(f"\n--- Iteration {i+1}/{ITERATIONS} at {now.isoformat(timespec='seconds')} ---")

            in_rth = is_rth(now)
            in_ext = is_extended_session(now)
            in_eod = is_eod_window(now)
            # Get market regime once per iteration (SPY-based)
            market_regime = get_market_regime()

            if in_rth or in_ext:
                session_label = "RTH" if in_rth else "Extended-hours"
                print(f"Within {session_label} session. Running decision cycles for all symbols...")

                if in_eod:
                    print("EOD window active (last 15 minutes of RTH). Applying EOD policies.")

                    if not eod_orders_canceled:
                        print("EOD: Cancelling all open orders before applying EOD policies...")
                        cancel_all_open_orders()
                        eod_orders_canceled = True

                for symbol in SYMBOLS:
                    state = build_symbol_state(symbol)
                    target = decide_target_position(state)

                    if in_eod:
                        target = apply_eod_policy(state, target)

                    # ------------------------------------------------
                    # Regime filter: block NEW entries in bad regimes
                    # ------------------------------------------------
                    if (
                        state.position_qty == 0.0
                        and target.target_qty > 0.0
                        and market_regime is not None
                        and not market_regime.is_bull
                    ):
                        original_reason = target.reason
                        target = TargetPosition(
                            symbol=state.symbol,
                            target_qty=0.0,
                            reason=(
                                f"Regime filter: blocking new long entries. "
                                f"{market_regime.explanation} "
                                f"(original entry reason: {original_reason})"
                            ),
                            entry_type="market",
                            entry_limit_price=None,
                            take_profit_price=None,
                            stop_loss_price=None,
                        )
                        
                    log_decision(
                        state=state,
                        target=target,
                        session_label=session_label,
                        in_eod_window=in_eod,
                        now=now,
                    )

                    log_decision_to_db(
                        timestamp=now.isoformat(timespec="seconds"),
                        session=session_label,
                        in_eod_window=in_eod,
                        symbol=state.symbol,
                        bid=state.bid,
                        ask=state.ask,
                        position_qty=state.position_qty,
                        avg_entry_price=state.avg_entry_price,
                        pnl_pct=state.pnl_percent(),
                        target_qty=target.target_qty,
                        reason=target.reason,
                    )

                    reconcile_position(state, target, extended=in_ext)

                    # After any entry/exit adjustments for this symbol, ensure that
                    # if we hold a long position, we have a TP exit order on the book.
                    ensure_exit_orders_for_symbol(state, extended=in_ext)

                # After we've processed ALL symbols in the EOD window, we can exit gracefully
                if in_eod:
                    eod_completed = True
                    print("EOD: All EOD policies applied for all symbols. Exiting gracefully.")
                    break

            else:
                print("Outside all trading sessions. Skipping trading logic this cycle.")

            time.sleep(INTERVAL_SECONDS)

        # ---- after the for-loop ends ----
        if eod_completed:
            print("Program ended after completing EOD actions.")
        else:
            print("Program ended after hitting iteration limit.")

    except KeyboardInterrupt:
        print("\nLoop interrupted by user (Ctrl+C). Exiting early.\n")
    finally:
        print("\nCompleted program. Exiting gracefully.\n")


if __name__ == "__main__":
    main()
