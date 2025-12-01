from typing import Optional

from models import SymbolState, TargetPosition
from config import (
    TP_PERCENT_BY_SYMBOL,
    DEFAULT_TP_PERCENT,
    BUY_QTY_BY_SYMBOL,
    DEFAULT_BUY_QTY,
    MAX_INTRADAY_PULLBACK_PCT,
    MIN_INTRADAY_BARS_FOR_CONFIRMATION,
    INTRADAY_LOOKBACK_MINUTES,
    INTRADAY_BAR_SIZE_MINUTES,
    ATR_PERIOD_DEFAULT,
    ATR_STOP_MULTIPLIER_DEFAULT,
    ATR_TP_MULTIPLIER_DEFAULT,
    RISK_R_PER_TRADE_DEFAULT,
)
from data import get_recent_history, get_intraday_history
from signals import compute_recent_high_breakout_signal, EntrySignal
from indicators import compute_atr_series

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _compute_take_profit_level(
    average_entry_price: float,
    take_profit_percent: float,
) -> float:
    """
    Compute the take-profit price level from the average entry price
    and a TP percentage.
    Example: average_entry_price=100, take_profit_percent=5.0 -> 105.0
    """
    return average_entry_price * (1.0 + take_profit_percent / 100.0)


def _get_take_profit_percent_for_symbol(symbol: str) -> float:
    """
    Look up the per-symbol take-profit percentage, with a fallback default.
    """
    return TP_PERCENT_BY_SYMBOL.get(symbol, DEFAULT_TP_PERCENT)


def _get_buy_quantity_for_symbol(symbol: str) -> float:
    """
    Look up the per-symbol buy quantity, with a fallback default.
    """
    return BUY_QTY_BY_SYMBOL.get(symbol, DEFAULT_BUY_QTY)

def compute_risk_based_position_size(
    entry_price: float,
    stop_loss_price: float,
    account_size: float,
    risk_percent: float
) -> float:
    """
    Calculate the number of shares to buy based on risk per trade.
    """
    risk_dollars = account_size * risk_percent
    stop_distance = entry_price - stop_loss_price

    if stop_distance <= 0:
        return 0.0

    raw_quantity = risk_dollars / stop_distance
    return max(raw_quantity, 0.0)


# ---------------------------------------------------------------------------
# Entry logic: DAILY breakout + INTRADAY confirmation
# ---------------------------------------------------------------------------


def _decide_entry_from_daily_and_intraday(state: SymbolState) -> TargetPosition:
    """
    Decide whether to ENTER a long position when we are currently flat.

    Hybrid logic:
      1. Use DAILY bars to determine if there is a recent-high breakout signal.
      2. If there is a daily signal, use INTRADAY bars to confirm that price
         has not pulled back more than MAX_INTRADAY_PULLBACK_PCT below the
         proposed limit price.

    If both conditions are satisfied, we return a TargetPosition that:
      - sets target_qty to the configured buy quantity
      - marks entry_type="limit"
      - sets entry_limit_price to the signal's proposed limit price
    """
    symbol = state.symbol

    # -----------------------
    # Step 1: Daily breakout
    # -----------------------
    daily_bars_dataframe = get_recent_history(symbol, lookback_days=60)

    if daily_bars_dataframe is None or daily_bars_dataframe.empty:
        return TargetPosition(
            symbol=symbol,
            target_qty=0.0,
            reason=(
                "Flat; no daily bars available for breakout evaluation. "
                "Staying flat."
            ),
        )

    daily_entry_signal: Optional[EntrySignal] = compute_recent_high_breakout_signal(
        daily_bars_dataframe
    )

    if daily_entry_signal is None:
        return TargetPosition(
            symbol=symbol,
            target_qty=0.0,
            reason=(
                "Flat; no DAILY breakout entry signal. "
                "See [signal] debug logs for details."
            ),
        )

    proposed_limit_price = daily_entry_signal.limit_price

    # -----------------------
    # Step 2: Intraday confirmation
    # -----------------------
    intraday_bars_dataframe = get_intraday_history(
        symbol=symbol,
        lookback_minutes=INTRADAY_LOOKBACK_MINUTES,
        bar_size_minutes=INTRADAY_BAR_SIZE_MINUTES,
    )

    if (
        intraday_bars_dataframe is None
        or intraday_bars_dataframe.empty
        or len(intraday_bars_dataframe) < MIN_INTRADAY_BARS_FOR_CONFIRMATION
    ):
        return TargetPosition(
            symbol=symbol,
            target_qty=0.0,
            reason=(
                "Daily breakout signal present, but insufficient INTRADAY data "
                "for confirmation. Staying flat."
            ),
        )

    last_intraday_bar = intraday_bars_dataframe.iloc[-1]
    last_intraday_close_price = float(last_intraday_bar["close"])

    allowed_minimum_price = proposed_limit_price * (
        1.0 - MAX_INTRADAY_PULLBACK_PCT / 100.0
    )

    if last_intraday_close_price < allowed_minimum_price:
        pullback_percent = (
            (proposed_limit_price - last_intraday_close_price) / proposed_limit_price
        ) * 100.0

        return TargetPosition(
            symbol=symbol,
            target_qty=0.0,
            reason=(
                "Daily breakout signal, but intraday confirmation FAILED: "
                f"last intraday close={last_intraday_close_price:.2f} is "
                f"{pullback_percent:.2f}% below proposed limit={proposed_limit_price:.2f}, "
                f"which exceeds MAX_INTRADAY_PULLBACK_PCT="
                f"{MAX_INTRADAY_PULLBACK_PCT:.2f}%. Staying flat."
            ),
        )

    # ----------------------------------------------------
    # Intraday confirmation passed: we are willing to enter.
    # Here is where we tell the broker layer to use a LIMIT entry.
    # ----------------------------------------------------
    # Intraday confirmation passed: we are willing to enter.
    from config import RISK_PERCENT_PER_TRADE

    stop_loss_price = daily_entry_signal.limit_price * (1 - MAX_INTRADAY_PULLBACK_PCT/100)
    account_size = 100000.0  # later: replace with real fetched account value

    buy_quantity = compute_risk_based_position_size(
        entry_price=daily_entry_signal.limit_price,
        stop_loss_price=stop_loss_price,
        account_size=account_size,
        risk_percent=RISK_PERCENT_PER_TRADE
    )

    # Express this as a LIMIT entry so the broker layer can place a limit order
    # at the breakout-derived price.
    return TargetPosition(
        symbol=symbol,
        target_qty=buy_quantity,
        reason=(
            "Daily breakout signal + intraday confirmation PASSED. "
            f"Proposed limit entry at {proposed_limit_price:.2f}. "
            f"Daily reason: {daily_entry_signal.reason}"
        ),
        entry_type="limit",
        entry_limit_price=proposed_limit_price,
    )


# ---------------------------------------------------------------------------
# Public strategy entry point
# ---------------------------------------------------------------------------


def decide_target_position(state: SymbolState) -> TargetPosition:
    """
    Core strategy:

      - If open orders exist → do nothing, hold position.
      - If flat → consider long entry using DAILY breakout + INTRADAY confirmation.
      - If long → consider exit using per-symbol TP% above average entry price.
      - Otherwise → hold.

    This function operates only on the abstract position size; the broker layer
    turns TargetPosition into actual orders.
    """

    symbol = state.symbol

    # ----------------------------------------------------
    # 1. If there are open orders → hold until resolved
    # ----------------------------------------------------
    if state.has_open_orders:
        return TargetPosition(
            symbol=symbol,
            target_qty=state.position_qty,
            reason="Open orders exist; waiting for them to fill or cancel.",
        )

    # ----------------------------------------------------
    # 2. If FLAT → hybrid daily + intraday entry logic
    # ----------------------------------------------------
    if state.position_qty == 0.0:
        return _decide_entry_from_daily_and_intraday(state)

    # ----------------------------------------------------
    # 3. If LONG → check per-symbol TP exit
    # ----------------------------------------------------
    if state.avg_entry_price is None:
        # Should not happen in normal use, but fail-safe:
        return TargetPosition(
            symbol=symbol,
            target_qty=state.position_qty,
            reason="Long position but avg_entry_price is missing; holding.",
        )

    if state.bid is None or state.bid <= 0:
        return TargetPosition(
            symbol=symbol,
            target_qty=state.position_qty,
            reason="Long position but no valid bid; holding.",
        )

    take_profit_percent = _get_take_profit_percent_for_symbol(symbol)
    take_profit_level = _compute_take_profit_level(
        state.avg_entry_price,
        take_profit_percent,
    )

    if state.bid >= take_profit_level:
        return TargetPosition(
            symbol=symbol,
            target_qty=0.0,
            reason=(
                f"Bid={state.bid:.2f} >= TP level={take_profit_level:.2f} "
                f"({take_profit_percent}% profit); exiting position."
            ),
        )

    # ----------------------------------------------------
    # 4. LONG but TP not hit → hold
    # ----------------------------------------------------
    return TargetPosition(
        symbol=symbol,
        target_qty=state.position_qty,
        reason=(
            "In long position; TP not hit. "
            f"bid={state.bid:.2f}, TP level={take_profit_level:.2f}, "
            f"tp_percent={take_profit_percent}%."
        ),
    )
