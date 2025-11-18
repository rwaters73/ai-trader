from models import SymbolState, TargetPosition
from config import (
    MAX_ENTRY_PRICE_BY_SYMBOL,
    DEFAULT_MAX_ENTRY_PRICE,
    TP_PERCENT_BY_SYMBOL,
    DEFAULT_TP_PERCENT,
    BUY_QTY_BY_SYMBOL,
    DEFAULT_BUY_QTY,
    BRACKET_TP_PERCENT_BY_SYMBOL,
    DEFAULT_BRACKET_TP_PERCENT,
    BRACKET_SL_PERCENT_BY_SYMBOL,
    DEFAULT_BRACKET_SL_PERCENT,
)

from history import fetch_price_history
from signals import compute_recent_high_breakout_signal


def _compute_tp_level(avg_entry_price: float, tp_percent: float) -> float:
    """
    Compute the take-profit level from avg_entry_price and a TP percentage.
    Example: avg=100, tp_percent=5.0 -> 105.0
    """
    return avg_entry_price * (1.0 + tp_percent / 100.0)


def decide_target_position(state: SymbolState) -> TargetPosition:
    """
    Core strategy:
      - If open orders exist → do nothing, hold position.
      - If flat → use a history-based breakout signal + per-symbol entry thresholds
                  and per-symbol BUY_QTY to decide on a LONG entry.
      - If long → consider exit using per-symbol TP% above avg entry.
      - Otherwise → hold.
    """

    # ----------------------------------------------------
    # 1. If open orders → hold until resolved
    # ----------------------------------------------------
    if state.has_open_orders:
        return TargetPosition(
            symbol=state.symbol,
            target_qty=state.position_qty,
            reason="Open orders exist; waiting for them to fill or cancel.",
        )

    # ----------------------------------------------------
    # 2. If FLAT → consider LONG entry via breakout signal
    # ----------------------------------------------------
    if state.position_qty == 0.0:
        max_entry_price = MAX_ENTRY_PRICE_BY_SYMBOL.get(
            state.symbol,
            DEFAULT_MAX_ENTRY_PRICE,
        )

        buy_qty = BUY_QTY_BY_SYMBOL.get(
            state.symbol,
            DEFAULT_BUY_QTY,
        )

        # Fetch recent history and compute the breakout signal
        bars = fetch_price_history(state.symbol)
        signal = compute_recent_high_breakout_signal(bars)

        if signal is None:
            # No valid breakout setup according to our signal
            return TargetPosition(
                symbol=state.symbol,
                target_qty=0.0,
                reason=(
                    "Flat; no recent-high breakout entry signal. "
                    "See [signal] debug logs for details about bars count, "
                    "uptrend (close vs SMA), and breakout tolerance checks."
                ),
            )

        # We have a breakout signal; use a reference price to compare to max_entry
        # Prefer bid if valid, otherwise fall back to close from last bar.
        ref_price = None
        if state.bid is not None and state.bid > 0:
            ref_price = state.bid
        elif bars is not None and not bars.empty and "close" in bars.columns:
            ref_price = float(bars.iloc[-1]["close"])

        if ref_price is None or ref_price <= 0:
            return TargetPosition(
                symbol=state.symbol,
                target_qty=0.0,
                reason=(
                    "Flat; breakout signal present but no valid reference price "
                    f"(bid/close) for {state.symbol}."
                ),
            )

        if ref_price >= max_entry_price:
            return TargetPosition(
                symbol=state.symbol,
                target_qty=0.0,
                reason=(
                    "Flat; breakout signal present but reference price "
                    f"{ref_price:.2f} >= max_entry_price={max_entry_price:.2f} "
                    "for this symbol. Not entering."
                ),
            )

        # If we get here: we have a breakout signal and price is below our guardrail.
        # Use the signal's suggested limit price as our entry price.
        entry_price = signal.limit_price

        # Per-symbol bracket-style TP/SL percentages
        tp_pct = BRACKET_TP_PERCENT_BY_SYMBOL.get(state.symbol, DEFAULT_BRACKET_TP_PERCENT)
        sl_pct = BRACKET_SL_PERCENT_BY_SYMBOL.get(state.symbol, DEFAULT_BRACKET_SL_PERCENT)

        # Compute absolute TP/SL levels from the entry price
        take_profit_price = entry_price * (1.0 + tp_pct / 100.0)
        stop_loss_price   = entry_price * (1.0 - sl_pct / 100.0)

        return TargetPosition(
            symbol=state.symbol,
            target_qty=buy_qty,
            reason=(
                signal.reason
                + " Using LIMIT entry based on signal "
                  f"(ref_price={ref_price:.2f} < max_entry_price={max_entry_price:.2f}, "
                  f"qty={buy_qty}, limit={entry_price:.2f}, "
                  f"TP={take_profit_price:.2f} (+{tp_pct}%), "
                  f"SL={stop_loss_price:.2f} (-{sl_pct}%))."
            ),
            entry_type="limit",
            entry_limit_price=entry_price,
            take_profit_price=take_profit_price,
            stop_loss_price=stop_loss_price,
        )

    # ----------------------------------------------------
    # 3. If LONG → check per-symbol TP exit
    # ----------------------------------------------------
    if state.avg_entry_price is None:
        # Should not happen in normal use, but fail-safe:
        return TargetPosition(
            symbol=state.symbol,
            target_qty=state.position_qty,
            reason="Long position but avg_entry_price missing; holding.",
        )

    if state.bid is None or state.bid <= 0:
        return TargetPosition(
            symbol=state.symbol,
            target_qty=state.position_qty,
            reason="Long position but no valid bid; holding.",
        )

    # Per-symbol TP%, with fallback
    tp_percent = TP_PERCENT_BY_SYMBOL.get(
        state.symbol,
        DEFAULT_TP_PERCENT,
    )
    tp_level = _compute_tp_level(state.avg_entry_price, tp_percent)

    if state.bid >= tp_level:
        return TargetPosition(
            symbol=state.symbol,
            target_qty=0.0,
            reason=(
                f"Bid={state.bid:.2f} >= TP level={tp_level:.2f} "
                f"({tp_percent}% profit); exiting position."
            ),
        )

    # ----------------------------------------------------
    # 4. LONG but TP not hit → hold
    # ----------------------------------------------------
    return TargetPosition(
        symbol=state.symbol,
        target_qty=state.position_qty,
        reason=(
            "In long position; TP not hit "
            f"(bid={state.bid:.2f}, TP level={tp_level:.2f}, tp_percent={tp_percent}%). "
            "Holding."
        ),
    )
