"""
circuit_breakers.py

Simple daily loss circuit breaker for live trading.

Uses Alpaca's TradingClient to read current account equity and compares it
to the equity at the start of the session.

If the loss in dollars or percent crosses configured thresholds, we signal
that trading should stop for the day.
"""

from typing import Optional

from alpaca.trading.client import TradingClient

from config import (
    ALPACA_API_KEY_ID,
    ALPACA_API_SECRET_KEY,
    #DAILY_LOSS_LIMIT_DOLLARS,
    #DAILY_LOSS_LIMIT_PERCENT,
    MAX_DAILY_LOSS_DOLLARS,
    MAX_DAILY_LOSS_PERCENT,
)

# One shared trading client for this module
_trading_client = TradingClient(
    ALPACA_API_KEY_ID,
    ALPACA_API_SECRET_KEY,
    paper=True,
)


def get_current_equity() -> float:
    """
    Fetch the current account equity from Alpaca as a float.
    """
    account = _trading_client.get_account()
    return float(account.equity)


def has_hit_daily_loss_limit(session_start_equity: float) -> bool:
    """
    Return True if today's loss exceeds either:

      - DAILY_LOSS_LIMIT_DOLLARS (absolute dollars), or
      - DAILY_LOSS_LIMIT_PERCENT (percent of session_start_equity)

    session_start_equity is the equity at the time your bot started running
    for the day. main.py should capture that once and pass it here.
    """
    current_equity = get_current_equity()
    equity_change = current_equity - session_start_equity

    # If we are not down, then no circuit breaker.
    if equity_change >= 0:
        return False

    loss_dollars = -equity_change

    # 1) Check dollar loss limit
    if MAX_DAILY_LOSS_DOLLARS is not None and loss_dollars >= MAX_DAILY_LOSS_DOLLARS:
        print(
            f"[circuit] Daily dollar loss limit hit. "
            f"Loss={loss_dollars:.2f}, limit={MAX_DAILY_LOSS_DOLLARS:.2f}"
        )
        return True

    # 2) Check percent loss limit
    if (
        MAX_DAILY_LOSS_PERCENT is not None
        and session_start_equity > 0
    ):
        loss_percent = loss_dollars / session_start_equity * 100.0
        if loss_percent >= MAX_DAILY_LOSS_PERCENT:
            print(
                f"[circuit] Daily percent loss limit hit. "
                f"Loss={loss_percent:.2f}%, limit={MAX_DAILY_LOSS_PERCENT:.2f}%"
            )
            return True

    return False
