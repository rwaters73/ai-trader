from __future__ import annotations

from typing import Tuple

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAccountRequest  # imported for completeness, not strictly needed

from config import (
    ALPACA_API_KEY_ID,
    ALPACA_API_SECRET_KEY,
    MAX_DAILY_LOSS_DOLLARS,
)


# One dedicated trading client for circuit breaker checks.
# This keeps the module self-contained and avoids importing from broker.
_trading_client = TradingClient(
    ALPACA_API_KEY_ID,
    ALPACA_API_SECRET_KEY,
    paper=True,
)


def get_daily_pnl_dollars() -> float:
    """
    Compute *today's* realized+unrealized PnL in dollars, based on Alpaca account fields.

    Alpaca's TradingAccount object has:
      - equity: current total equity
      - last_equity: prior session's equity

    A simple approximation for today's PnL is:
        daily_pnl = equity - last_equity
    """
    account = _trading_client.get_account()

    # These are strings in Alpaca's response; cast to float.
    equity = float(account.equity)
    last_equity = float(account.last_equity)

    daily_pnl = equity - last_equity
    return daily_pnl


def has_hit_daily_loss_limit() -> Tuple[bool, str]:
    """
    Check whether today's PnL is below the configured daily loss limit.

    Returns:
        (hit_limit, message)
        - hit_limit: True if we should stop trading for the day.
        - message: human-readable explanation.
    """
    try:
        daily_pnl = get_daily_pnl_dollars()
    except Exception as exc:
        # If we cannot fetch the account for some reason, fail safe:
        # allow trading but log the issue.
        return (
            False,
            f"[CIRCUIT] Could not compute daily PnL (exception: {exc}). "
            f"Proceeding without daily-loss enforcement.",
        )

    loss_limit = -abs(MAX_DAILY_LOSS_DOLLARS)

    if daily_pnl <= loss_limit:
        return (
            True,
            (
                f"[CIRCUIT] Daily loss limit reached: PnL={daily_pnl:.2f} "
                f"<= {loss_limit:.2f}. Halting trading for the rest of the day."
            ),
        )
    else:
        return (
            False,
            (
                f"[CIRCUIT] Daily PnL OK: {daily_pnl:.2f} vs limit {loss_limit:.2f}. "
                f"Continuing to trade."
            ),
        )
