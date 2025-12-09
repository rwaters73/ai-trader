# daily_risk.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional, Tuple

from config import MAX_DAILY_TRADES_PER_DAY


@dataclass
class DailyRiskState:
    """
    Tracks simple per day risk limits:
      - trades_opened: how many *new positions* we have opened
      - halted: whether we have hit a circuit breaker
      - halt_reason: human readable reason
    """
    trade_date: date
    trades_opened: int = 0
    halted: bool = False
    halt_reason: Optional[str] = None


_daily_state: Optional[DailyRiskState] = None


def _get_today() -> date:
    """Return today's calendar date in local time."""
    return datetime.now().date()


def _ensure_state_for_today() -> DailyRiskState:
    """
    Initialize or roll over the in memory state when the calendar day changes.
    This keeps the counters per day without touching the database.
    """
    global _daily_state
    today = _get_today()

    if _daily_state is None or _daily_state.trade_date != today:
        _daily_state = DailyRiskState(trade_date=today)

    return _daily_state


def can_open_new_trade() -> Tuple[bool, Optional[str]]:
    """
    Decide whether we are allowed to open a *new position* right now.

    Returns:
        (allowed, reason_if_blocked)
    """
    state = _ensure_state_for_today()

    if state.halted:
        return False, state.halt_reason or "Daily risk limit already hit."

    if state.trades_opened >= MAX_DAILY_TRADES_PER_DAY:
        state.halted = True
        state.halt_reason = (
            f"Max daily trades limit reached "
            f"({state.trades_opened}/{MAX_DAILY_TRADES_PER_DAY})."
        )
        return False, state.halt_reason

    return True, None


def register_new_trade() -> None:
    """
    Increment the count of new positions opened today.

    Call this *only* after a new entry order has been successfully submitted.
    """
    state = _ensure_state_for_today()
    state.trades_opened += 1
