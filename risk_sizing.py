# risk_sizing.py

from __future__ import annotations

from typing import Optional

from config import BUY_QTY_BY_SYMBOL, DEFAULT_BUY_QTY


def compute_risk_limited_buy_quantity(
    symbol: str,
    entry_price: float,
    account_cash: float,
    stop_loss_percent: float,
    max_risk_percent_per_trade: float = 1.0,
    max_capital_percent_per_trade: float = 10.0,
) -> float:
    """
    Calculate how many shares we are allowed to buy for a given trade, based on:

      - account_cash: how much cash we currently have available
      - stop_loss_percent: distance to the planned stop (e.g., 2.0 means -2%)
      - max_risk_percent_per_trade: max % of account_cash we are willing to LOSE
      - max_capital_percent_per_trade: max % of account_cash we are willing to TIE UP

    We also cap the result by the config-based BUY_QTY_BY_SYMBOL / DEFAULT_BUY_QTY
    so your existing per-symbol sizing still acts as an upper bound.

    Returns a non-negative integer number of shares.
    """

    if entry_price <= 0 or account_cash <= 0:
        return 0.0

    # If no SL defined or invalid, fall back to a config-based fixed size
    if stop_loss_percent is None or stop_loss_percent <= 0:
        return float(BUY_QTY_BY_SYMBOL.get(symbol, DEFAULT_BUY_QTY))

    # 1) How many dollars are we allowed to lose on this trade?
    dollar_risk_allowed = account_cash * (max_risk_percent_per_trade / 100.0)

    # 2) How many dollars are we allowed to tie up on this trade?
    max_capital_to_use = account_cash * (max_capital_percent_per_trade / 100.0)

    # Risk per share given our planned stop
    risk_per_share = entry_price * (stop_loss_percent / 100.0)
    if risk_per_share <= 0:
        return 0.0

    # Sizing from risk and from capital constraints
    risk_limited_quantity = dollar_risk_allowed / risk_per_share
    capital_limited_quantity = max_capital_to_use / entry_price

    raw_quantity = min(risk_limited_quantity, capital_limited_quantity)

    # Apply config-based upper cap
    max_config_quantity = float(BUY_QTY_BY_SYMBOL.get(symbol, DEFAULT_BUY_QTY))
    raw_quantity = min(raw_quantity, max_config_quantity)

    # Use whole shares
    integer_quantity = int(raw_quantity)
    if integer_quantity < 0:
        integer_quantity = 0

    return float(integer_quantity)
