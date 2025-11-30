"""
Risk-aware position sizing utilities.

We provide two related helpers:

- compute_risk_limited_buy_quantity:
    Respect the "risk-limited account" idea (effective $2k, 50% cash reserve),
    and return a share count based only on price and available cash.

- compute_risk_based_position_size:
    Use true R-based risk per trade:
      R_dollars = r_per_trade * starting_cash
      risk_per_share = |entry_price - stop_loss_price|
      max_shares_from_R = R_dollars / risk_per_share

    Then cap that by the cash-based limit from compute_risk_limited_buy_quantity.
"""

from config import (
    RISK_LIMITED_STARTING_CASH,
    MIN_CASH_RESERVE_FRACTION,
)


def compute_risk_limited_buy_quantity(
    entry_price: float,
    available_cash: float,
    risk_fraction_of_cash: float = 1.0,
) -> float:
    """
    Compute how many shares we can buy based on:

      - A "risk-limited" effective starting capital (RISK_LIMITED_STARTING_CASH)
      - A minimum cash reserve fraction (MIN_CASH_RESERVE_FRACTION)
      - A fraction of the *tradable* cash we are willing to allocate
        to this specific trade (risk_fraction_of_cash).

    This does NOT consider stop-loss distance. It is purely a
    cash-budget-based sizing helper.
    """
    if entry_price <= 0:
        return 0.0

    # Cap effective cash to the risk-limited account size
    effective_cash = min(available_cash, RISK_LIMITED_STARTING_CASH)

    # Reserve a fraction of that effective cash
    min_cash_reserve = RISK_LIMITED_STARTING_CASH * MIN_CASH_RESERVE_FRACTION
    tradable_cash = max(0.0, effective_cash - min_cash_reserve)

    # Only risk some fraction of that tradable cash on this trade
    budget_for_trade = tradable_cash * risk_fraction_of_cash

    raw_shares = budget_for_trade / entry_price
    whole_shares = int(raw_shares)

    if whole_shares < 1:
        return 0.0

    return float(whole_shares)


def compute_risk_based_position_size(
    entry_price: float,
    stop_loss_price: float,
    available_cash: float,
    r_per_trade: float,
    starting_cash: float,
) -> float:
    """
    Compute position size using R-based risk management:

      - r_per_trade is the fraction of *starting_cash* you are willing to risk
        per trade (for example, 0.01 = 1% of starting equity per trade).

      - risk_per_share = |entry_price - stop_loss_price|
      - max_risk_dollars = r_per_trade * starting_cash
      - max_shares_from_R = max_risk_dollars / risk_per_share

    Then we cap that by a cash-based limit from compute_risk_limited_buy_quantity,
    so we also honor the "virtual $2k / 50% reserve" account safety rule.

    Returns a whole-number share quantity (float), or 0.0 if no valid size.
    """
    # Basic validation
    if entry_price <= 0 or stop_loss_price <= 0:
        return 0.0
    if r_per_trade <= 0 or starting_cash <= 0 or available_cash <= 0:
        return 0.0

    risk_per_share = abs(entry_price - stop_loss_price)
    if risk_per_share <= 0:
        return 0.0

    # How many dollars we are willing to lose on this trade
    max_risk_dollars = r_per_trade * starting_cash

    # Theoretical maximum shares allowed by that risk
    shares_from_risk = max_risk_dollars / risk_per_share

    # Cash-based cap, using our risk-limited cash logic
    shares_from_cash = compute_risk_limited_buy_quantity(
        entry_price=entry_price,
        available_cash=available_cash,
        risk_fraction_of_cash=1.0,
    )

    if shares_from_cash <= 0:
        return 0.0

    # Final size is the minimum of the risk and cash limits
    final_shares = min(shares_from_risk, shares_from_cash)
    whole_shares = int(final_shares)

    if whole_shares < 1:
        return 0.0

    return float(whole_shares)
