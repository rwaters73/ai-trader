from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from alpaca.trading.client import TradingClient

# ---------------------------------------------------------------------------
# Configurable risk rules
# ---------------------------------------------------------------------------

# Treat this as the "virtual" starting account size, regardless of
# what Alpaca reports as your real cash/equity.
RISK_STARTING_CAPITAL = 2_000.0

# Never let projected cash fall below this fraction of starting capital.
# 0.5 means: don't open a new trade if we'd end up below $1,000.
MIN_CASH_FRACTION = 0.5


@dataclass
class RiskContext:
    """
    Snapshot of account + risk-adjusted values.
    """
    raw_cash: float
    raw_equity: float
    raw_buying_power: float

    effective_starting_capital: float
    min_cash_reserve: float
    effective_cash_for_trading: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_risk_context(trading_client: TradingClient) -> RiskContext:
    """
    Pull account info from Alpaca and build a RiskContext
    using the configured RISK_STARTING_CAPITAL and MIN_CASH_FRACTION.
    """
    account = trading_client.get_account()

    raw_cash = float(account.cash)
    raw_equity = float(account.equity)
    raw_buying_power = float(account.buying_power)

    # You can choose different behavior here. I like:
    # - Use the LOWER of (RISK_STARTING_CAPITAL, raw_cash) as starting_cap
    #   so you don't over-estimate what you really have.
    effective_start = min(RISK_STARTING_CAPITAL, raw_cash)

    min_reserve = effective_start * MIN_CASH_FRACTION

    # Effective cash for trading is also capped at our risk starting capital.
    # If you have 100k in paper, we still only "treat" 2k as usable.
    effective_cash_for_trading = min(raw_cash, effective_start)

    return RiskContext(
        raw_cash=raw_cash,
        raw_equity=raw_equity,
        raw_buying_power=raw_buying_power,
        effective_starting_capital=effective_start,
        min_cash_reserve=min_reserve,
        effective_cash_for_trading=effective_cash_for_trading,
    )


def can_open_new_position(
    risk_ctx: RiskContext,
    order_cost: float,
) -> bool:
    """
    Decide whether we are allowed to open a new position of given cost
    under our risk rules.

    Rules:
      1) Use effective_cash_for_trading as our "cash" for decisions.
      2) After this order, projected_cash must be >= min_cash_reserve.
    """
    if order_cost <= 0:
        return False

    projected_cash = risk_ctx.effective_cash_for_trading - order_cost

    # Enforce "never below half starting balance".
    if projected_cash < risk_ctx.min_cash_reserve:
        return False

    # Optional: you can also enforce "don't spend more than X% of starting_cap on one trade"
    # here if you want later.

    return True

def compute_risk_based_position_size(
    symbol: str,
    entry_price: float,
    stop_loss_price: float,
    available_cash: float,
    r_per_trade: float,
    starting_cash: float,
) -> float:
    """
    Compute how many shares to buy for a single trade, given:

      - entry_price:       proposed entry price
      - stop_loss_price:   SL level for the trade
      - available_cash:    how much cash we can spend right now
      - r_per_trade:       fraction of starting_cash we are willing to risk (e.g. 0.01 = 1R)
      - starting_cash:     baseline account size (e.g. 2000 for a “risk-limited” account)

    The idea:
      - Risk per share = entry_price - stop_loss_price
      - Max risk in dollars = r_per_trade * starting_cash
      - Max shares by risk = max_risk_dollars / risk_per_share
      - Also cap by what we can actually pay for with available_cash.
    """
    if entry_price <= 0 or stop_loss_price <= 0:
        return 0.0

    risk_per_share = entry_price - stop_loss_price
    if risk_per_share <= 0:
        # No meaningful SL below entry => cannot size by risk.
        return 0.0

    max_risk_dollars = r_per_trade * starting_cash
    if max_risk_dollars <= 0:
        return 0.0

    # Shares limited by risk
    max_shares_by_risk = max_risk_dollars / risk_per_share

    # Shares limited by available cash
    max_shares_by_cash = available_cash / entry_price

    # Final allowed share count
    buy_quantity = min(max_shares_by_risk, max_shares_by_cash)

    # We’ll round down to whole shares for now.
    buy_quantity = int(buy_quantity)
    return float(buy_quantity)
