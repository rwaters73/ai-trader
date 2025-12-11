from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from alpaca.trading.client import TradingClient
from config import MAX_OPEN_POSITIONS, MAX_CAPITAL_PER_SYMBOL_FRACTION

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

@dataclass
class PortfolioExposureContext:
    """
    Snapshot of current portfolio positions and notional values.
    """
    total_equity: float
    open_positions_count: int
    per_symbol_notional: dict[str, float]


def build_portfolio_exposure_context(trading_client: TradingClient) -> PortfolioExposureContext:
    """
    Build a PortfolioExposureContext using *account equity* as the base,
    and per-symbol notionals from open positions.

    total_equity:
        - primary source: account.equity reported by Alpaca
        - fallback: sum of open position market values if equity is missing/zero

    This avoids the “total_equity ~= a few dollars” issue when you only
    have tiny positions open.
    """
    # 1) Get full account equity from Alpaca
    account = trading_client.get_account()
    account_equity = float(getattr(account, "equity", 0.0) or 0.0)

    # 2) Get all open positions and compute per-symbol notionals
    positions = trading_client.get_all_positions()

    positions_notional_sum = 0.0
    per_symbol_notional: dict[str, float] = {}
    open_positions_count = 0

    for position in positions:
        symbol = position.symbol
        market_value = float(position.market_value) if position.market_value is not None else 0.0
        qty = float(position.qty) if position.qty is not None else 0.0

        if qty != 0:
            positions_notional_sum += market_value
            per_symbol_notional[symbol] = market_value
            open_positions_count += 1

    # 3) Choose a sensible total_equity:
    #    normally account_equity (cash + positions); fall back to positions sum.
    if account_equity > 0:
        total_equity = account_equity
    else:
        total_equity = positions_notional_sum

    return PortfolioExposureContext(
        total_equity=total_equity,
        open_positions_count=open_positions_count,
        per_symbol_notional=per_symbol_notional,
    )

def can_open_new_position_with_portfolio_caps(
    symbol: str,
    order_notional: float,
    portfolio_ctx: PortfolioExposureContext,
) -> bool:
    """
    Check if we can open a new position in `symbol` given portfolio exposure caps.

    Rules:
      1) If symbol is not already open, we can only open if:
         portfolio_ctx.open_positions_count < MAX_OPEN_POSITIONS
      2) The notional value after this trade must not exceed the per-symbol cap:
         (current_symbol_notional + order_notional) <= MAX_CAPITAL_PER_SYMBOL_FRACTION * total_equity

    Args:
      - symbol: the ticker we want to trade
      - order_notional: the notional value (cost) of the order we want to place
      - portfolio_ctx: snapshot of current positions and equity

    Returns:
      - True if both constraints are satisfied; False otherwise.
    """
    current_symbol_notional = portfolio_ctx.per_symbol_notional.get(symbol, 0.0)
    is_symbol_already_open = current_symbol_notional > 0

    # Check max open positions constraint
    if not is_symbol_already_open:
        if portfolio_ctx.open_positions_count >= MAX_OPEN_POSITIONS:
            print(
                f"[RISK] Cannot open {symbol}: "
                f"already at max open positions ({portfolio_ctx.open_positions_count} >= {MAX_OPEN_POSITIONS})."
            )
            return False

    # Check per-symbol notional cap
    total_equity = max(portfolio_ctx.total_equity, 1.0)  # Avoid division by zero
    max_notional_per_symbol = MAX_CAPITAL_PER_SYMBOL_FRACTION * total_equity
    projected_symbol_notional = current_symbol_notional + order_notional

    if projected_symbol_notional > max_notional_per_symbol:
        print(
            f"[RISK] Cannot open {symbol}: "
            f"projected notional ({projected_symbol_notional:.2f}) exceeds cap "
            f"({max_notional_per_symbol:.2f} = {MAX_CAPITAL_PER_SYMBOL_FRACTION * 100:.1f}% of ${total_equity:.2f})."
        )
        return False

    return True
