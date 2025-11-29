from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from pathlib import Path

import pandas as pd

from config import (
    TP_PERCENT_BY_SYMBOL,
    DEFAULT_TP_PERCENT,
    BRACKET_SL_PERCENT_BY_SYMBOL,
    DEFAULT_BRACKET_SL_PERCENT,
    BUY_QTY_BY_SYMBOL,
    DEFAULT_BUY_QTY,
    MAX_INTRADAY_PULLBACK_PCT,
    MIN_INTRADAY_BARS_FOR_CONFIRMATION,
)

RISK_R_PER_TRADE_DEFAULT = 1.0  # how many "R" per trade
MAX_RISK_FRACTION_PER_TRADE = 0.01  # 1% of starting cash per trade

from signals import compute_recent_high_breakout_signal, EntrySignal

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--symbol", default="TSLA")
args = parser.parse_args()


# ------------------------------------------------------------------
# Configuration – change this symbol to backtest a different ticker
# ------------------------------------------------------------------

SYMBOL_TO_TEST = args.symbol
DAILY_CSV_PATH_TEMPLATE = "data/{symbol}_daily.csv"
INTRADAY_CSV_PATH_TEMPLATE = "data/{symbol}_intraday.csv"

STARTING_CASH_DEFAULT = 100_000.0
MAX_HOLDING_DAYS = 30  # simple safety cap on how long we hold a position


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class BacktestTrade:
    symbol: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: float
    pnl_dollars: float
    pnl_percent: float
    holding_days: int


# ------------------------------------------------------------------
# Helpers to read CSV and normalize
# ------------------------------------------------------------------

def load_daily_bars_from_csv(csv_path: str) -> pd.DataFrame:
    file_path = Path(csv_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Daily CSV does not exist: {csv_path}")

    daily_bars_dataframe = pd.read_csv(file_path)

    if "timestamp" not in daily_bars_dataframe.columns:
        raise ValueError(f"CSV {csv_path} must have a 'timestamp' column.")

    daily_bars_dataframe["timestamp"] = pd.to_datetime(
        daily_bars_dataframe["timestamp"],
        utc=True,
    )

    daily_bars_dataframe = (
        daily_bars_dataframe.sort_values("timestamp").reset_index(drop=True)
    )
    return daily_bars_dataframe


def load_intraday_bars_from_csv(csv_path: str) -> pd.DataFrame:
    file_path = Path(csv_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Intraday CSV does not exist: {csv_path}")

    intraday_bars_dataframe = pd.read_csv(file_path)

    if "timestamp" not in intraday_bars_dataframe.columns:
        raise ValueError(f"CSV {csv_path} must have a 'timestamp' column.")

    intraday_bars_dataframe["timestamp"] = pd.to_datetime(
        intraday_bars_dataframe["timestamp"],
        utc=True,
    )

    intraday_bars_dataframe = (
        intraday_bars_dataframe.sort_values("timestamp").reset_index(drop=True)
    )
    return intraday_bars_dataframe


def get_take_profit_percent_for_symbol(symbol: str) -> float:
    return TP_PERCENT_BY_SYMBOL.get(symbol, DEFAULT_TP_PERCENT)


def get_stop_loss_percent_for_symbol(symbol: str) -> float:
    return BRACKET_SL_PERCENT_BY_SYMBOL.get(symbol, DEFAULT_BRACKET_SL_PERCENT)


def get_buy_quantity_for_symbol(symbol: str) -> float:
    """
    Look up the per symbol buy quantity, with a fallback default.
    """
    return BUY_QTY_BY_SYMBOL.get(symbol, DEFAULT_BUY_QTY)


# ------------------------------------------------------------------
# Risk based decisions
# ------------------------------------------------------------------

def compute_risk_based_position_size(
    symbol: str,
    entry_price: float,
    stop_loss_price: float,
    available_cash: float,
    r_per_trade: float,
    starting_cash: float = STARTING_CASH_DEFAULT,
) -> float:
    """
    Compute position size based on risk per trade:
      - risk per share = entry_price - stop_loss_price (for longs)
      - dollar risk allowed = starting_cash * MAX_RISK_FRACTION_PER_TRADE * r_per_trade
      - shares = min(available_cash / entry_price, dollar_risk_allowed / risk_per_share)

    Returns:
      - number of shares to buy (float, but you can round if you like)
    """
    if entry_price <= 0 or stop_loss_price <= 0:
        return 0.0

    risk_per_share = entry_price - stop_loss_price
    if risk_per_share <= 0:
        # No risk or invalid (stop above entry for a long)
        return 0.0

    dollar_risk_allowed = starting_cash * MAX_RISK_FRACTION_PER_TRADE * r_per_trade

    # Shares limited both by cash and by risk
    max_shares_by_cash = available_cash / entry_price
    max_shares_by_risk = dollar_risk_allowed / risk_per_share

    shares = min(max_shares_by_cash, max_shares_by_risk)

    if shares <= 0:
        return 0.0

    # Use whole shares
    return float(int(shares))


# ------------------------------------------------------------------
# Intraday confirmation logic
# ------------------------------------------------------------------

def intraday_confirmation_for_day(
    proposed_limit_price: float,
    daily_timestamp: pd.Timestamp,
    intraday_bars_dataframe: pd.DataFrame,
) -> bool:
    """
    Approximation of your live intraday confirmation:

      - Look at all intraday bars for the same calendar date as `daily_timestamp`.
      - Require at least MIN_INTRADAY_BARS_FOR_CONFIRMATION bars.
      - Use the last close of that day as the "current" intraday price.
      - Confirm that this last intraday close has not pulled back more than
        MAX_INTRADAY_PULLBACK_PCT below the proposed limit price.
    """
    if intraday_bars_dataframe.empty:
        return False

    intraday_bars_dataframe = intraday_bars_dataframe.copy()

    intraday_bars_dataframe["date_only"] = intraday_bars_dataframe["timestamp"].dt.date
    target_date = daily_timestamp.date()

    intraday_slice_for_day = intraday_bars_dataframe[
        intraday_bars_dataframe["date_only"] == target_date
    ]

    if len(intraday_slice_for_day) < MIN_INTRADAY_BARS_FOR_CONFIRMATION:
        return False

    last_intraday_bar = intraday_slice_for_day.iloc[-1]
    last_intraday_close_price = float(last_intraday_bar["close"])

    allowed_minimum_price = proposed_limit_price * (
        1.0 - MAX_INTRADAY_PULLBACK_PCT / 100.0
    )

    if last_intraday_close_price < allowed_minimum_price:
        # Too deep a pullback; confirmation fails
        return False

    return True


# ------------------------------------------------------------------
# Core backtest logic (daily + intraday confirmation)
# ------------------------------------------------------------------

def compute_entry_signal_for_index(
    symbol: str,
    daily_bars_dataframe: pd.DataFrame,
    current_index: int,
) -> Optional[EntrySignal]:
    """
    Daily breakout signal used as the base entry condition.
    """
    bars_up_to_now = daily_bars_dataframe.iloc[: current_index + 1]
    entry_signal = compute_recent_high_breakout_signal(bars_up_to_now)
    return entry_signal


def simulate_backtest_for_symbol_with_intraday(
    symbol: str,
    daily_bars_dataframe: pd.DataFrame,
    intraday_bars_dataframe: pd.DataFrame,
    starting_cash: float = STARTING_CASH_DEFAULT,
) -> Tuple[float, List[BacktestTrade]]:
    """
    Backtest with DAILY breakout plus INTRADAY confirmation:

      - Uses daily breakout logic (compute_recent_high_breakout_signal).
      - Only enters if intraday_confirmation_for_day passes.
      - Entries use the daily bar open for the signal day.
      - Exits via TP or SL based on per symbol percentages, or at MAX_HOLDING_DAYS/end of data.
    """
    cash_balance = starting_cash
    current_position_quantity: float = 0.0
    current_entry_price: Optional[float] = None
    current_entry_date: Optional[pd.Timestamp] = None
    current_take_profit_price: Optional[float] = None
    current_stop_loss_price: Optional[float] = None

    executed_trades: List[BacktestTrade] = []

    num_bars = len(daily_bars_dataframe)
    if num_bars < 2:
        return cash_balance, executed_trades

    take_profit_percent = get_take_profit_percent_for_symbol(symbol)
    stop_loss_percent = get_stop_loss_percent_for_symbol(symbol)

    bar_index = 0
    while bar_index < num_bars - 1:
        current_bar_row = daily_bars_dataframe.iloc[bar_index]
        current_timestamp: pd.Timestamp = current_bar_row["timestamp"]

        # ------------------------------------------------------
        # If we are flat: daily breakout plus intraday confirmation
        # ------------------------------------------------------
        if current_position_quantity == 0.0:
            # 1) Daily breakout signal
            entry_signal = compute_entry_signal_for_index(
                symbol=symbol,
                daily_bars_dataframe=daily_bars_dataframe,
                current_index=bar_index,
            )

            if entry_signal is None:
                bar_index += 1
                continue

            # 2) Intraday confirmation for the same calendar date
            is_confirmed = intraday_confirmation_for_day(
                proposed_limit_price=entry_signal.limit_price,
                daily_timestamp=current_timestamp,
                intraday_bars_dataframe=intraday_bars_dataframe,
            )

            if not is_confirmed:
                bar_index += 1
                continue

            # 3) Define entry price and stop loss
            entry_price = float(current_bar_row["open"])
            stop_loss_price = entry_price * (1.0 - stop_loss_percent / 100.0)

            # 4) Risk sizing: how many shares can we afford
            risk_r_per_trade = RISK_R_PER_TRADE_DEFAULT
            buy_quantity = compute_risk_based_position_size(
                symbol=symbol,
                entry_price=entry_price,
                stop_loss_price=stop_loss_price,
                available_cash=cash_balance,
                r_per_trade=risk_r_per_trade,
                starting_cash=starting_cash,
            )

            if buy_quantity <= 0:
                bar_index += 1
                continue

            required_cash = buy_quantity * entry_price
            if required_cash > cash_balance:
                bar_index += 1
                continue

            # 5) Open the position
            current_position_quantity = buy_quantity
            current_entry_price = entry_price
            current_entry_date = current_timestamp

            current_take_profit_price = entry_price * (
                1.0 + take_profit_percent / 100.0
            )
            current_stop_loss_price = stop_loss_price

            cash_balance -= required_cash

        # ------------------------------------------------------
        # If we are in a position: check exits
        # ------------------------------------------------------
        else:
            assert current_entry_price is not None
            assert current_entry_date is not None
            assert current_take_profit_price is not None
            assert current_stop_loss_price is not None

            high_price = float(current_bar_row["high"])
            low_price = float(current_bar_row["low"])
            close_price = float(current_bar_row["close"])

            exit_price: Optional[float] = None

            if low_price <= current_stop_loss_price:
                exit_price = current_stop_loss_price
            elif high_price >= current_take_profit_price:
                exit_price = current_take_profit_price
            else:
                holding_days = (current_timestamp - current_entry_date).days
                if holding_days >= MAX_HOLDING_DAYS:
                    exit_price = close_price

            if exit_price is not None:
                gross_proceeds = current_position_quantity * exit_price
                cash_balance += gross_proceeds

                pnl_dollars = (
                    (exit_price - current_entry_price) * current_position_quantity
                )
                pnl_percent = (
                    (exit_price - current_entry_price)
                    / current_entry_price
                    * 100.0
                )
                holding_days = (current_timestamp - current_entry_date).days

                executed_trades.append(
                    BacktestTrade(
                        symbol=symbol,
                        entry_date=current_entry_date,
                        exit_date=current_timestamp,
                        entry_price=current_entry_price,
                        exit_price=exit_price,
                        quantity=current_position_quantity,
                        pnl_dollars=pnl_dollars,
                        pnl_percent=pnl_percent,
                        holding_days=holding_days,
                    )
                )

                current_position_quantity = 0.0
                current_entry_price = None
                current_entry_date = None
                current_take_profit_price = None
                current_stop_loss_price = None

        bar_index += 1

    # ------------------------------------------------------
    # If still in a trade at end of data, exit at last close
    # ------------------------------------------------------
    if current_position_quantity > 0.0:
        last_bar_row = daily_bars_dataframe.iloc[-1]
        final_timestamp: pd.Timestamp = last_bar_row["timestamp"]
        final_close_price = float(last_bar_row["close"])

        assert current_entry_price is not None
        assert current_entry_date is not None

        gross_proceeds = current_position_quantity * final_close_price
        cash_balance += gross_proceeds

        pnl_dollars = (
            (final_close_price - current_entry_price) * current_position_quantity
        )
        pnl_percent = (
            (final_close_price - current_entry_price)
            / current_entry_price
            * 100.0
        )
        holding_days = (final_timestamp - current_entry_date).days

        executed_trades.append(
            BacktestTrade(
                symbol=symbol,
                entry_date=current_entry_date,
                exit_date=final_timestamp,
                entry_price=current_entry_price,
                exit_price=final_close_price,
                quantity=current_position_quantity,
                pnl_dollars=pnl_dollars,
                pnl_percent=pnl_percent,
                holding_days=holding_days,
            )
        )

    return cash_balance, executed_trades


# ------------------------------------------------------------------
# Stats + summary printing
# ------------------------------------------------------------------

def compute_trade_statistics(trades: List[BacktestTrade]) -> dict:
    if not trades:
        return {
            "total_pnl": 0.0,
            "total_pnl_pct": 0.0,
            "winner_count": 0,
            "loser_count": 0,
            "win_rate": 0.0,
            "average_win": 0.0,
            "median_win": 0.0,
            "average_loss": 0.0,
            "median_loss": 0.0,
            "average_holding_days": 0.0,
            "median_holding_days": 0.0,
            "min_holding_days": 0,
            "max_holding_days": 0,
        }

    pnl_values = [trade.pnl_dollars for trade in trades]
    total_pnl = sum(pnl_values)

    winner_pnls = [trade.pnl_dollars for trade in trades if trade.pnl_dollars > 0]
    loser_pnls = [trade.pnl_dollars for trade in trades if trade.pnl_dollars < 0]

    holding_days_list = [trade.holding_days for trade in trades]

    def safe_average(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def safe_median(values: List[float]) -> float:
        if not values:
            return 0.0
        sorted_values = sorted(values)
        count = len(sorted_values)
        mid_index = count // 2
        if count % 2 == 1:
            return sorted_values[mid_index]
        return 0.5 * (sorted_values[mid_index - 1] + sorted_values[mid_index])

    winner_count = len(winner_pnls)
    loser_count = len(loser_pnls)
    total_trades = len(trades)

    win_rate = (winner_count / total_trades * 100.0) if total_trades > 0 else 0.0

    stats = {
        "total_pnl": total_pnl,
        "winner_count": winner_count,
        "loser_count": loser_count,
        "win_rate": win_rate,
        "average_win": safe_average(winner_pnls),
        "median_win": safe_median(winner_pnls),
        "average_loss": safe_average(loser_pnls),
        "median_loss": safe_median(loser_pnls),
        "average_holding_days": safe_average(holding_days_list),
        "median_holding_days": safe_median(holding_days_list),
        "min_holding_days": min(holding_days_list) if holding_days_list else 0,
        "max_holding_days": max(holding_days_list) if holding_days_list else 0,
    }

    return stats


def print_backtest_summary(
    symbol: str,
    starting_cash: float,
    ending_cash: float,
    trades: List[BacktestTrade],
):
    stats = compute_trade_statistics(trades)

    total_pnl = stats["total_pnl"]
    total_pnl_pct = (total_pnl / starting_cash * 100.0) if starting_cash > 0 else 0.0

    print("==================== BACKTEST (INTRADAY) SUMMARY ====================")
    print(f"Symbol:            {symbol}")
    print(f"Starting cash:     {starting_cash:,.2f}")
    print(f"Ending cash:       {ending_cash:,.2f}")
    print(f"Total PnL:         {total_pnl:,.2f} ({total_pnl_pct:.2f}%)")
    print(f"Trades:            {len(trades)}")
    print(f"Winners:           {stats['winner_count']}")
    print(f"Losers:            {stats['loser_count']}")
    print(f"Win rate:          {stats['win_rate']:.2f}%")
    print()
    print(f"Average win:       {stats['average_win']:.2f}")
    print(f"Median win:        {stats['median_win']:.2f}")
    print(f"Average loss:      {stats['average_loss']:.2f}")
    print(f"Median loss:       {stats['median_loss']:.2f}")
    print()
    print(f"Avg holding days:  {stats['average_holding_days']:.2f}")
    print(f"Median holding:    {stats['median_holding_days']:.2f}")
    print(
        f"Min holding:       {stats['min_holding_days']} days"
        if stats["min_holding_days"] is not None
        else "Min holding:       n/a"
    )
    print(
        f"Max holding:       {stats['max_holding_days']} days"
        if stats["max_holding_days"] is not None
        else "Max holding:       n/a"
    )
    print("=========================================================")


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

def run_backtest_for_symbol_with_intraday(symbol: str, starting_cash: float = STARTING_CASH_DEFAULT):
    daily_csv_path = DAILY_CSV_PATH_TEMPLATE.format(symbol=symbol)
    intraday_csv_path = INTRADAY_CSV_PATH_TEMPLATE.format(symbol=symbol)

    daily_bars_dataframe = load_daily_bars_from_csv(daily_csv_path)
    intraday_bars_dataframe = load_intraday_bars_from_csv(intraday_csv_path)

    ending_cash, trades = simulate_backtest_for_symbol_with_intraday(
        symbol=symbol,
        daily_bars_dataframe=daily_bars_dataframe,
        intraday_bars_dataframe=intraday_bars_dataframe,
        starting_cash=starting_cash,
    )

    print_backtest_summary(
        symbol=symbol,
        starting_cash=starting_cash,
        ending_cash=ending_cash,
        trades=trades,
    )

    trades_output_path = Path(f"data/{symbol}_backtest_intraday_trades.csv")
    if trades:
        trades_dataframe = pd.DataFrame([trade.__dict__ for trade in trades])
        trades_dataframe.to_csv(trades_output_path, index=False)
        print(f"[backtest_intraday] Wrote {len(trades)} trades to {trades_output_path}")


if __name__ == "__main__":
    run_backtest_for_symbol_with_intraday(
        symbol=SYMBOL_TO_TEST,
        starting_cash=STARTING_CASH_DEFAULT,
    )
