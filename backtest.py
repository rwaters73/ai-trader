from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from pathlib import Path
from datetime import timedelta

import pandas as pd

from config import (
    TP_PERCENT_BY_SYMBOL,
    DEFAULT_TP_PERCENT,
    BRACKET_SL_PERCENT_BY_SYMBOL,
    DEFAULT_BRACKET_SL_PERCENT,
    BUY_QTY_BY_SYMBOL,
    DEFAULT_BUY_QTY,
)

from signals import (
    compute_recent_high_breakout_signal,
    compute_sma_trend_entry_signal,
    EntrySignal,
)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--symbol", type=str, default="TSLA")
parser.add_argument("--signal", type=str, default="breakout") #breakout or sma_trend
args = parser.parse_args()


# ------------------------------------------------------------------
# Configuration – change this symbol to backtest a different ticker
# ------------------------------------------------------------------
SYMBOL_TO_TEST = args.symbol
#SYMBOL_TO_TEST = "MNDR"
DAILY_CSV_PATH_TEMPLATE = "data/{symbol}_daily.csv"

STARTING_CASH_DEFAULT = 100000.0
MAX_HOLDING_DAYS = 30  # simple safety cap on how long we hold a position

# Which entry signal function to use: "breakout" or "sma_trend"
ENTRY_SIGNAL_MODE = args.signal
#ENTRY_SIGNAL_MODE = "breakout"
# To try the SMA trend strategy instead, change this to:
# ENTRY_SIGNAL_MODE = "sma_trend"


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
    """
    Load daily OHLCV bars from a CSV produced by download_history.py.
    Ensures:
      - 'timestamp' is parsed as datetime
      - sorted ascending by timestamp
    """
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


def get_take_profit_percent_for_symbol(symbol: str) -> float:
    return TP_PERCENT_BY_SYMBOL.get(symbol, DEFAULT_TP_PERCENT)


def get_stop_loss_percent_for_symbol(symbol: str) -> float:
    return BRACKET_SL_PERCENT_BY_SYMBOL.get(symbol, DEFAULT_BRACKET_SL_PERCENT)


def get_buy_quantity_for_symbol(symbol: str) -> float:
    return BUY_QTY_BY_SYMBOL.get(symbol, DEFAULT_BUY_QTY)


# ------------------------------------------------------------------
# Core backtest logic (daily-only)
# ------------------------------------------------------------------

def compute_entry_signal_for_index(
    symbol: str,
    daily_bars_dataframe: pd.DataFrame,
    current_index: int,
) -> Optional[EntrySignal]:
    """
    For a given bar index, slice the daily bars up to *and including* that bar,
    and run the chosen daily entry logic (breakout or SMA trend).
    """
    bars_up_to_now = daily_bars_dataframe.iloc[: current_index + 1]

    if bars_up_to_now.empty:
        return None

    if ENTRY_SIGNAL_MODE == "breakout":
        return compute_recent_high_breakout_signal(bars_up_to_now)
    elif ENTRY_SIGNAL_MODE == "sma_trend":
        return compute_sma_trend_entry_signal(bars_up_to_now)
    else:
        raise ValueError(f"Unknown ENTRY_SIGNAL_MODE: {ENTRY_SIGNAL_MODE}")


def simulate_backtest_for_symbol_daily(
    symbol: str,
    daily_bars_dataframe: pd.DataFrame,
    starting_cash: float = STARTING_CASH_DEFAULT,
) -> Tuple[float, List[BacktestTrade]]:
    """
    Daily-only backtest:

      - Uses the chosen daily entry signal:
          * compute_recent_high_breakout_signal  (ENTRY_SIGNAL_MODE = 'breakout')
          * compute_sma_trend_entry_signal       (ENTRY_SIGNAL_MODE = 'sma_trend')
      - Buys a fixed quantity from BUY_QTY_BY_SYMBOL / DEFAULT_BUY_QTY when signal is present.
      - Enters at NEXT DAY'S OPEN after the signal bar.
      - Exits via TP or SL based on per-symbol percentages.
      - Also forces exit after MAX_HOLDING_DAYS or at end-of-data.
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
        # Need at least two bars to have "signal day" + "entry day"
        return cash_balance, executed_trades

    take_profit_percent = get_take_profit_percent_for_symbol(symbol)
    stop_loss_percent = get_stop_loss_percent_for_symbol(symbol)
    buy_quantity = get_buy_quantity_for_symbol(symbol)

    bar_index = 0
    while bar_index < num_bars - 1:
        current_bar_row = daily_bars_dataframe.iloc[bar_index]
        next_bar_row = daily_bars_dataframe.iloc[bar_index + 1]

        current_timestamp: pd.Timestamp = current_bar_row["timestamp"]

        # ------------------------------------------------------
        # If we are flat: look for a daily entry signal
        # ------------------------------------------------------
        if current_position_quantity == 0.0:
            entry_signal = compute_entry_signal_for_index(
                symbol=symbol,
                daily_bars_dataframe=daily_bars_dataframe,
                current_index=bar_index,
            )

            if entry_signal is not None:
                # Enter at NEXT day open, assuming we have enough cash.
                entry_price = float(next_bar_row["open"])
                required_cash = buy_quantity * entry_price

                if required_cash <= cash_balance:
                    current_position_quantity = buy_quantity
                    current_entry_price = entry_price
                    current_entry_date = next_bar_row["timestamp"]

                    current_take_profit_price = current_entry_price * (
                        1.0 + take_profit_percent / 100.0
                    )
                    current_stop_loss_price = current_entry_price * (
                        1.0 - stop_loss_percent / 100.0
                    )

                    cash_balance -= required_cash

        # ------------------------------------------------------
        # If we are in a position: check exit conditions
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
            exit_reason = ""

            # Stop-loss first
            if low_price <= current_stop_loss_price:
                exit_price = current_stop_loss_price
                exit_reason = "stop_loss"
            # Then take-profit
            elif high_price >= current_take_profit_price:
                exit_price = current_take_profit_price
                exit_reason = "take_profit"
            else:
                # Time-based exit: holding too long?
                holding_days = (current_timestamp - current_entry_date).days
                if holding_days >= MAX_HOLDING_DAYS:
                    exit_price = close_price
                    exit_reason = "time_exit"

            # If we have an exit event, realize the trade
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

                # Reset position
                current_position_quantity = 0.0
                current_entry_price = None
                current_entry_date = None
                current_take_profit_price = None
                current_stop_loss_price = None

        bar_index += 1

    # ------------------------------------------------------
    # If still in a trade at end-of-data, exit at last close
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

    print("==================== BACKTEST SUMMARY ====================")
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

def run_backtest_for_symbol(symbol: str, starting_cash: float = STARTING_CASH_DEFAULT):
    daily_csv_path = DAILY_CSV_PATH_TEMPLATE.format(symbol=symbol)
    daily_bars_dataframe = load_daily_bars_from_csv(daily_csv_path)

    ending_cash, trades = simulate_backtest_for_symbol_daily(
        symbol=symbol,
        daily_bars_dataframe=daily_bars_dataframe,
        starting_cash=starting_cash,
    )

    print_backtest_summary(
        symbol=symbol,
        starting_cash=starting_cash,
        ending_cash=ending_cash,
        trades=trades,
    )

    # Optional: write trades to CSV for inspection
    trades_output_path = Path(f"data/{symbol}_backtest_trades.csv")
    if trades:
        trades_dataframe = pd.DataFrame([trade.__dict__ for trade in trades])
        trades_dataframe.to_csv(trades_output_path, index=False)
        print(f"[backtest] Wrote {len(trades)} trades to {trades_output_path}")


if __name__ == "__main__":
    run_backtest_for_symbol(symbol=SYMBOL_TO_TEST, starting_cash=STARTING_CASH_DEFAULT)
