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
    ATR_PERIOD_DEFAULT,
    ATR_TP_MULTIPLIER_DEFAULT,
    ATR_SL_MULTIPLIER_DEFAULT,
    RISK_R_PER_TRADE_DEFAULT,
)

from risk_sizing import compute_risk_based_position_size

import math

from signals import (
    compute_recent_high_breakout_signal,
    compute_sma_trend_entry_signal,
    EntrySignal,
    compute_entry_signal_for_mode,
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

# How much of account equity to risk per trade (as a fraction, e.g. 0.01 = 1%)
RISK_FRACTION_PER_TRADE = 0.01

# Optional: cap each position to at most this fraction of equity
MAX_POSITION_FRACTION_OF_EQUITY = 0.20  # 20% of equity in any one trade

# ATR-based exit configuration
ATR_LOOKBACK_DAYS = 14          # standard ATR period
ATR_STOP_MULTIPLIER = 1.5       # stop-loss = entry_price - 1.5 * ATR
ATR_TP_MULTIPLIER = 3.0         # take-profit = entry_price + 3.0 * ATR

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

def compute_atr(daily_bars_dataframe: pd.DataFrame, period: int) -> pd.Series:
    """
    Compute Average True Range (ATR) over the given period.
    Returns a pandas Series aligned with daily_bars_dataframe.index.
    """
    high = daily_bars_dataframe["high"].astype(float)
    low = daily_bars_dataframe["low"].astype(float)
    close = daily_bars_dataframe["close"].astype(float)

    previous_close = close.shift(1)

    true_range_1 = high - low
    true_range_2 = (high - previous_close).abs()
    true_range_3 = (low - previous_close).abs()

    true_range = pd.concat(
        [true_range_1, true_range_2, true_range_3],
        axis=1,
    ).max(axis=1)

    atr = true_range.rolling(window=period, min_periods=period).mean()
    return atr


def get_take_profit_percent_for_symbol(symbol: str) -> float:
    return TP_PERCENT_BY_SYMBOL.get(symbol, DEFAULT_TP_PERCENT)


def get_stop_loss_percent_for_symbol(symbol: str) -> float:
    return BRACKET_SL_PERCENT_BY_SYMBOL.get(symbol, DEFAULT_BRACKET_SL_PERCENT)


def get_buy_quantity_for_symbol(symbol: str) -> float:
    return BUY_QTY_BY_SYMBOL.get(symbol, DEFAULT_BUY_QTY)

def compute_risk_based_buy_quantity(
    entry_price: float,
    stop_loss_percent: float,
    current_equity: float,
) -> int:
    """
    Compute how many shares to buy based on:
      - entry_price
      - stop_loss_percent (e.g. 2.0 means SL is 2% below entry)
      - current_equity (cash + any open position value; when flat, this is just cash)

    We risk at most RISK_FRACTION_PER_TRADE * current_equity on the trade,
    and optionally cap the position size to MAX_POSITION_FRACTION_OF_EQUITY * current_equity.
    """
    if entry_price <= 0 or stop_loss_percent <= 0 or current_equity <= 0:
        return 0

    # Distance from entry to stop in dollars per share
    stop_loss_price = entry_price * (1.0 - stop_loss_percent / 100.0)
    risk_per_share = entry_price - stop_loss_price
    if risk_per_share <= 0:
        return 0

    # How many dollars of equity we are willing to risk on this trade
    max_risk_dollars = current_equity * RISK_FRACTION_PER_TRADE

    # Shares limited by risk
    max_shares_by_risk = max_risk_dollars / risk_per_share

    # Also cap by not putting too much *capital* in one trade
    max_position_dollars = current_equity * MAX_POSITION_FRACTION_OF_EQUITY
    max_shares_by_capital = max_position_dollars / entry_price

    # Take the stricter of the two caps
    raw_shares = min(max_shares_by_risk, max_shares_by_capital)

    # You can only trade whole shares in this backtest
    shares_to_buy = int(raw_shares)

    # Never return negative
    if shares_to_buy < 0:
        return 0

    return shares_to_buy

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

    # Single dispatcher call into the signals module
    return compute_entry_signal_for_mode(bars_up_to_now, mode=ENTRY_SIGNAL_MODE)

def add_atr_column(
    daily_bars_dataframe: pd.DataFrame,
    atr_lookback_days: int = ATR_LOOKBACK_DAYS,
) -> pd.DataFrame:
    """
    Compute a simple ATR (Average True Range) column and attach it to the
    provided daily bars DataFrame.

    ATR is based on:
      TR = max(
        high - low,
        abs(high - previous_close),
        abs(low - previous_close)
      )

    We then take a rolling mean of TR over atr_lookback_days.
    """
    bars_with_atr = daily_bars_dataframe.copy()

    bars_with_atr["previous_close"] = bars_with_atr["close"].shift(1)

    true_range_1 = bars_with_atr["high"] - bars_with_atr["low"]
    true_range_2 = (bars_with_atr["high"] - bars_with_atr["previous_close"]).abs()
    true_range_3 = (bars_with_atr["low"] - bars_with_atr["previous_close"]).abs()

    bars_with_atr["true_range"] = pd.concat(
        [true_range_1, true_range_2, true_range_3],
        axis=1,
    ).max(axis=1)

    bars_with_atr["atr"] = (
        bars_with_atr["true_range"]
        .rolling(window=atr_lookback_days, min_periods=atr_lookback_days)
        .mean()
    )

    return bars_with_atr


def simulate_backtest_for_symbol_daily(
    symbol: str,
    daily_bars_dataframe: pd.DataFrame,
    starting_cash: float = STARTING_CASH_DEFAULT,
) -> Tuple[float, List[BacktestTrade]]:
    """
    Daily-only backtest with ATR-based TP/SL and risk-based position sizing:

      - Uses the same daily entry logic (breakout or SMA trend).
      - Enters at NEXT DAY'S OPEN after the signal bar.
      - Uses ATR-based stop-loss and take-profit:
          stop_loss  = entry_price - ATR_SL_MULTIPLIER_DEFAULT * ATR
          take_profit = entry_price + ATR_TP_MULTIPLIER_DEFAULT * ATR
      - Position size is computed via compute_risk_based_position_size,
        risking RISK_R_PER_TRADE_DEFAULT * available_cash per trade.
      - Exits via TP, SL, time-based (MAX_HOLDING_DAYS), or at end-of-data.
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

    # Precompute ATR for ATR-based TP/SL
    atr_series = compute_atr(daily_bars_dataframe, ATR_PERIOD_DEFAULT)

    bar_index = 0
    while bar_index < num_bars - 1:
        current_bar_row = daily_bars_dataframe.iloc[bar_index]
        next_bar_row = daily_bars_dataframe.iloc[bar_index + 1]

        current_timestamp: pd.Timestamp = current_bar_row["timestamp"]
        current_atr = float(atr_series.iloc[bar_index]) if not pd.isna(atr_series.iloc[bar_index]) else None

        # ------------------------------------------------------
        # If we are flat: look for an entry signal
        # ------------------------------------------------------
        if current_position_quantity == 0.0:
            # We need ATR to be available
            if current_atr is None:
                bar_index += 1
                continue

            entry_signal = compute_entry_signal_for_index(
                symbol=symbol,
                daily_bars_dataframe=daily_bars_dataframe,
                current_index=bar_index,
            )

            if entry_signal is None:
                bar_index += 1
                continue

            # Enter at NEXT day's open
            entry_price = float(next_bar_row["open"])

            # ATR-based stop loss and take profit
            stop_loss_price = entry_price - ATR_SL_MULTIPLIER_DEFAULT * current_atr
            take_profit_price = entry_price + ATR_TP_MULTIPLIER_DEFAULT * current_atr

            # Risk-based position sizing
            risk_r_per_trade = RISK_R_PER_TRADE_DEFAULT

            buy_quantity = compute_risk_based_position_size(
                entry_price=entry_price,
                stop_loss_price=stop_loss_price,
                available_cash=cash_balance,
                r_per_trade=risk_r_per_trade,
            )

            if buy_quantity <= 0:
                bar_index += 1
                continue

            required_cash = buy_quantity * entry_price
            if required_cash > cash_balance:
                bar_index += 1
                continue

            # Open the position
            current_position_quantity = buy_quantity
            current_entry_price = entry_price
            current_entry_date = next_bar_row["timestamp"]
            current_take_profit_price = take_profit_price
            current_stop_loss_price = stop_loss_price

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

            # Stop-loss first
            if low_price <= current_stop_loss_price:
                exit_price = current_stop_loss_price
            # Then take-profit
            elif high_price >= current_take_profit_price:
                exit_price = current_take_profit_price
            else:
                # Time-based exit: holding too long?
                holding_days = (current_timestamp - current_entry_date).days
                if holding_days >= MAX_HOLDING_DAYS:
                    exit_price = close_price

            # If we have an exit event, realize the trade
            if exit_price is not None:
                gross_proceeds = current_position_quantity * exit_price
                cash_balance += gross_proceeds

                pnl_dollars = (exit_price - current_entry_price) * current_position_quantity
                pnl_percent = (exit_price - current_entry_price) / current_entry_price * 100.0
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

        pnl_dollars = (final_close_price - current_entry_price) * current_position_quantity
        pnl_percent = (final_close_price - current_entry_price) / current_entry_price * 100.0
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
