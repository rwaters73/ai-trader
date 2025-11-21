from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from datetime import datetime, time

import os
import pandas as pd

from signals import compute_recent_high_breakout_signal, EntrySignal
from config import (
    BUY_QTY_BY_SYMBOL,
    DEFAULT_BUY_QTY,
    TP_PERCENT_BY_SYMBOL,
    DEFAULT_TP_PERCENT,
    MAX_INTRADAY_PULLBACK_PCT,
    MIN_INTRADAY_BARS_FOR_CONFIRMATION,
)

# ---------------------------------------------------------------------------
# Helpers to read config for a single symbol
# ---------------------------------------------------------------------------

def get_buy_quantity_for_symbol(symbol: str) -> float:
    return BUY_QTY_BY_SYMBOL.get(symbol, DEFAULT_BUY_QTY)


def get_take_profit_percent_for_symbol(symbol: str) -> float:
    return TP_PERCENT_BY_SYMBOL.get(symbol, DEFAULT_TP_PERCENT)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_daily_bars_from_csv(csv_path: str, symbol: str) -> pd.DataFrame:
    """
    Load DAILY bars from CSV into a DataFrame indexed by timestamp.

    Assumes columns at least: timestamp, open, high, low, close, volume
    and optionally 'symbol'. If 'symbol' exists, we filter to the given symbol.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Daily CSV not found at: {csv_path}")

    daily_bars_dataframe = pd.read_csv(csv_path, parse_dates=["timestamp"])

    # If there is a 'symbol' column, filter by it
    if "symbol" in daily_bars_dataframe.columns:
        daily_bars_dataframe = daily_bars_dataframe[
            daily_bars_dataframe["symbol"] == symbol
        ]

    daily_bars_dataframe = daily_bars_dataframe.set_index("timestamp")
    daily_bars_dataframe = daily_bars_dataframe.sort_index()

    return daily_bars_dataframe


def load_intraday_bars_from_csv(csv_path: str, symbol: str) -> pd.DataFrame:
    """
    Load INTRADAY bars from CSV into a DataFrame indexed by timestamp.

    Same assumptions as daily: timestamp column and optionally 'symbol'.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Intraday CSV not found at: {csv_path}")

    intraday_bars_dataframe = pd.read_csv(csv_path, parse_dates=["timestamp"])

    if "symbol" in intraday_bars_dataframe.columns:
        intraday_bars_dataframe = intraday_bars_dataframe[
            intraday_bars_dataframe["symbol"] == symbol
        ]

    intraday_bars_dataframe = intraday_bars_dataframe.set_index("timestamp")
    intraday_bars_dataframe = intraday_bars_dataframe.sort_index()

    return intraday_bars_dataframe


# ---------------------------------------------------------------------------
# Intraday confirmation logic (CSV-based)
# ---------------------------------------------------------------------------

def confirm_entry_with_intraday(
    symbol: str,
    trading_day: pd.Timestamp,
    proposed_limit_price: float,
    intraday_bars_dataframe: pd.DataFrame,
) -> bool:
    """
    Given a daily breakout entry (with proposed_limit_price) and a trading day,
    confirm the entry using INTRADAY bars:

      - Look at intraday bars for that calendar day between 08:30 and 15:00.
      - Require at least MIN_INTRADAY_BARS_FOR_CONFIRMATION bars.
      - Check that the last intraday close is not more than
        MAX_INTRADAY_PULLBACK_PCT below the proposed limit price.

    Returns True if intraday confirmation PASSES, otherwise False.
    """
    # Slice intraday bars to that calendar day (RTH window for simplicity)
    day_start = trading_day.normalize().replace(hour=8, minute=30, second=0, microsecond=0)
    day_end = trading_day.normalize().replace(hour=15, minute=0, second=0, microsecond=0)

    day_intraday = intraday_bars_dataframe.loc[day_start:day_end]

    if day_intraday.empty:
        print(f"[intraday] {symbol} {trading_day.date()}: no intraday bars for this day.")
        return False

    if len(day_intraday) < MIN_INTRADAY_BARS_FOR_CONFIRMATION:
        print(
            f"[intraday] {symbol} {trading_day.date()}: "
            f"only {len(day_intraday)} bars, need at least "
            f"{MIN_INTRADAY_BARS_FOR_CONFIRMATION} for confirmation."
        )
        return False

    last_intraday_close_price = float(day_intraday["close"].iloc[-1])

    allowed_minimum_price = proposed_limit_price * (
        1.0 - MAX_INTRADAY_PULLBACK_PCT / 100.0
    )

    if last_intraday_close_price < allowed_minimum_price:
        pullback_percent = (
            (proposed_limit_price - last_intraday_close_price)
            / proposed_limit_price
        ) * 100.0

        print(
            f"[intraday] {symbol} {trading_day.date()}: "
            f"confirmation FAILED. last_close={last_intraday_close_price:.2f} "
            f"is {pullback_percent:.2f}% below proposed limit "
            f"{proposed_limit_price:.2f}, which exceeds "
            f"MAX_INTRADAY_PULLBACK_PCT={MAX_INTRADAY_PULLBACK_PCT:.2f}%."
        )
        return False

    print(
        f"[intraday] {symbol} {trading_day.date()}: confirmation PASSED. "
        f"last_close={last_intraday_close_price:.2f} is within "
        f"{MAX_INTRADAY_PULLBACK_PCT:.2f}% of proposed limit "
        f"{proposed_limit_price:.2f}."
    )
    return True


# ---------------------------------------------------------------------------
# Backtest trade record
# ---------------------------------------------------------------------------

@dataclass
class BacktestTrade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: float
    pnl_dollars: float
    holding_days: int


# ---------------------------------------------------------------------------
# Core backtest
# ---------------------------------------------------------------------------

def run_backtest_with_intraday(
    symbol: str,
    daily_csv_path: str,
    intraday_csv_path: str,
    starting_cash: float = 100_000.0,
) -> None:
    """
    Backtest that uses:

      - DAILY breakout signal (same logic as live via signals.compute_recent_high_breakout_signal)
      - Intraday confirmation based on TSLA_intraday.csv
      - Fixed buy quantity from config
      - Take-profit percent from config

    Exits are via TP only (no SL here; we are mirroring the simple harness).
    """

    daily_bars = load_daily_bars_from_csv(daily_csv_path, symbol)
    intraday_bars = load_intraday_bars_from_csv(intraday_csv_path, symbol)

    if daily_bars.empty:
        print(f"[backtest] No daily bars for {symbol} at {daily_csv_path}")
        return

    print(f"[backtest] Loaded {len(daily_bars)} daily bars for {symbol}.")
    print(f"[backtest] Loaded {len(intraday_bars)} intraday bars for {symbol}.")

    current_cash = starting_cash
    current_quantity = 0.0
    current_entry_price: Optional[float] = None
    current_entry_date: Optional[pd.Timestamp] = None

    trades: List[BacktestTrade] = []

    take_profit_percent = get_take_profit_percent_for_symbol(symbol)
    buy_quantity = get_buy_quantity_for_symbol(symbol)

    # We will iterate over daily bars in chronological order
    daily_index = daily_bars.index

    for row_index in range(len(daily_bars)):
        current_timestamp = daily_index[row_index]
        current_row = daily_bars.iloc[row_index]
        current_close_price = float(current_row["close"])

        # For signals, we want a history window up to *and including* today
        daily_window = daily_bars.iloc[: row_index + 1]

        # If we have an open position, check TP exit first
        if current_quantity > 0 and current_entry_price is not None:
            take_profit_level = current_entry_price * (1.0 + take_profit_percent / 100.0)

            if current_close_price >= take_profit_level:
                # Exit at today's close
                pnl_dollars = (current_close_price - current_entry_price) * current_quantity
                holding_days = (current_timestamp - current_entry_date).days  # type: ignore

                current_cash += current_close_price * current_quantity

                trades.append(
                    BacktestTrade(
                        entry_date=current_entry_date,      # type: ignore
                        exit_date=current_timestamp,
                        entry_price=current_entry_price,
                        exit_price=current_close_price,
                        quantity=current_quantity,
                        pnl_dollars=pnl_dollars,
                        holding_days=holding_days,
                    )
                )

                print(
                    f"[exit] {symbol} {current_timestamp.date()} TP hit: "
                    f"entry={current_entry_price:.2f}, exit={current_close_price:.2f}, "
                    f"qty={current_quantity}, pnl={pnl_dollars:.2f}"
                )

                # Flat after exit
                current_quantity = 0.0
                current_entry_price = None
                current_entry_date = None

        # If we are flat, consider new entry
        if current_quantity == 0.0:
            if len(daily_window) < 30:
                # Same MIN_BARS_FOR_SIGNAL logic as in signals, but we let
                # compute_recent_high_breakout_signal enforce its own limit too.
                continue

            entry_signal: Optional[EntrySignal] = compute_recent_high_breakout_signal(
                daily_window
            )

            if entry_signal is None:
                continue

            proposed_limit_price = entry_signal.limit_price

            # Confirm with intraday bars for this trading day
            did_confirm = confirm_entry_with_intraday(
                symbol=symbol,
                trading_day=current_timestamp,
                proposed_limit_price=proposed_limit_price,
                intraday_bars_dataframe=intraday_bars,
            )

            if not did_confirm:
                continue

            # Entry: buy fixed quantity at today's close (approximation)
            total_cost = current_close_price * buy_quantity
            if total_cost > current_cash:
                print(
                    f"[entry] {symbol} {current_timestamp.date()}: "
                    f"Not enough cash to buy {buy_quantity} at {current_close_price:.2f}"
                )
                continue

            current_cash -= total_cost
            current_quantity = buy_quantity
            current_entry_price = current_close_price
            current_entry_date = current_timestamp

            print(
                f"[entry] {symbol} {current_timestamp.date()}: "
                f"ENTER long {buy_quantity} at close={current_close_price:.2f}. "
                f"Signal reason: {entry_signal.reason}"
            )

    # After loop, if we are still in a position, close at last close for simplicity
    if current_quantity > 0 and current_entry_price is not None:
        last_row = daily_bars.iloc[-1]
        last_timestamp = daily_bars.index[-1]
        last_close_price = float(last_row["close"])

        pnl_dollars = (last_close_price - current_entry_price) * current_quantity
        holding_days = (last_timestamp - current_entry_date).days  # type: ignore

        current_cash += last_close_price * current_quantity

        trades.append(
            BacktestTrade(
                entry_date=current_entry_date,      # type: ignore
                exit_date=last_timestamp,
                entry_price=current_entry_price,
                exit_price=last_close_price,
                quantity=current_quantity,
                pnl_dollars=pnl_dollars,
                holding_days=holding_days,
            )
        )

        print(
            f"[final-exit] {symbol} {last_timestamp.date()}: "
            f"Forced exit at end of data: entry={current_entry_price:.2f}, "
            f"exit={last_close_price:.2f}, qty={current_quantity}, "
            f"pnl={pnl_dollars:.2f}"
        )

    # -------------------------------------------------------------------
    # Summary stats
    # -------------------------------------------------------------------

    total_pnl = sum(trade.pnl_dollars for trade in trades)
    ending_cash = current_cash
    total_pnl_pct = (ending_cash - starting_cash) / starting_cash * 100.0

    winning_trades = [trade for trade in trades if trade.pnl_dollars > 0]
    losing_trades = [trade for trade in trades if trade.pnl_dollars <= 0]

    number_of_trades = len(trades)
    number_of_winners = len(winning_trades)
    number_of_losers = len(losing_trades)
    win_rate = (number_of_winners / number_of_trades * 100.0) if number_of_trades > 0 else 0.0

    holding_days_list = [trade.holding_days for trade in trades]

    print("\n==================== INTRADAY BACKTEST SUMMARY ====================")
    print(f"Symbol:            {symbol}")
    print(f"Starting cash:     {starting_cash:,.2f}")
    print(f"Ending cash:       {ending_cash:,.2f}")
    print(f"Total PnL:         {total_pnl:,.2f} ({total_pnl_pct:.2f}%)")
    print(f"Trades:            {number_of_trades}")
    print(f"Winners:           {number_of_winners}")
    print(f"Losers:            {number_of_losers}")
    print(f"Win rate:          {win_rate:.2f}%")

    if winning_trades:
        average_win = sum(trade.pnl_dollars for trade in winning_trades) / len(winning_trades)
        median_win = float(pd.Series([trade.pnl_dollars for trade in winning_trades]).median())
        print(f"\nAverage win:       {average_win:.2f}")
        print(f"Median win:        {median_win:.2f}")

    if losing_trades:
        average_loss = sum(trade.pnl_dollars for trade in losing_trades) / len(losing_trades)
        median_loss = float(pd.Series([trade.pnl_dollars for trade in losing_trades]).median())
        print(f"Average loss:      {average_loss:.2f}")
        print(f"Median loss:       {median_loss:.2f}")

    if holding_days_list:
        average_holding = sum(holding_days_list) / len(holding_days_list)
        median_holding = float(pd.Series(holding_days_list).median())
        print(f"\nAvg holding days:  {average_holding:.2f}")
        print(f"Median holding:    {median_holding:.2f} days")
        print(f"Min holding:       {min(holding_days_list)} days")
        print(f"Max holding:       {max(holding_days_list)} days")

    print("===================================================================")

    # Optionally write trades to CSV for inspection
    output_path = os.path.join("data", f"{symbol}_intraday_backtest_trades.csv")
    trades_dataframe = pd.DataFrame([trade.__dict__ for trade in trades])
    trades_dataframe.to_csv(output_path, index=False)
    print(f"[backtest] Wrote {len(trades)} trades to {os.path.abspath(output_path)}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tsla_daily_path = os.path.join("data", "TSLA_daily.csv")
    tsla_intraday_path = os.path.join("data", "TSLA_intraday_5min.csv")

    run_backtest_with_intraday(
        symbol="TSLA",
        daily_csv_path=tsla_daily_path,
        intraday_csv_path=tsla_intraday_path,
        starting_cash=100000.0,
    )
