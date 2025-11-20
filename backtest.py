from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from pathlib import Path

import pandas as pd

from config import (
    TP_PERCENT_BY_SYMBOL,
    DEFAULT_TP_PERCENT,
    BRACKET_SL_PERCENT_BY_SYMBOL,
    DEFAULT_BRACKET_SL_PERCENT,
    BUY_QTY_BY_SYMBOL,
    DEFAULT_BUY_QTY,
)
from signals import compute_recent_high_breakout_signal, EntrySignal


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BacktestTrade:
    symbol: str
    entry_timestamp: pd.Timestamp
    entry_price: float
    exit_timestamp: pd.Timestamp
    exit_price: float
    quantity: float
    entry_reason: str
    exit_reason: str

    @property
    def pnl_dollars(self) -> float:
        return (self.exit_price - self.entry_price) * self.quantity

    @property
    def pnl_percent(self) -> float:
        if self.entry_price <= 0:
            return 0.0
        return (self.exit_price - self.entry_price) / self.entry_price * 100.0
    
    @property
    def holding_days(self) -> int:
        """How many days the trade was held (entry → exit)."""
        return (self.exit_timestamp - self.entry_timestamp).days


@dataclass
class BacktestSummary:
    symbol: str
    starting_cash: float
    ending_cash: float
    trades: List[BacktestTrade]

    @property
    def total_pnl_dollars(self) -> float:
        return self.ending_cash - self.starting_cash

    @property
    def total_pnl_percent(self) -> float:
        if self.starting_cash <= 0:
            return 0.0
        return (self.ending_cash - self.starting_cash) / self.starting_cash * 100.0

    @property
    def winning_trades(self) -> List[BacktestTrade]:
        return [trade for trade in self.trades if trade.pnl_dollars > 0]

    @property
    def losing_trades(self) -> List[BacktestTrade]:
        return [trade for trade in self.trades if trade.pnl_dollars <= 0]

    @property
    def winning_trades_count(self) -> int:
        return len(self.winning_trades)

    @property
    def losing_trades_count(self) -> int:
        return len(self.losing_trades)

    @property
    def win_rate_percent(self) -> float:
        total_trades = len(self.trades)
        if total_trades == 0:
            return 0.0
        return self.winning_trades_count / total_trades * 100.0

    @property
    def average_win_percent(self) -> float:
        if not self.winning_trades:
            return 0.0
        return sum(t.pnl_percent for t in self.winning_trades) / len(self.winning_trades)

    @property
    def average_loss_percent(self) -> float:
        if not self.losing_trades:
            return 0.0
        return sum(t.pnl_percent for t in self.losing_trades) / len(self.losing_trades)

    @property
    def best_trade_percent(self) -> float:
        if not self.trades:
            return 0.0
        return max(t.pnl_percent for t in self.trades)

    @property
    def worst_trade_percent(self) -> float:
        if not self.trades:
            return 0.0
        return min(t.pnl_percent for t in self.trades)



# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_take_profit_percent_for_symbol(symbol: str) -> float:
    return TP_PERCENT_BY_SYMBOL.get(symbol, DEFAULT_TP_PERCENT)


def get_stop_loss_percent_for_symbol(symbol: str) -> float:
    return BRACKET_SL_PERCENT_BY_SYMBOL.get(symbol, DEFAULT_BRACKET_SL_PERCENT)


def get_buy_quantity_for_symbol(symbol: str) -> float:
    return BUY_QTY_BY_SYMBOL.get(symbol, DEFAULT_BUY_QTY)


def load_daily_bars_from_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load the CSV created by tools/download_history.py.

    Ensures:
      - 'timestamp' is a datetime index.
      - rows are sorted by timestamp ascending.
    """
    daily_bars_dataframe = pd.read_csv(csv_path)

    if "timestamp" in daily_bars_dataframe.columns:
        daily_bars_dataframe["timestamp"] = pd.to_datetime(
            daily_bars_dataframe["timestamp"], utc=True
        )
        daily_bars_dataframe = daily_bars_dataframe.set_index("timestamp")

    daily_bars_dataframe = daily_bars_dataframe.sort_index()
    return daily_bars_dataframe


def write_trades_to_csv(summary: BacktestSummary, output_path: Path) -> None:
    """
    Save all trades from the backtest to a CSV so we can inspect them in Excel.
    """
    if not summary.trades:
        print(f"[backtest] No trades to write for {summary.symbol}.")
        return

    rows = []
    for trade in summary.trades:
        rows.append(
            {
                "symbol": trade.symbol,
                "entry_timestamp": trade.entry_timestamp.isoformat(),
                "entry_price": trade.entry_price,
                "exit_timestamp": trade.exit_timestamp.isoformat(),
                "exit_price": trade.exit_price,
                "quantity": trade.quantity,
                "pnl_dollars": trade.pnl_dollars,
                "pnl_percent": trade.pnl_percent,
                "entry_reason": trade.entry_reason,
                "exit_reason": trade.exit_reason,
            }
        )

    trades_dataframe = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trades_dataframe.to_csv(output_path, index=False)

    print(f"[backtest] Wrote {len(rows)} trades to {output_path}")


# ---------------------------------------------------------------------------
# Core backtest logic
# ---------------------------------------------------------------------------

def run_backtest_for_symbol(
    symbol: str,
    csv_path: Path,
    starting_cash: float = 100_000.0,
) -> BacktestSummary:
    """
    Simple daily-bar backtest for your breakout strategy on a single symbol.

    Assumptions:
      - We only ever hold 0 or +quantity shares (no shorts).
      - Entry:
          * On day D, we compute the breakout signal using data up to D.
          * If a signal is generated, we BUY at the OPEN of day D+1.
      - Exit:
          * After entry, each subsequent bar is checked for TP/SL:
              - If low <= stop_loss_price and high >= take_profit_price in
                the same bar, we assume STOP-LOSS is hit first (pessimistic).
              - Else if high >= take_profit_price → exit at TP price.
              - Else if low <= stop_loss_price → exit at SL price.
              - Otherwise we hold.
      - No commissions/slippage yet.
    """
    daily_bars_dataframe = load_daily_bars_from_csv(csv_path)

    if daily_bars_dataframe.empty:
        print(f"[backtest] No bars loaded from {csv_path}.")
        return BacktestSummary(
            symbol=symbol,
            starting_cash=starting_cash,
            ending_cash=starting_cash,
            trades=[],
        )

    take_profit_percent = get_take_profit_percent_for_symbol(symbol)
    stop_loss_percent = get_stop_loss_percent_for_symbol(symbol)
    position_quantity = 0.0
    entry_price: Optional[float] = None
    entry_timestamp: Optional[pd.Timestamp] = None
    entry_reason: str = ""

    trades: List[BacktestTrade] = []
    cash_balance = starting_cash

    # We will iterate by index so we can look at "previous history" easily.
    timestamps = daily_bars_dataframe.index.to_list()

    for bar_index in range(len(daily_bars_dataframe)):
        bar_timestamp = timestamps[bar_index]
        bar_row = daily_bars_dataframe.iloc[bar_index]

        bar_open_price = float(bar_row["open"])
        bar_high_price = float(bar_row["high"])
        bar_low_price = float(bar_row["low"])
        bar_close_price = float(bar_row["close"])

        # -------------------------------------------------------------------
        # If we are FLAT → look for a new entry signal
        # -------------------------------------------------------------------
        if position_quantity == 0.0:
            # We need at least one full bar of history *before* today
            if bar_index == 0:
                continue

            # History up to yesterday (inclusive)
            history_dataframe = daily_bars_dataframe.iloc[: bar_index + 1]

            entry_signal: Optional[
                EntrySignal
            ] = compute_recent_high_breakout_signal(history_dataframe)

            if entry_signal is None:
                continue

            # We will enter at today's OPEN (the current bar's open_price)
            buy_quantity = get_buy_quantity_for_symbol(symbol)
            entry_price = bar_open_price
            entry_timestamp = bar_timestamp
            entry_reason = entry_signal.reason
            position_quantity = buy_quantity

            cash_balance -= entry_price * position_quantity

            print(
                f"[backtest] ENTRY {symbol}: {position_quantity} @ {entry_price:.2f} "
                f"on {entry_timestamp.date()} | {entry_reason}"
            )
            continue

        # -------------------------------------------------------------------
        # If we are LONG → check TP / SL to decide exit
        # -------------------------------------------------------------------
        assert entry_price is not None and entry_timestamp is not None

        take_profit_price = entry_price * (1.0 + take_profit_percent / 100.0)
        stop_loss_price = entry_price * (1.0 - stop_loss_percent / 100.0)

        exit_price: Optional[float] = None
        exit_reason: str = ""

        # If both TP and SL are touched in the same bar, we assume
        # STOP-LOSS is hit first (pessimistic).
        if bar_low_price <= stop_loss_price and bar_high_price >= take_profit_price:
            exit_price = stop_loss_price
            exit_reason = (
                f"TP and SL hit in same bar; assuming STOP first "
                f"(SL={stop_loss_price:.2f}, TP={take_profit_price:.2f})."
            )
        elif bar_low_price <= stop_loss_price:
            exit_price = stop_loss_price
            exit_reason = f"Stop-loss hit at {stop_loss_price:.2f}."
        elif bar_high_price >= take_profit_price:
            exit_price = take_profit_price
            exit_reason = f"Take-profit hit at {take_profit_price:.2f}."

        if exit_price is not None:
            exit_timestamp = bar_timestamp

            cash_balance += exit_price * position_quantity

            trade = BacktestTrade(
                symbol=symbol,
                entry_timestamp=entry_timestamp,
                entry_price=entry_price,
                exit_timestamp=exit_timestamp,
                exit_price=exit_price,
                quantity=position_quantity,
                entry_reason=entry_reason,
                exit_reason=exit_reason,
            )
            trades.append(trade)

            print(
                f"[backtest] EXIT  {symbol}: {position_quantity} @ {exit_price:.2f} "
                f"on {exit_timestamp.date()} | PnL={trade.pnl_dollars:.2f} "
                f"({trade.pnl_percent:.2f}%) | {exit_reason}"
            )

            # Reset position
            position_quantity = 0.0
            entry_price = None
            entry_timestamp = None
            entry_reason = ""

    # If a position is still open at the end, we mark-to-market at the last close.
    if position_quantity > 0.0 and entry_price is not None and entry_timestamp is not None:
        last_bar_row = daily_bars_dataframe.iloc[-1]
        last_close_price = float(last_bar_row["close"])
        last_timestamp = daily_bars_dataframe.index[-1]

        cash_balance += last_close_price * position_quantity

        trade = BacktestTrade(
            symbol=symbol,
            entry_timestamp=entry_timestamp,
            entry_price=entry_price,
            exit_timestamp=last_timestamp,
            exit_price=last_close_price,
            quantity=position_quantity,
            entry_reason=entry_reason,
            exit_reason="Closed at end of backtest (mark-to-market).",
        )
        trades.append(trade)

        print(
            f"[backtest] FORCE EXIT {symbol}: {position_quantity} @ {last_close_price:.2f} "
            f"on {last_timestamp.date()} | PnL={trade.pnl_dollars:.2f} "
            f"({trade.pnl_percent:.2f}%)"
        )

    summary = BacktestSummary(
        symbol=symbol,
        starting_cash=starting_cash,
        ending_cash=cash_balance,
        trades=trades,
    )

    import statistics

    num_trades = len(trades)

    # Use attribute access on the BacktestTrade dataclass
    winner_pnls = [trade.pnl_dollars for trade in trades if trade.pnl_dollars > 0]
    loser_pnls  = [trade.pnl_dollars for trade in trades if trade.pnl_dollars < 0]
    holding_days_list = [trade.holding_days for trade in trades]

    winners = len(winner_pnls)
    losers = len(loser_pnls)
    win_rate = (winners / num_trades * 100.0) if num_trades > 0 else 0.0

    avg_win = statistics.mean(winner_pnls) if winner_pnls else 0.0
    median_win = statistics.median(winner_pnls) if winner_pnls else 0.0
    avg_loss = statistics.mean(loser_pnls) if loser_pnls else 0.0
    median_loss = statistics.median(loser_pnls) if loser_pnls else 0.0

    avg_hold = statistics.mean(holding_days_list) if holding_days_list else 0.0
    median_hold = statistics.median(holding_days_list) if holding_days_list else 0.0
    min_hold = min(holding_days_list) if holding_days_list else 0
    max_hold = max(holding_days_list) if holding_days_list else 0

    print("==================== BACKTEST SUMMARY ====================")
    print(f"Symbol:            {symbol}")
    print(f"Starting cash:     {starting_cash:,.2f}")
    print(f"Ending cash:       {cash_balance:,.2f}")
    print(
        f"Total PnL:         {summary.total_pnl_dollars:,.2f} "
        f"({summary.total_pnl_percent:.2f}%)"
    )
    print(f"Trades:            {num_trades}")
    print(f"Winners:           {winners}")
    print(f"Losers:            {losers}")
    print(f"Win rate:          {win_rate:.2f}%")
    print()
    print(f"Average win:       {avg_win:,.2f}")
    print(f"Median win:        {median_win:,.2f}")
    print(f"Average loss:      {avg_loss:,.2f}")
    print(f"Median loss:       {median_loss:,.2f}")
    print()
    print(f"Avg holding days:  {avg_hold:.2f}")
    print(f"Median holding:    {median_hold:.2f}")
    print(f"Min holding:       {min_hold} days")
    print(f"Max holding:       {max_hold} days")
    print("=========================================================")

    # Write trades to CSV for offline inspection
    output_trades_csv = csv_path.parent / f"{symbol}_backtest_trades.csv"
    write_trades_to_csv(summary, output_trades_csv)

    return summary


# ---------------------------------------------------------------------------
# Script entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    data_folder = project_root / "data"
    tsla_csv_path = data_folder / "TSLA_daily.csv"

    if not tsla_csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found at {tsla_csv_path}. "
            f"Run tools/download_history.py for TSLA first."
        )

    run_backtest_for_symbol(
        symbol="TSLA",
        csv_path=tsla_csv_path,
        starting_cash=100_000.0,
    )
