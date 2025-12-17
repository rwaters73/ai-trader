"""tools/compare_atr.py

Compare ATR calculations between the live-style indicator (EWMA ATR in
`indicators.compute_atr_series`) and the backtest-style ATR (rolling mean in
`backtest.compute_atr`).

Usage:
    python tools/compare_atr.py --symbol AAPL [--date YYYY-MM-DD | --index N] [--period 14]

If neither --date nor --index are provided, the latest available date is used.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional
import sys

# When executed as a script from tools/, ensure project root is on sys.path
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import pandas as pd

from indicators import compute_atr_series
from backtest import compute_atr, load_daily_bars_from_csv
from config import ATR_PERIOD_DEFAULT


def format_float(x: Optional[float]) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x:.6f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--date", required=False, help="Date in YYYY-MM-DD to inspect (uses CSV timestamp dates)")
    parser.add_argument("--index", required=False, type=int, help="Integer index into the daily bars (0-based)")
    parser.add_argument("--period", required=False, type=int, default=ATR_PERIOD_DEFAULT)

    args = parser.parse_args()

    symbol = args.symbol
    period = args.period

    # Resolve CSV with same logic used by backtest helper (case-insensitive, fuzzy suggestions)
    try:
        from backtest import resolve_daily_csv_path_for_symbol

        csv_path_str = resolve_daily_csv_path_for_symbol(symbol)
        csv_path = Path(csv_path_str)
    except Exception as e:
        raise FileNotFoundError(f"Daily CSV not found for symbol '{symbol}': {e}")

    df = load_daily_bars_from_csv(str(csv_path))

    if df.empty:
        raise ValueError(f"Daily CSV for {symbol} is empty: {csv_path}")

    # Compute both ATR series
    atr_live = compute_atr_series(df, period)
    atr_backtest = compute_atr(df, period)

    # Choose which row to inspect
    if args.index is not None:
        idx = args.index
        if idx < 0 or idx >= len(df):
            raise IndexError(f"Index {idx} out of range for {len(df)} bars")
        row_timestamp = df.iloc[idx]["timestamp"]
        live_val = atr_live.iloc[idx] if idx < len(atr_live) else None
        back_val = atr_backtest.iloc[idx] if idx < len(atr_backtest) else None

    elif args.date is not None:
        # Parse date and find first matching row with same date (date portion)
        target_date = pd.to_datetime(args.date).date()
        df_dates = pd.to_datetime(df["timestamp"]).dt.date
        matches = df_dates == target_date
        if not matches.any():
            raise ValueError(f"Date {args.date} not found in {csv_path}")
        idx = matches.to_list().index(True)
        row_timestamp = df.iloc[idx]["timestamp"]
        live_val = atr_live.iloc[idx]
        back_val = atr_backtest.iloc[idx]

    else:
        # default: last row
        idx = len(df) - 1
        row_timestamp = df.iloc[idx]["timestamp"]
        live_val = atr_live.iloc[idx]
        back_val = atr_backtest.iloc[idx]

    # Prepare and print results
    print("=== ATR comparison ===")
    print(f"Symbol: {symbol}")
    print(f"Timestamp: {row_timestamp}")
    print(f"Period: {period}")
    print()

    live_str = format_float(float(live_val)) if live_val is not None and not pd.isna(live_val) else "n/a"
    back_str = format_float(float(back_val)) if back_val is not None and not pd.isna(back_val) else "n/a"

    print(f"Live ATR (EWMA) : {live_str}")
    print(f"Backtest ATR (roll mean) : {back_str}")

    if (live_val is not None and not pd.isna(live_val)) and (back_val is not None and not pd.isna(back_val)):
        diff = float(live_val) - float(back_val)
        pct = diff / float(back_val) * 100.0 if float(back_val) != 0 else float('inf')
        print(f"Difference: {diff:.6f} ({pct:.4f}%)")
    else:
        print("Difference: n/a (one or both values missing)")


if __name__ == "__main__":
    main()
