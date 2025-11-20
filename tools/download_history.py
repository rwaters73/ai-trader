# tools/download_history.py

from datetime import datetime, timedelta, timezone
import os
from typing import Optional

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from config import ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY


# One shared client for this script
historical_data_client = StockHistoricalDataClient(
    api_key=ALPACA_API_KEY_ID,
    secret_key=ALPACA_API_SECRET_KEY,
)


def download_daily_history_for_symbol(
    symbol: str,
    number_of_years: int = 3,
    feed_name: str = "iex",
    output_file_path: Optional[str] = None,
) -> None:
    """
    Download daily OHLCV bars for `symbol` using Alpaca and write them
    to a CSV file.

    Columns in the CSV:
        timestamp, open, high, low, close, volume

    - number_of_years: how many years of history to request (approximate).
    - feed_name: typically "iex" for your current subscription tier.
    - output_file_path: if None, writes to data/{symbol}_daily.csv
    """
    end_timestamp = datetime.now(timezone.utc)
    start_timestamp = end_timestamp - timedelta(days=365 * number_of_years)

    print(
        f"[download] Requesting ~{number_of_years} years of DAILY bars for {symbol} "
        f"from {start_timestamp.isoformat()} to {end_timestamp.isoformat()} using feed={feed_name}..."
    )

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(1, TimeFrameUnit.Day),
        start=start_timestamp,
        end=end_timestamp,
        feed=feed_name,
    )

    bars_response = historical_data_client.get_stock_bars(request)

    # Alpaca's response usually has a .df attribute with a MultiIndex
    if hasattr(bars_response, "df"):
        bars_dataframe = bars_response.df
    else:
        # Fallback, though this is unlikely needed
        bars_dataframe = pd.DataFrame(bars_response)

    if bars_dataframe is None or bars_dataframe.empty:
        print(f"[download] No data received for {symbol}. Nothing written.")
        return

    # If it is a MultiIndex with 'symbol' as one of the levels, select the symbol
    if (
        isinstance(bars_dataframe.index, pd.MultiIndex)
        and "symbol" in bars_dataframe.index.names
    ):
        bars_dataframe = bars_dataframe.xs(symbol, level="symbol")

    # Ensure bars are sorted by timestamp
    bars_dataframe = bars_dataframe.sort_index()

    # Keep only the columns we care about
    required_columns = ["open", "high", "low", "close", "volume"]
    missing_columns = [column_name for column_name in required_columns if column_name not in bars_dataframe.columns]
    if missing_columns:
        print(
            f"[download] WARNING: Missing columns {missing_columns} in data for {symbol}. "
            f"Available columns: {list(bars_dataframe.columns)}"
        )

    # Select intersection of required and available columns
    available_columns = [column_name for column_name in required_columns if column_name in bars_dataframe.columns]
    cleaned_dataframe = bars_dataframe[available_columns].copy()

    # Move timestamp index into a regular column
    cleaned_dataframe = cleaned_dataframe.reset_index()

    # Alpaca typically names the index column "timestamp"; if not, rename it
    if "timestamp" not in cleaned_dataframe.columns:
        # Assume the first column is the timestamp-like column
        first_column_name = cleaned_dataframe.columns[0]
        cleaned_dataframe = cleaned_dataframe.rename(columns={first_column_name: "timestamp"})

    # Decide where to save the file
    if output_file_path is None:
        data_directory = os.path.join("data")
        os.makedirs(data_directory, exist_ok=True)
        output_file_path = os.path.join(data_directory, f"{symbol}_daily.csv")

    cleaned_dataframe.to_csv(output_file_path, index=False)

    print(
        f"[download] Wrote {len(cleaned_dataframe)} rows for {symbol} "
        f"to {output_file_path}"
    )


if __name__ == "__main__":
    # For now, we hard-code TSLA and 3 years as agreed
    download_daily_history_for_symbol(symbol="TSLA", number_of_years=3)
