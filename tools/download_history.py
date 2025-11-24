# tools/download_history.py
"""
Download DAILY OHLCV history for GME and save it to data/GME_daily.csv.

Run from the project root, e.g.:

    python tools\download_history.py
"""

import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--symbol", default="TSLA")
args = parser.parse_args()

# -------------------------------------------------------------------
# Ensure we can import from the project root (config, etc.)
# -------------------------------------------------------------------
CURRENT_DIRECTORY = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIRECTORY, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY  # noqa: E402

# -------------------------------------------------------------------
# Alpaca client + constants
# -------------------------------------------------------------------

DATA_CLIENT = StockHistoricalDataClient(
    api_key=ALPACA_API_KEY_ID,
    secret_key=ALPACA_API_SECRET_KEY,
)

SYMBOL = args.symbol
#SYMBOL = "MNDR"
LOOKBACK_DAYS = 3 * 365  # ~3 years of daily bars


def download_gme_daily_history() -> None:
    """Fetch recent DAILY bars for GME and write them to data/GME_daily.csv."""
    end_timestamp = datetime.now(timezone.utc)
    start_timestamp = end_timestamp - timedelta(days=LOOKBACK_DAYS)

    print(
        f"[download_history] Requesting DAILY bars for {SYMBOL} "
        f"from {start_timestamp.isoformat()} to {end_timestamp.isoformat()} (IEX feed)..."
    )

    request = StockBarsRequest(
        symbol_or_symbols=SYMBOL,
        timeframe=TimeFrame(1, TimeFrameUnit.Day),
        start=start_timestamp,
        end=end_timestamp,
        feed="iex",
    )

    bars_response = DATA_CLIENT.get_stock_bars(request)

    # Convert to DataFrame
    if hasattr(bars_response, "df"):
        daily_bars_dataframe = bars_response.df
    else:
        daily_bars_dataframe = pd.DataFrame(bars_response)

    if daily_bars_dataframe.empty:
        print(f"[download_history] No daily bars returned for {SYMBOL}.")
        return

    # If index is MultiIndex (symbol, timestamp), select just this symbol
    if (
        isinstance(daily_bars_dataframe.index, pd.MultiIndex)
        and "symbol" in daily_bars_dataframe.index.names
    ):
        daily_bars_dataframe = daily_bars_dataframe.xs(SYMBOL, level="symbol")

    # Sort by timestamp
    daily_bars_dataframe = daily_bars_dataframe.sort_index()

    # Ensure data directory exists
    data_directory = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(data_directory, exist_ok=True)

    output_path = os.path.join(data_directory, f"{SYMBOL}_daily.csv")
    daily_bars_dataframe.to_csv(output_path)

    print(
        f"[download_history] Wrote {len(daily_bars_dataframe)} rows "
        f"to {output_path}"
    )


if __name__ == "__main__":
    download_gme_daily_history()
