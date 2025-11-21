# tools/download_intraday_history.py
"""
Download INTRADAY minute OHLCV history for GME and save it to data/GME_intraday.csv.

Run from the project root, e.g.:

    python tools\download_intraday_history.py
"""

import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

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

SYMBOL = "TSLA"

# Be conservative here: 60 calendar days of 1-minute bars is already
# a decent-sized dataset and usually enough for intraday backtests.
INTRADAY_LOOKBACK_DAYS = 60
INTRADAY_BAR_SIZE_MINUTES = 1


def download_gme_intraday_history() -> None:
    """
    Fetch recent INTRADAY minute bars for GME and write them to data/GME_intraday.csv.
    """
    end_timestamp = datetime.now(timezone.utc)
    start_timestamp = end_timestamp - timedelta(days=INTRADAY_LOOKBACK_DAYS)

    print(
        f"[download_intraday_history] Requesting {INTRADAY_BAR_SIZE_MINUTES}-minute "
        f"bars for {SYMBOL} from {start_timestamp.isoformat()} "
        f"to {end_timestamp.isoformat()} (IEX feed)..."
    )

    request = StockBarsRequest(
        symbol_or_symbols=SYMBOL,
        timeframe=TimeFrame(INTRADAY_BAR_SIZE_MINUTES, TimeFrameUnit.Minute),
        start=start_timestamp,
        end=end_timestamp,
        feed="iex",
    )

    bars_response = DATA_CLIENT.get_stock_bars(request)

    # Convert to DataFrame
    if hasattr(bars_response, "df"):
        intraday_bars_dataframe = bars_response.df
    else:
        intraday_bars_dataframe = pd.DataFrame(bars_response)

    if intraday_bars_dataframe.empty:
        print(f"[download_intraday_history] No intraday bars returned for {SYMBOL}.")
        return

    # If index is MultiIndex (symbol, timestamp), select just this symbol
    if (
        isinstance(intraday_bars_dataframe.index, pd.MultiIndex)
        and "symbol" in intraday_bars_dataframe.index.names
    ):
        intraday_bars_dataframe = intraday_bars_dataframe.xs(SYMBOL, level="symbol")

    # Sort by timestamp
    intraday_bars_dataframe = intraday_bars_dataframe.sort_index()

    # Ensure data directory exists
    data_directory = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(data_directory, exist_ok=True)

    output_path = os.path.join(data_directory, f"{SYMBOL}_intraday.csv")
    intraday_bars_dataframe.to_csv(output_path)

    print(
        f"[download_intraday_history] Wrote {len(intraday_bars_dataframe)} rows "
        f"to {output_path}"
    )


if __name__ == "__main__":
    download_gme_intraday_history()
