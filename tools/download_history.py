"""
Download intraday (minute) OHLCV bars from Alpaca and save to CSV.

Usage (from repo root):
    python -m tools.download_intraday_history

This will download 5-minute bars for TSLA for the last N days and write:
    data/TSLA_intraday_5m.csv
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from config import ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY


# ------------- CONFIG -------------
SYMBOL = "TSLA"

# how far back to fetch intraday history
LOOKBACK_DAYS = 30

# bar size (in minutes)
BAR_SIZE_MINUTES = 5

# use IEX feed (since that's what your live code uses)
DATA_FEED = "iex"

OUTPUT_DIR = Path("data")
OUTPUT_FILENAME = f"{SYMBOL}_intraday_{BAR_SIZE_MINUTES}m.csv"
# ----------------------------------


def get_intraday_bars_dataframe(
    symbol: str,
    lookback_days: int,
    bar_size_minutes: int,
    feed: str = "iex",
) -> pd.DataFrame:
    """Fetch intraday minute bars for a symbol as a pandas DataFrame."""

    data_client = StockHistoricalDataClient(
        api_key=ALPACA_API_KEY_ID,
        secret_key=ALPACA_API_SECRET_KEY,
    )

    end_timestamp = datetime.now(timezone.utc)
    start_timestamp = end_timestamp - timedelta(days=lookback_days)

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(bar_size_minutes, TimeFrameUnit.Minute),
        start=start_timestamp,
        end=end_timestamp,
        feed=feed,
    )

    bars_response = data_client.get_stock_bars(request)

    if hasattr(bars_response, "df"):
        bars_dataframe = bars_response.df
    else:
        bars_dataframe = pd.DataFrame(bars_response)

    if bars_dataframe.empty:
        print(f"[download_intraday] No intraday bars returned for {symbol}.")
        return bars_dataframe

    # If index is MultiIndex (symbol, timestamp), slice by symbol and drop the symbol level.
    if bars_dataframe.index.nlevels > 1 and "symbol" in bars_dataframe.index.names:
        bars_dataframe = bars_dataframe.xs(symbol, level="symbol")

    # Ensure sorted by time
    bars_dataframe = bars_dataframe.sort_index()

    # Make the index a plain column for easier CSV handling
    bars_dataframe = bars_dataframe.reset_index().rename(columns={"timestamp": "timestamp_utc"})

    return bars_dataframe


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / OUTPUT_FILENAME

    print(
        f"[download_intraday] Fetching {BAR_SIZE_MINUTES}-minute bars for "
        f"{SYMBOL} over last {LOOKBACK_DAYS} days (feed={DATA_FEED})..."
    )

    bars_dataframe = get_intraday_bars_dataframe(
        symbol=SYMBOL,
        lookback_days=LOOKBACK_DAYS,
        bar_size_minutes=BAR_SIZE_MINUTES,
        feed=DATA_FEED,
    )

    if bars_dataframe.empty:
        print("[download_intraday] No data fetched. Aborting.")
        return

    print(f"[download_intraday] Got {len(bars_dataframe)} rows. Writing to {output_path}...")
    bars_dataframe.to_csv(output_path, index=False)
    print("[download_intraday] Done.")


if __name__ == "__main__":
    main()
