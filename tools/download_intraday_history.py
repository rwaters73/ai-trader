# tools/download_intraday_history.py

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from config import ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY

SYMBOL = "GME"
LOOKBACK_DAYS = 3 * 365  # ~3 years
BAR_SIZE_MINUTES = 5     # 5-minute bars
FEED = "iex"             # same as you’re using elsewhere

def main():
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    output_path = data_dir / f"{SYMBOL}_intraday_{BAR_SIZE_MINUTES}min.csv"

    print(f"[download_intraday] Downloading {SYMBOL} {BAR_SIZE_MINUTES}min bars "
          f"for last {LOOKBACK_DAYS} days...")

    data_client = StockHistoricalDataClient(
        api_key=ALPACA_API_KEY_ID,
        secret_key=ALPACA_API_SECRET_KEY,
    )

    end_ts = datetime.now(timezone.utc)
    start_ts = end_ts - timedelta(days=LOOKBACK_DAYS)

    request = StockBarsRequest(
        symbol_or_symbols=SYMBOL,
        timeframe=TimeFrame(BAR_SIZE_MINUTES, TimeFrameUnit.Minute),
        start=start_ts,
        end=end_ts,
        feed=FEED,
    )

    bars = data_client.get_stock_bars(request)

    if hasattr(bars, "df"):
        df = bars.df
    else:
        df = pd.DataFrame(bars)

    if df.empty:
        print("[download_intraday] No intraday data returned.")
        return

    # If multi-index (symbol, timestamp), slice to our symbol and flatten
    if df.index.nlevels > 1 and "symbol" in df.index.names:
        df = df.xs(SYMBOL, level="symbol")

    df = df.sort_index()
    df = df.reset_index()
    df.rename(columns={"timestamp": "timestamp"}, inplace=True)

    print(f"[download_intraday] Got {len(df)} intraday rows. Writing to {output_path}")
    df.to_csv(output_path, index=False)
    print("[download_intraday] Done.")

if __name__ == "__main__":
    main()
