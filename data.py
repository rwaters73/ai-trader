from datetime import datetime, timedelta, timezone

import pandas as pd

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from config import ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY


# One shared data client for the process
_data_client = StockHistoricalDataClient(
    api_key=ALPACA_API_KEY_ID,
    secret_key=ALPACA_API_SECRET_KEY,
)


def get_latest_quote(symbol: str):
    """
    Fetch the latest quote object for a stock symbol from Alpaca's data API.
    Returns the Alpaca quote model; caller pulls bid/ask from it.
    """
    request_params = StockLatestQuoteRequest(symbol_or_symbols=symbol)
    latest_quote = _data_client.get_stock_latest_quote(request_params)
    return latest_quote[symbol]

def get_recent_history(symbol: str, lookback_days: int = 60) -> pd.DataFrame:
    """
    Fetch recent DAILY OHLCV bars for `symbol` over the last `lookback_days`.

    Returns:
        A pandas DataFrame indexed by timestamp, with columns like
        ['open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap'].

    Uses the IEX feed to avoid SIP permission issues.
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(1, TimeFrameUnit.Day),  # 1-day bars
        start=start,
        end=end,
        feed="iex",  # IMPORTANT: avoid SIP subscription errors
    )

    bars = _data_client.get_stock_bars(request)

    # Alpaca's response object usually has a `.df` attribute.
    if hasattr(bars, "df"):
        df = bars.df
    else:
        # Fallback: try to interpret directly as a DataFrame-like structure.
        try:
            df = pd.DataFrame(bars)
        except Exception:
            return pd.DataFrame()

    if df.empty:
        return df

    # If multiple symbols were requested, bars.df has a MultiIndex (symbol, timestamp).
    # We requested a single symbol, but we still handle the MultiIndex shape for safety.
    if isinstance(df.index, pd.MultiIndex) and "symbol" in df.index.names:
        df = df.xs(symbol, level="symbol")

    # Ensure rows are in ascending time order
    df = df.sort_index()

    return df
