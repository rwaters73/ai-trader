from datetime import datetime, timedelta, timezone

import pandas as pd

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from config import (
    ALPACA_API_KEY_ID,
    ALPACA_API_SECRET_KEY,
    INTRADAY_LOOKBACK_MINUTES,
    INTRADAY_BAR_SIZE_MINUTES,
)

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
        A pandas DataFrame indexed by timestamp, with columns such as:
        ['open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap'].

    Uses the IEX feed to avoid SIP permission issues.
    """
    end_timestamp = datetime.now(timezone.utc)
    start_timestamp = end_timestamp - timedelta(days=lookback_days)

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(1, TimeFrameUnit.Day),  # 1-day bars
        start=start_timestamp,
        end=end_timestamp,
        feed="iex",
    )

    bars_response = _data_client.get_stock_bars(request)

    if hasattr(bars_response, "df"):
        daily_bars_data_frame = bars_response.df
    else:
        try:
            daily_bars_data_frame = pd.DataFrame(bars_response)
        except Exception:
            return pd.DataFrame()

    if daily_bars_data_frame.empty:
        return daily_bars_data_frame

    # Handle possible MultiIndex (symbol, timestamp) from Alpaca.
    if (
        isinstance(daily_bars_data_frame.index, pd.MultiIndex)
        and "symbol" in daily_bars_data_frame.index.names
    ):
        daily_bars_data_frame = daily_bars_data_frame.xs(symbol, level="symbol")

    daily_bars_data_frame = daily_bars_data_frame.sort_index()

    return daily_bars_data_frame

def get_intraday_history(
    symbol: str,
    lookback_minutes: int = INTRADAY_LOOKBACK_MINUTES,
    bar_size_minutes: int = INTRADAY_BAR_SIZE_MINUTES,
) -> pd.DataFrame:
    """
    Fetch recent INTRADAY OHLCV bars for `symbol`.

    - lookback_minutes: how far back to fetch (e.g., last 60 minutes).
    - bar_size_minutes: bar size in minutes (e.g., 5-minute bars).

    Returns:
        A pandas DataFrame indexed by timestamp, with OHLCV columns.
    """
    end_timestamp = datetime.now(timezone.utc)
    start_timestamp = end_timestamp - timedelta(minutes=lookback_minutes)

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(bar_size_minutes, TimeFrameUnit.Minute),
        start=start_timestamp,
        end=end_timestamp,
        feed="iex",
    )

    bars_response = _data_client.get_stock_bars(request)

    if hasattr(bars_response, "df"):
        intraday_bars_data_frame = bars_response.df
    else:
        try:
            intraday_bars_data_frame = pd.DataFrame(bars_response)
        except Exception:
            return pd.DataFrame()

    if intraday_bars_data_frame.empty:
        return intraday_bars_data_frame

    if (
        isinstance(intraday_bars_data_frame.index, pd.MultiIndex)
        and "symbol" in intraday_bars_data_frame.index.names
    ):
        intraday_bars_data_frame = intraday_bars_data_frame.xs(symbol, level="symbol")

    intraday_bars_data_frame = intraday_bars_data_frame.sort_index()

    return intraday_bars_data_frame
