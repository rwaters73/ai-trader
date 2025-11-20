from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import requests
from alpaca.common.exceptions import APIError
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from config import (
    ALPACA_API_KEY_ID,
    ALPACA_API_SECRET_KEY,
    INTRADAY_LOOKBACK_MINUTES,
    INTRADAY_BAR_SIZE_MINUTES,
)

# -------------------------------------------------------------------
# Shared Alpaca data client
# -------------------------------------------------------------------

_data_client = StockHistoricalDataClient(
    api_key=ALPACA_API_KEY_ID,
    secret_key=ALPACA_API_SECRET_KEY,
)


# -------------------------------------------------------------------
# Latest quote
# -------------------------------------------------------------------

def get_latest_quote(symbol: str):
    """
    Fetch the latest quote object for a stock symbol from Alpaca's data API.
    Returns the Alpaca quote model; caller pulls bid/ask from it.

    NOTE: This still uses Alpaca's configured feed (e.g., IEX for your plan).
    """
    request_params = StockLatestQuoteRequest(symbol_or_symbols=symbol)
    latest_quote_map = _data_client.get_stock_latest_quote(request_params)
    return latest_quote_map[symbol]


# -------------------------------------------------------------------
# Daily history (used by breakout logic)
# -------------------------------------------------------------------

def get_recent_history(symbol: str, lookback_days: int = 60) -> Optional[pd.DataFrame]:
    """
    Fetch recent DAILY bars for the given symbol over the last `lookback_days`.
    Returns a pandas DataFrame or None if data cannot be fetched.

    This function is hardened against Alpaca API errors (e.g., 500s) so the
    trading loop does not crash if Alpaca has a temporary issue.
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

    try:
        bars_response = _data_client.get_stock_bars(request)

    except APIError as api_error:
        print(
            f"[WARN] Alpaca APIError while fetching DAILY bars for {symbol}: "
            f"{api_error}. Returning None."
        )
        return None

    except requests.exceptions.HTTPError as http_error:
        print(
            f"[WARN] HTTPError while fetching DAILY bars for {symbol}: "
            f"{http_error}. Returning None."
        )
        return None

    except Exception as unexpected_error:
        print(
            f"[WARN] Unexpected error fetching DAILY bars for {symbol}: "
            f"{unexpected_error}. Returning None."
        )
        return None

    # ----------------------------
    # Convert response → DataFrame
    # ----------------------------
    if hasattr(bars_response, "df"):
        daily_bars_data_frame = bars_response.df
    else:
        try:
            daily_bars_data_frame = pd.DataFrame(bars_response)
        except Exception:
            return None

    if daily_bars_data_frame is None or daily_bars_data_frame.empty:
        return None

    # Handle MultiIndex (symbol, timestamp)
    if (
        isinstance(daily_bars_data_frame.index, pd.MultiIndex)
        and "symbol" in daily_bars_data_frame.index.names
    ):
        daily_bars_data_frame = daily_bars_data_frame.xs(symbol, level="symbol")

    # Always sort chronologically
    daily_bars_data_frame = daily_bars_data_frame.sort_index()

    return daily_bars_data_frame


# -------------------------------------------------------------------
# Intraday history (used for confirmation)
# -------------------------------------------------------------------

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
        Returns an empty DataFrame on errors.
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

    try:
        bars_response = _data_client.get_stock_bars(request)

    except APIError as api_error:
        print(
            f"[WARN] Alpaca APIError while fetching INTRADAY bars for {symbol}: "
            f"{api_error}. Returning empty DataFrame."
        )
        return pd.DataFrame()

    except requests.exceptions.HTTPError as http_error:
        print(
            f"[WARN] HTTPError while fetching INTRADAY bars for {symbol}: "
            f"{http_error}. Returning empty DataFrame."
        )
        return pd.DataFrame()

    except Exception as unexpected_error:
        print(
            f"[WARN] Unexpected error fetching INTRADAY bars for {symbol}: "
            f"{unexpected_error}. Returning empty DataFrame."
        )
        return pd.DataFrame()

    if hasattr(bars_response, "df"):
        intraday_bars_data_frame = bars_response.df
    else:
        try:
            intraday_bars_data_frame = pd.DataFrame(bars_response)
        except Exception:
            return pd.DataFrame()

    if intraday_bars_data_frame.empty:
        return intraday_bars_data_frame

    # Handle MultiIndex (symbol, timestamp)
    if (
        isinstance(intraday_bars_data_frame.index, pd.MultiIndex)
        and "symbol" in intraday_bars_data_frame.index.names
    ):
        intraday_bars_data_frame = intraday_bars_data_frame.xs(symbol, level="symbol")

    intraday_bars_data_frame = intraday_bars_data_frame.sort_index()

    return intraday_bars_data_frame
