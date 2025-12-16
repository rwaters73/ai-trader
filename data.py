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

# Lightweight in-memory cache for the most-recent quote per symbol. This lets
# the bot continue using a slightly stale quote if Alpaca has a temporary
# outage (e.g., 502). The cache is updated on every successful quote fetch.
_last_quote_cache: dict[str, object] = {}


# -------------------------------------------------------------------
# Latest quote
# -------------------------------------------------------------------

def get_latest_quote(symbol: str):
    """
    Fetch the latest quote object for a stock symbol from Alpaca's data API.

    This function is hardened against transient Alpaca server errors (e.g., 502/503)
    by performing a small number of retries with exponential backoff. On persistent
    failure we return None so callers can continue operating rather than raising
    an exception that crashes the whole process.
    """
    request_params = StockLatestQuoteRequest(symbol_or_symbols=symbol)

    import time
    import random
    MAX_ATTEMPTS = 3
    backoff_base = 0.5

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            latest_quote_map = _data_client.get_stock_latest_quote(request_params)
            # latest_quote_map is a dict keyed by symbol
            result_quote = latest_quote_map.get(symbol)
            if result_quote is not None:
                _last_quote_cache[symbol] = result_quote
            return result_quote

        except APIError as api_err:
            # Alpaca SDK wraps HTTP errors in APIError; log and retry
            if attempt < MAX_ATTEMPTS:
                sleep_time = backoff_base * (2 ** (attempt - 1))
                sleep_time += random.uniform(0, 0.2)
                print(
                    f"[WARN] Alpaca APIError fetching latest quote for {symbol} (attempt {attempt}/{MAX_ATTEMPTS}): {api_err}. Retrying in {sleep_time:.1f}s"
                )
                time.sleep(sleep_time)
                continue
            else:
                print(
                    f"[ERROR] Alpaca APIError fetching latest quote for {symbol}: {api_err}. Giving up."
                )
                # Fall back to cached quote if we have one
                cached = _last_quote_cache.get(symbol)
                if cached is not None:
                    print(f"[WARN] Using cached (stale) quote for {symbol} after API errors.")
                    return cached
                return None

        except requests.exceptions.HTTPError as http_err:
            # Direct requests HTTP errors (in case the SDK raises them)
            if attempt < MAX_ATTEMPTS:
                sleep_time = backoff_base * (2 ** (attempt - 1))
                sleep_time += random.uniform(0, 0.2)
                print(
                    f"[WARN] HTTPError fetching latest quote for {symbol} (attempt {attempt}/{MAX_ATTEMPTS}): {http_err}. Retrying in {sleep_time:.1f}s"
                )
                time.sleep(sleep_time)
                continue
            else:
                print(
                    f"[ERROR] HTTPError fetching latest quote for {symbol}: {http_err}. Giving up."
                )
                cached = _last_quote_cache.get(symbol)
                if cached is not None:
                    print(f"[WARN] Using cached (stale) quote for {symbol} after HTTP errors.")
                    return cached
                return None

        except Exception as unexpected:
            # Be conservative: don't crash the main loop on unexpected issues
            print(
                f"[WARN] Unexpected error fetching latest quote for {symbol}: {unexpected}. Returning None."
            )
            return None

    return None


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
