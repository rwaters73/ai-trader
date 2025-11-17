from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

import pandas as pd

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import Adjustment, DataFeed

from config import (
    ALPACA_API_KEY_ID,
    ALPACA_API_SECRET_KEY,
    HISTORY_LOOKBACK_VALUE,
    HISTORY_LOOKBACK_UNIT,
    INTRADAY_TIMEFRAME_MINUTES,
)

# Type for the lookback unit we support
LookbackUnit = Literal["hours", "days"]

# One shared data client for historical bars
_data_client = StockHistoricalDataClient(
    ALPACA_API_KEY_ID,
    ALPACA_API_SECRET_KEY,
)


def _compute_time_window(
    lookback_value: int,
    lookback_unit: LookbackUnit,
):
    """
    Compute (start, end, timeframe) for the given lookback.
    - For 'hours': use intraday bars (e.g. 5-minute bars)
    - For 'days': use daily bars
    """
    now_utc = datetime.now(timezone.utc)

    if lookback_unit == "hours":
        start = now_utc - timedelta(hours=lookback_value)
        timeframe = TimeFrame(INTRADAY_TIMEFRAME_MINUTES, TimeFrameUnit.Minute)
    else:
        # Default to days
        start = now_utc - timedelta(days=lookback_value)
        timeframe = TimeFrame(1, TimeFrameUnit.Day)

    end = now_utc
    return start, end, timeframe


def fetch_price_history(
    symbol: str,
    lookback_value: Optional[int] = None,
    lookback_unit: Optional[LookbackUnit] = None,
) -> pd.DataFrame:
    """
    Fetch recent OHLCV bars for a symbol using Alpaca's market data.

    Returns a pandas DataFrame indexed by timestamp with columns like:
    ['open', 'high', 'low', 'close', 'volume', ...].

    - If lookback_value/unit are not provided, use the defaults from config.py.
    - For 'hours', we use intraday bars (e.g., 5-minute).
    - For 'days', we use daily bars.
    """
    if lookback_value is None:
        lookback_value = HISTORY_LOOKBACK_VALUE
    if lookback_unit is None:
        lookback_unit = HISTORY_LOOKBACK_UNIT  # type: ignore[assignment]

    start, end, timeframe = _compute_time_window(lookback_value, lookback_unit)  # type: ignore[arg-type]

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        start=start,
        end=end,
        timeframe=timeframe,
        adjustment=Adjustment.RAW,
        feed=DataFeed.IEX,  # important for your subscription - IEX instead of SIP
    )

    bars_resp = _data_client.get_stock_bars(request)
    df = bars_resp.df

    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    if df.index.nlevels == 2:
        df = df.xs(symbol)

    return df
