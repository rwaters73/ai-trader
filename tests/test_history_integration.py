import os
import pytest

# Skip tests gracefully if ALPACA credentials are not present in environment
if not (os.getenv("ALPACA_API_KEY_ID") and os.getenv("ALPACA_API_SECRET_KEY")):
    pytest.skip("Alpaca API keys not set; skipping integration tests.", allow_module_level=True)

import pandas as pd
from history import fetch_price_history


def test_fetch_recent_daily_and_intraday_are_non_empty():
    df_days = fetch_price_history("AAPL", lookback_value=3, lookback_unit="days")
    assert isinstance(df_days, pd.DataFrame)
    assert not df_days.empty

    df_hours = fetch_price_history("AAPL", lookback_value=6, lookback_unit="hours")
    assert isinstance(df_hours, pd.DataFrame)
    assert not df_hours.empty
