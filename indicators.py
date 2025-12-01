# indicators.py
import pandas as pd

def compute_atr_series(
    bars_dataframe: pd.DataFrame,
    period: int,
) -> pd.Series:
    """
    Compute Wilder-style ATR over `period` bars using high/low/close.
    Returns a pandas Series aligned with bars_dataframe.index.
    """
    high = bars_dataframe["high"]
    low = bars_dataframe["low"]
    close = bars_dataframe["close"]

    previous_close = close.shift(1)

    true_range_1 = (high - low).abs()
    true_range_2 = (high - previous_close).abs()
    true_range_3 = (low - previous_close).abs()

    true_range = pd.concat(
        [true_range_1, true_range_2, true_range_3],
        axis=1,
    ).max(axis=1)

    atr_series = true_range.ewm(alpha=1.0 / period, adjust=False).mean()
    return atr_series
