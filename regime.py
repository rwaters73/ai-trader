from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional
import threading

import pandas as pd

from data import get_recent_history
import config

# Module-level cache and lock
_cached_regime: Optional["MarketRegime"] = None
_cached_at: Optional[datetime] = None
_cache_lock = threading.Lock()


@dataclass
class MarketRegime:
    is_bull: bool
    spy_close: float
    spy_ma: float
    as_of: datetime
    explanation: str


def _fetch_spy_history(symbol: str, limit: int):
    """
    Try a few plausible calling conventions for get_recent_history to be
    tolerant of different implementations.
    """
    # Preferred: limit kw
    try:
        return get_recent_history(symbol, limit=limit)
    except TypeError:
        pass

    # Positional limit
    try:
        return get_recent_history(symbol, limit)
    except TypeError:
        pass

    # Named timeframe + limit
    try:
        return get_recent_history(symbol, timeframe="1D", limit=limit)
    except TypeError:
        pass

    # Last resort: single-arg
    try:
        return get_recent_history(symbol)
    except Exception as e:
        raise RuntimeError("Unable to call get_recent_history with known signatures") from e


def get_market_regime(force_refresh: bool = False) -> MarketRegime:
    """
    Return a MarketRegime for SPY based on its moving average.

    Uses REGIME_* constants from config.py when present:
      - REGIME_SMA_PERIOD (default 200)
      - REGIME_CACHE_SECONDS (default 60)
      - REGIME_SPY_SYMBOL (default "SPY")
      - REGIME_BULL_EXPLANATION (optional template)
      - REGIME_BEAR_EXPLANATION (optional template)
    """
    global _cached_regime, _cached_at

    period = getattr(config, "REGIME_SMA_PERIOD", 200)
    cache_seconds = getattr(config, "REGIME_CACHE_SECONDS", 60)
    spy_symbol = getattr(config, "REGIME_SPY_SYMBOL", "SPY")

    bull_template = getattr(
        config,
        "REGIME_BULL_EXPLANATION",
        "Bull regime: {symbol} close {close:.2f} >= {ma:.2f} ({period}-day SMA)",
    )
    bear_template = getattr(
        config,
        "REGIME_BEAR_EXPLANATION",
        "Bear regime: {symbol} close {close:.2f} < {ma:.2f} ({period}-day SMA)",
    )

    now = datetime.now(timezone.utc)

    with _cache_lock:
        if not force_refresh and _cached_regime is not None and _cached_at is not None:
            if (now - _cached_at) < timedelta(seconds=cache_seconds):
                return _cached_regime

    # Fetch history (try to request a bit more than `period` to be safe)
    limit = max(period + 10, period)
    df = _fetch_spy_history(spy_symbol, limit=limit)

    if df is None:
        raise RuntimeError("get_recent_history returned None for SPY")

    # Ensure DataFrame-like interface
    if isinstance(df, pd.DataFrame):
        # prefer a 'close' column (case-insensitive)
        if "close" in df.columns:
            close_series = df["close"].astype(float)
        elif "Close" in df.columns:
            close_series = df["Close"].astype(float)
        else:
            # try last column
            close_series = df.iloc[:, -1].astype(float)
        # Use the most recent rows
        recent = close_series.dropna().tail(period)
        if recent.empty:
            raise RuntimeError("SPY history contains no close prices")
        spy_close = float(recent.iloc[-1])
        spy_ma = float(recent.mean())
        # as_of: try to use last index if it's a timestamp-series, otherwise now
        try:
            last_idx = df.dropna().index[-1]
            if isinstance(last_idx, pd.Timestamp):
                as_of = last_idx.to_pydatetime()
                if as_of.tzinfo is None:
                    as_of = as_of.replace(tzinfo=timezone.utc)
            else:
                as_of = now
        except Exception:
            as_of = now
    else:
        # If get_recent_history returned another structure, attempt minimal parsing
        raise RuntimeError("Unexpected return type from get_recent_history; expected pandas.DataFrame")

    is_bull = spy_close >= spy_ma
    explanation = bull_template.format(symbol=spy_symbol, close=spy_close, ma=spy_ma, period=period) if is_bull else bear_template.format(
        symbol=spy_symbol, close=spy_close, ma=spy_ma, period=period
    )

    regime = MarketRegime(is_bull=is_bull, spy_close=spy_close, spy_ma=spy_ma, as_of=as_of, explanation=explanation)

    with _cache_lock:
        _cached_regime = regime
        _cached_at = now

    return regime


__all__ = ["MarketRegime", "get_market_regime"]