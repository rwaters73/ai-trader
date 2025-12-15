from dataclasses import dataclass
from typing import Optional

import pandas as pd

from config import (
    MIN_BARS_FOR_SIGNAL,
    SIGNAL_LOOKBACK_BARS,
    BREAKOUT_TOLERANCE_PCT,
    ENTRY_LIMIT_OFFSET_PCT,
    UPTREND_TOLERANCE_PCT,   
    ATR_SMA_ENTRY_PERIOD_DEFAULT,
    ATR_ENTRY_PERIOD_DEFAULT,
    ATR_SMA_ENTRY_TOLERANCE_PCT,
    ATR_ENTRY_MIN_PCT,
    ATR_ENTRY_MAX_PCT,
)


@dataclass
class EntrySignal:
    """
    Represents a proposed entry:
      - limit_price: price to place a BUY LIMIT order
      - reason: human-readable explanation
    """
    limit_price: float
    reason: str


def compute_recent_high_breakout_signal(bars: pd.DataFrame) -> Optional[EntrySignal]:
    """
    Inspect recent OHLCV bars and decide whether to propose a long entry.

    Strategy:
      - Require at least MIN_BARS_FOR_SIGNAL bars.
      - Use the previous N bars (SIGNAL_LOOKBACK_BARS or fewer if not enough)
        as "history", and the last bar as "current".
      - Compute:
          recent_high = max(high over history)
          recent_sma  = average(close over history)
      - If current_close is above recent_sma (uptrend) and
        within BREAKOUT_TOLERANCE_PCT of recent_high (or above it),
        propose a long entry near the current close.

    Returns:
      - EntrySignal(limit_price, reason) if a trade should be considered.
      - None if no trade.

    NOTE: This version prints debug information explaining WHY it returns None.
    """

    if bars is None or bars.empty:
        print("[signal] No bars provided or DataFrame is empty; cannot compute breakout signal.")
        return None

    # We want at least MIN_BARS_FOR_SIGNAL bars total
    if len(bars) < MIN_BARS_FOR_SIGNAL:
        print(
            f"[signal] Not enough bars for breakout signal: "
            f"have {len(bars)}, need at least {MIN_BARS_FOR_SIGNAL}."
        )
        return None

    # Last bar is "current", the rest are history
    current = bars.iloc[-1]
    history = bars.iloc[:-1]

    # Limit history to the most recent SIGNAL_LOOKBACK_BARS
    if len(history) > SIGNAL_LOOKBACK_BARS:
        history = history.iloc[-SIGNAL_LOOKBACK_BARS:]

    # Basic sanity checks
    if "high" not in history or "close" not in history:
        print("[signal] Missing 'high' or 'close' columns in history; cannot compute breakout.")
        return None

    if "close" not in current:
        print("[signal] Current bar missing 'close' field; cannot compute breakout.")
        return None

    recent_high = history["high"].max()
    recent_sma = history["close"].mean()
    current_close = float(current["close"])

    # If we somehow have NaNs, bail out
    if pd.isna(recent_high) or pd.isna(recent_sma) or pd.isna(current_close):
        print(
            "[signal] NaN encountered in recent_high/recent_sma/current_close; "
            "cannot compute breakout signal."
        )
        return None

    # Uptrend filter: allow price to be up to UPTREND_TOLERANCE_PCT below SMA.
    min_uptrend_level = recent_sma * (1.0 - UPTREND_TOLERANCE_PCT / 100.0)

    if current_close < min_uptrend_level:
        print(
            "[signal] Uptrend filter failed: "
            f"current_close={current_close:.2f} < min_uptrend_level={min_uptrend_level:.2f} "
            f"(recent_sma={recent_sma:.2f}, tolerance={UPTREND_TOLERANCE_PCT:.2f}%)."
        )
        return None
    else:
        print(
            "[signal] Uptrend filter passed: "
            f"current_close={current_close:.2f} >= min_uptrend_level={min_uptrend_level:.2f} "
            f"(recent_sma={recent_sma:.2f}, tolerance={UPTREND_TOLERANCE_PCT:.2f}%)."
        )

    # Breakout condition: current close within tolerance of recent_high or above
    tolerance_factor = 1.0 + (BREAKOUT_TOLERANCE_PCT / 100.0)
    min_breakout_level = recent_high / tolerance_factor  # within X% of high

    if current_close < min_breakout_level:
        # Not close enough to the recent high to consider a breakout
        print(
            "[signal] Breakout filter failed: "
            f"current_close={current_close:.2f} < min_breakout_level={min_breakout_level:.2f} "
            f"(recent_high={recent_high:.2f}, tolerance={BREAKOUT_TOLERANCE_PCT:.2f}%)."
        )
        return None

    # Suggest a limit price slightly below the current close
    limit_offset_factor = 1.0 - (ENTRY_LIMIT_OFFSET_PCT / 100.0)
    limit_price = current_close * limit_offset_factor

    reason = (
        "Recent-high breakout signal: "
        f"current_close={current_close:.2f} > recent_sma={recent_sma:.2f}, "
        f"and close is within {BREAKOUT_TOLERANCE_PCT:.2f}% of recent_high={recent_high:.2f}. "
        f"Proposed limit entry at {limit_price:.2f}."
    )

    print(
        "[signal] Breakout signal GENERATED: "
        f"close={current_close:.2f}, recent_high={recent_high:.2f}, "
        f"recent_sma={recent_sma:.2f}, limit_price={limit_price:.2f}."
    )

    return EntrySignal(limit_price=limit_price, reason=reason)


def compute_sma_trend_entry_signal(bars: pd.DataFrame) -> Optional[EntrySignal]:
    """
    Inspect recent OHLCV bars and decide whether to propose a long entry
    based on a simple SMA trend idea.

    Logic:
      - Require at least MIN_BARS_FOR_SIGNAL bars.
      - Use up to SIGNAL_LOOKBACK_BARS most recent bars as history.
      - Compute:
          * long_sma  = SMA of all history closes
          * short_sma = SMA of the most recent half of that history
      - Conditions for an entry:
          * closing prices are generally trending up (last_close > first_close)
          * short_sma > long_sma  (short-term strength > long-term)
          * last_close > short_sma (price above fast SMA)
          * last_close is NOT too far above short_sma
            (capped by BREAKOUT_TOLERANCE_PCT to avoid chasing extremes)
      - If conditions pass, propose a limit entry slightly below last_close
        using ENTRY_LIMIT_OFFSET_PCT.

    Returns:
      - EntrySignal(limit_price, reason) if a trade should be considered.
      - None if no trade.
    """

    if bars is None or bars.empty:
        return None

    if len(bars) < MIN_BARS_FOR_SIGNAL:
        return None

    # Limit history length to the most recent SIGNAL_LOOKBACK_BARS
    if len(bars) > SIGNAL_LOOKBACK_BARS:
        history_for_signal = bars.iloc[-SIGNAL_LOOKBACK_BARS:]
    else:
        history_for_signal = bars

    if "close" not in history_for_signal:
        return None

    closes_series = history_for_signal["close"].astype(float)

    if closes_series.isna().any():
        return None

    first_close = float(closes_series.iloc[0])
    last_close = float(closes_series.iloc[-1])

    # Require an upward drift in closes
    if last_close <= first_close:
        return None

    long_sma = float(closes_series.mean())

    # Define a short SMA over the most recent half of the history (at least 3 bars)
    short_window_size = max(3, min(len(closes_series) // 2, SIGNAL_LOOKBACK_BARS // 2))
    short_sma = float(closes_series.tail(short_window_size).mean())

    # Short-term trend should be stronger than long-term
    if short_sma <= long_sma:
        return None

    # Price should be above the short SMA (momentum in our direction)
    if last_close <= short_sma:
        return None

    # Do not chase price if it is too extended above short_sma
    max_extension_factor = 1.0 + (BREAKOUT_TOLERANCE_PCT / 100.0)
    if last_close > short_sma * max_extension_factor:
        return None

    # Suggest a limit price slightly below the last close
    limit_offset_factor = 1.0 - (ENTRY_LIMIT_OFFSET_PCT / 100.0)
    limit_price = last_close * limit_offset_factor

    reason = (
        "SMA trend signal: closes trending up, "
        f"short_SMA={short_sma:.2f} > long_SMA={long_sma:.2f}, "
        f"last_close={last_close:.2f} above short_SMA but not extended by more than "
        f"{BREAKOUT_TOLERANCE_PCT:.2f}%. Proposed limit entry at {limit_price:.2f}."
    )

    return EntrySignal(limit_price=limit_price, reason=reason)

def compute_entry_signal_for_mode(
    bars: pd.DataFrame,
    mode: str = "breakout",
) -> Optional[EntrySignal]:
    """
    Dispatch to the correct entry signal implementation based on `mode`.

    mode:
        - "breakout": use compute_recent_high_breakout_signal
        - "sma_trend": use compute_sma_trend_entry_signal
    """
    if mode == "breakout":
        return compute_recent_high_breakout_signal(bars)
    elif mode == "sma_trend":
        return compute_sma_trend_entry_signal(bars)
    else:
        raise ValueError(f"Unknown entry signal mode: {mode}")


def compute_atr_sma_trend_entry_signal(
    daily_bars_dataframe: pd.DataFrame,
    sma_period: int = ATR_SMA_ENTRY_PERIOD_DEFAULT,
    atr_period: int = ATR_ENTRY_PERIOD_DEFAULT,
    sma_tolerance_pct: float = ATR_SMA_ENTRY_TOLERANCE_PCT,
    atr_min_pct: float = ATR_ENTRY_MIN_PCT,
    atr_max_pct: float = ATR_ENTRY_MAX_PCT,
) -> Optional[EntrySignal]:
    """
    Entry signal combining ATR volatility and SMA trend on daily bars.

    Returns EntrySignal(limit_price=current_close, reason=...) when
    conditions pass; otherwise returns None and prints a [signal] message.
    """

    if daily_bars_dataframe is None or daily_bars_dataframe.empty:
        print("[signal] No bars provided or DataFrame is empty; cannot compute atr_sma_trend signal.")
        return None

    required_bars = max(sma_period, atr_period) + 1
    if len(daily_bars_dataframe) < required_bars:
        print(
            f"[signal] Not enough bars for atr_sma_trend: have {len(daily_bars_dataframe)}, need at least {required_bars}."
        )
        return None

    # Ensure required columns
    for col in ("high", "low", "close"):
        if col not in daily_bars_dataframe.columns:
            print(f"[signal] Missing '{col}' column; cannot compute atr_sma_trend.")
            return None

    df = daily_bars_dataframe.copy()
    df["close"] = df["close"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)

    # SMA over close
    df["sma"] = df["close"].rolling(window=sma_period).mean()

    # ATR computation (True Range and SMA ATR)
    atr_series = _compute_atr_series(df, atr_period)

    last_row = df.iloc[-1]

    current_close = float(last_row["close"])
    current_sma = float(last_row["sma"])
    current_atr = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else None

    if pd.isna(current_sma):
        print(
            "[signal] atr_sma_trend uptrend failed: SMA is NaN (not enough data for SMA)."
        )
        return None

    # Uptrend condition
    min_uptrend_level = current_sma * (1.0 + sma_tolerance_pct / 100.0)
    if current_close < min_uptrend_level:
        print(
            "[signal] atr_sma_trend uptrend failed: "
            f"current_close={current_close:.2f} < threshold={min_uptrend_level:.2f} "
            f"(sma={current_sma:.2f}, tol={sma_tolerance_pct:.2f}%)."
        )
        return None

    if current_atr is None or pd.isna(current_atr):
        print(f"[signal] atr_sma_trend ATR filter failed: ATR is NaN or None; cannot compute volatility.")
        return None

    atr_pct = (current_atr / current_close) * 100.0
    if not (atr_min_pct <= atr_pct <= atr_max_pct):
        print(
            "[signal] atr_sma_trend ATR filter failed: "
            f"atr_pct={atr_pct:.4f}% not in [{atr_min_pct:.4f}%, {atr_max_pct:.4f}%]."
        )
        return None

    # Passed filters — propose entry at current close
    limit_price = current_close
    reason = (
        "ATR+SMA trend signal: "
        f"sma_period={sma_period}, atr_period={atr_period}, "
        f"close={current_close:.2f}, sma={current_sma:.2f}, atr_pct={atr_pct:.4f}%"
    )

    print(
        "[signal] atr_sma_trend GENERATED: "
        f"close={current_close:.2f}, sma={current_sma:.2f}, atr_pct={atr_pct:.4f}%."
    )

    return EntrySignal(limit_price=limit_price, reason=reason)


def _compute_atr_series(bars: pd.DataFrame, period: int) -> pd.Series:
    """Return ATR series computed as rolling mean of True Range over `period`.

    True Range = max(high-low, abs(high-prev_close), abs(low-prev_close)).
    """
    if bars is None or bars.empty:
        return pd.Series(dtype=float)

    prev_close = bars["close"].shift(1)
    tr1 = bars["high"] - bars["low"]
    tr2 = (bars["high"] - prev_close).abs()
    tr3 = (bars["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


if __name__ == "__main__":
    # Simple manual test harness for the breakout signal over all configured symbols.

    from data import get_recent_history
    from config import SYMBOLS

    if not SYMBOLS:
        print("[test] No symbols configured in SYMBOLS.")
    else:
        for test_symbol in SYMBOLS:
            print("\n" + "=" * 70)
            print(f"[test] Fetching recent history for {test_symbol}...")
            df = get_recent_history(test_symbol, lookback_days=60)

            if df is None or df.empty:
                print(f"[test] No history returned for {test_symbol}.")
                continue

            print(f"[test] Got {len(df)} daily bars for {test_symbol}.")
            signal = compute_recent_high_breakout_signal(df)

            if signal is None:
                print(f"[test] No entry signal generated for {test_symbol}.")
            else:
                print(f"[test] Entry signal generated for {test_symbol}:")
                print(f"       limit_price = {signal.limit_price:.2f}")
                print(f"       reason      = {signal.reason}")
