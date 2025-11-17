from dataclasses import dataclass
from typing import Optional

import pandas as pd

from config import (
    MIN_BARS_FOR_SIGNAL,
    SIGNAL_LOOKBACK_BARS,
    BREAKOUT_TOLERANCE_PCT,
    ENTRY_LIMIT_OFFSET_PCT,
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
    """

    if bars is None or bars.empty:
        return None

    # We want at least MIN_BARS_FOR_SIGNAL bars total
    if len(bars) < MIN_BARS_FOR_SIGNAL:
        return None

    # Last bar is "current", the rest are history
    current = bars.iloc[-1]
    history = bars.iloc[:-1]

    # Limit history to the most recent SIGNAL_LOOKBACK_BARS
    if len(history) > SIGNAL_LOOKBACK_BARS:
        history = history.iloc[-SIGNAL_LOOKBACK_BARS:]

    # Basic sanity checks
    if "high" not in history or "close" not in history:
        return None

    if "close" not in current:
        return None

    recent_high = history["high"].max()
    recent_sma = history["close"].mean()
    current_close = float(current["close"])

    # If we somehow have NaNs, bail out
    if pd.isna(recent_high) or pd.isna(recent_sma) or pd.isna(current_close):
        return None

    # Uptrend filter: current close should be above the recent SMA
    if current_close <= recent_sma:
        return None

    # Breakout condition: current close within tolerance of recent_high or above
    tolerance_factor = 1.0 + (BREAKOUT_TOLERANCE_PCT / 100.0)
    min_breakout_level = recent_high / tolerance_factor  # within X% of high

    if current_close < min_breakout_level:
        # Not close enough to the recent high to consider a breakout
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

    return EntrySignal(limit_price=limit_price, reason=reason)
