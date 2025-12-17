import os
import pytest

# Skip tests gracefully if ALPACA credentials are not present in environment
if not (os.getenv("ALPACA_API_KEY_ID") and os.getenv("ALPACA_API_SECRET_KEY")):
    pytest.skip("Alpaca API keys not set; skipping integration tests.", allow_module_level=True)

from history import fetch_price_history
from signals import compute_recent_high_breakout_signal, EntrySignal
from config import MIN_BARS_FOR_SIGNAL


def test_breakout_signal_runs_on_recent_history():
    bars = fetch_price_history("AAPL", lookback_value=130, lookback_unit="days")
    assert bars is not None
    assert not bars.empty

    sig = compute_recent_high_breakout_signal(bars)
    assert (sig is None) or isinstance(sig, EntrySignal)


def test_breakout_signal_returns_none_when_insufficient_bars():
    bars = fetch_price_history("AAPL", lookback_value=10, lookback_unit="days")
    # create a short DataFrame with fewer than MIN_BARS_FOR_SIGNAL bars
    short = bars.iloc[: max(0, MIN_BARS_FOR_SIGNAL - 1)]
    result = compute_recent_high_breakout_signal(short)
    assert result is None
