import os
from dotenv import load_dotenv

load_dotenv()

ALPACA_API_KEY_ID = os.getenv("ALPACA_API_KEY_ID")
ALPACA_API_SECRET_KEY = os.getenv("ALPACA_API_SECRET_KEY")
ALPACA_PAPER_BASE_URL = os.getenv("ALPACA_PAPER_BASE_URL")  # not used directly yet but kept


if not ALPACA_API_KEY_ID or not ALPACA_API_SECRET_KEY:
    raise RuntimeError("Alpaca API keys are missing. Check your .env file.")

# Symbols you want the bot to manage
SYMBOLS = ["AAPL", "TSLA", "BRK.B", "GME", "JNJ", "T", "AMZN", "BRCC"]

# Loop settings
ITERATIONS = 5000          # how many cycles to run before exiting
INTERVAL_SECONDS = 1    # seconds between cycles

# Per-symbol entry thresholds (max price at which we are willing to ENTER)
# You can adjust these per symbol as you see fit.
MAX_ENTRY_PRICE_BY_SYMBOL = {
    "AAPL": 270.0,
    "TSLA": 400.0,
    "BRK.B": 500.0,
    "GME": 20.075,
    "JNJ": 199.22,
    "T": 25.40,
    "AMZN": 230.00,
    "BRCC": 1.20,
}

# Fallback if a symbol isn't in MAX_ENTRY_PRICE_BY_SYMBOL
DEFAULT_MAX_ENTRY_PRICE = 300.0

# Per-symbol take-profit percentages (as whole percents, e.g. 5.0 = 5%)
TP_PERCENT_BY_SYMBOL = {
    "AAPL": 5.0,
    "TSLA": 10.0,
    "BRK.B": 5.0,
    "GME": 10.0,
}

# Fallback TP percent if a symbol isn't explicitly configured
DEFAULT_TP_PERCENT = 5.0

# Per-symbol base buy quantities (in shares; floats to allow fractional positions)
BUY_QTY_BY_SYMBOL = {
    "AAPL": 1.0,
    "TSLA": 1.0,
    "BRK.B": 1.0,
    "GME": 20.0,
    "JNJ": 2.0,
    "T": 10.0,
    "AMZN": 1.0,
    "BRCC": 100.0,
}

# Fallback buy qty if a symbol isn't explicitly configured
DEFAULT_BUY_QTY = 1.0

BRACKET_TP_PCT_BY_SYMBOL = {
    "AAPL": 5.0,   # TP at +5%
    "TSLA": 8.0,
}

BRACKET_SL_PCT_BY_SYMBOL = {
    "AAPL": 2.0,   # SL at -2%
    "TSLA": 3.0,
}



# End-of-day (EOD) policies per symbol.
# Types:
#   - "band_hold":          allow overnight only if min_pnl_pct <= PnL% <= max_pnl_pct
#   - "no_eod_action":      never override the strategy at EOD
#   - "always_flatten":     always go flat at EOD
#   - "min_profit_flatten": flatten if PnL% >= min_pnl_pct, else hold
EOD_POLICIES = {
    "TSLA": {
        "type": "band_hold",
        "min_pnl_pct": -5.0,
        "max_pnl_pct": 5.0,
    },
    "GME": {
        "type": "no_eod_action",
    },
    "BRK.B": {
        "type": "always_flatten",
    },
    "AAPL": {
        "type": "min_profit_flatten",
        "min_pnl_pct": 1.0,
    },
    "JNJ": {
        "type": "always_flatten",
    },
    "T": {
        "type": "always_flatten",
    },
    "AMZN": {
        "type": "always_flatten",
    },
    "BRK.B": {
        "type": "band_hold",
        "min_pnl_pct": -5.0,
        "max_pnl_pct": 5.0,
    },

}

# Default EOD behavior when a symbol has no specific policy
DEFAULT_EOD_POLICY = {
    "type": "no_eod_action",
}

# --- Logging configuration ---
LOG_DECISIONS = True
LOG_FILE_PATH = "logs/decisions.csv"

LOG_ORDERS = True
ORDER_LOG_FILE_PATH = "logs/orders.csv"

# ---------------------------------------------------------------------------
# History / signal configuration (for future strategy steps)
# ---------------------------------------------------------------------------

# How far back to look when evaluating entry signals.
# You can change these without touching code.
HISTORY_LOOKBACK_VALUE = 5          # e.g. 3 days or 3 hours
HISTORY_LOOKBACK_UNIT = "days"      # "days" or "hours"

# Intraday bar size when using hours-based lookback
INTRADAY_TIMEFRAME_MINUTES = 5      # 1, 5, 15, etc.

# ---------------------------------------------------------------------------
# Entry signal configuration
# ---------------------------------------------------------------------------

# Minimum number of bars required to even consider a signal
MIN_BARS_FOR_SIGNAL = 30

# How many of the most recent bars (excluding the last) to use when
# computing recent_high and SMA. If there are fewer than this number,
# we'll just use all available bars except the last.
SIGNAL_LOOKBACK_BARS = 10

# How close the current close must be to the recent_high (in percent)
# Example: 0.5 means within 0.5% of recent_high or above it.
BREAKOUT_TOLERANCE_PCT = 1.5

# Where to place the suggested limit price relative to the last close
# Example: 0.1 means 0.1% below the last close (slightly favorable fill).
ENTRY_LIMIT_OFFSET_PCT = 0.1

# How far below the recent SMA we are willing to tolerate and still call it an "uptrend".
# Example: 0.5 means we allow price to be up to 0.5% below the SMA.
UPTREND_TOLERANCE_PCT = 0.5

# ---------------------------------------------------------------------------
# Per-symbol bracket-style TP/SL percentages
# (These are *suggested* levels based on entry price; not yet wired to real
# bracket orders, but we'll use them soon.)
# ---------------------------------------------------------------------------

BRACKET_TP_PERCENT_BY_SYMBOL = {
    "AAPL": 5.0,   # +5% take profit
    "TSLA": 8.0,
    "BRK.B": 3.0,
    "GME": 10.0,
}

BRACKET_SL_PERCENT_BY_SYMBOL = {
    "AAPL": 2.0,   # -2% stop loss
    "TSLA": 4.0,
    "BRK.B": 1.5,
    "GME": 5.0,
}

DEFAULT_BRACKET_TP_PERCENT = 5.0
DEFAULT_BRACKET_SL_PERCENT = 2.0
