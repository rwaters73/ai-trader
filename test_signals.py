from history import fetch_price_history
from signals import compute_recent_high_breakout_signal


def main():
    symbol = "AAPL"

    print(f"Fetching recent history for {symbol}...")
    # You can play with lookback_value/unit here, independent of config defaults
    bars = fetch_price_history(symbol, lookback_value=130, lookback_unit="days")

    print(f"Got {len(bars)} bars.")
    print(bars.tail())

    signal = compute_recent_high_breakout_signal(bars)

    if signal is None:
        print("\nNo entry signal generated.")
    else:
        print("\nENTRY SIGNAL:")
        print(f"  Limit price: {signal.limit_price:.2f}")
        print(f"  Reason: {signal.reason}")


if __name__ == "__main__":
    main()
