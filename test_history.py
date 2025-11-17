from history import fetch_price_history

def main():
    symbol = "AAPL"

    print("=== Last 3 days of daily bars ===")
    df_days = fetch_price_history(symbol, lookback_value=3, lookback_unit="days")
    print(df_days.tail())

    print("\n=== Last 6 hours of intraday bars ===")
    df_hours = fetch_price_history(symbol, lookback_value=6, lookback_unit="hours")
    print(df_hours.tail())

if __name__ == "__main__":
    main()
