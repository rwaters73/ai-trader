import sys
from pathlib import Path
from alpaca.trading.client import TradingClient

# Add parent directory to path so we can import config, broker, etc.
sys.path.insert(0, str(Path(__file__).parent.parent))


from config import ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY

def main():
    client = TradingClient(
        ALPACA_API_KEY_ID,
        ALPACA_API_SECRET_KEY,
        paper=True
    )

    account = client.get_account()

    print("----- ACCOUNT STATE -----")
    print("Cash:", account.cash)
    print("Equity:", account.equity)
    print("Buying Power:", account.buying_power)
    print("Portfolio Value:", account.portfolio_value)
    print("Long Market Value:", account.long_market_value)
    print("Short Market Value:", account.short_market_value)
    print("Initial Margin:", account.initial_margin)
    print("Maintenance Margin:", account.maintenance_margin)
    print("Daytrade Count:", account.daytrade_count)

if __name__ == "__main__":
    main()
