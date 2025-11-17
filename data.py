from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest

from config import ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY

# One shared data client for the process
_data_client = StockHistoricalDataClient(
    api_key=ALPACA_API_KEY_ID,
    secret_key=ALPACA_API_SECRET_KEY,
)


def get_latest_quote(symbol: str):
    """
    Fetch the latest quote object for a stock symbol from Alpaca's data API.
    Returns the Alpaca quote model; caller pulls bid/ask from it.
    """
    request_params = StockLatestQuoteRequest(symbol_or_symbols=symbol)
    latest_quote = _data_client.get_stock_latest_quote(request_params)
    return latest_quote[symbol]
