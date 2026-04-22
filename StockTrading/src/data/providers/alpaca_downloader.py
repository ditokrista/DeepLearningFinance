import pandas as pd
from alpaca.data import StockHistoricalDataClient, StockLatestQuoteRequest

client = StockHistoricalDataClient("PKQX4WSF535KRZP6Y5OTZZ2SQZ","5SnioevMNz1UiUw5e86vrM1533gSzU836wuiACzaHqQm")

multisymbol_request_params = StockLatestQuoteRequest(symbol_or_symbols=["SPY", "AAPL", "MSFT", "AMZN", "GOOGL", "TSLA"])

latest_multisymbol_quotes = client.get_stock_latest_quote(multisymbol_request_params)

spy_latest_ask_price = latest_multisymbol_quotes["SPY"].ask_price

print(spy_latest_ask_price)
