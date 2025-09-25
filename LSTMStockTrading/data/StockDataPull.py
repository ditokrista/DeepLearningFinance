import requests
import pandas as pd
from pathlib import Path

data_directory = Path(__file__).parent

API_KEY = "GpaBfd6B6Z3Nn2ZAfeaNkoqHkN6g64TW"
symbol = "NVDA"
price_url = f"https://financialmodelingprep.com/stable/historical-price-eod/full?symbol={symbol}&apikey={API_KEY}" 

price_data_path = data_directory / f"{symbol}.csv"

def get_stock_data(symbol, price_url):
    response = requests.get(price_url)
    data = response.json()
    price_data = pd.DataFrame(data)
    price_data["date"] = pd.to_datetime(price_data["date"])
    price_data = price_data.sort_values(by="date")
    return price_data

price_data = get_stock_data(symbol, price_url)
print(price_data.info())
print(price_data.shape)
price_data.to_csv(price_data_path, index=False)

