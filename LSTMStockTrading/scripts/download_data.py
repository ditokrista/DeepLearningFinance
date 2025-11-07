import requests
import pandas as pd
from pathlib import Path
from alpha_vantage.timeseries import TimeSeries

data_directory = Path(__file__).parent

API_KEY = "OATB63W8ANXV31S9"
symbol = "AAPL"
price_data_path = data_directory.parent / "src" / "data" / "price" / f"{symbol}.csv"

ts = TimeSeries(key=API_KEY, output_format='pandas')
data, meta_data = ts.get_daily(symbol=symbol, outputsize='full')
data.to_csv(price_data_path)
print(f"Data successfully saved to: {price_data_path}")
