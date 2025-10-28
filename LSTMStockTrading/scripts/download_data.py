import requests
import pandas as pd
from pathlib import Path

data_directory = Path(__file__).parent

API_KEY = "GpaBfd6B6Z3Nn2ZAfeaNkoqHkN6g64TW"
symbol = "GOOG"

# FIXED: Corrected API endpoint - using /api/v3/ instead of /stable/
price_url = f"https://financialmodelingprep.com/stable/historical-price-eod/full?symbol={symbol}&apikey={API_KEY}"

price_data_path = data_directory / f"{symbol}.csv"


def get_stock_data(symbol, price_url):
    response = requests.get(price_url)

    # Debug: Print response details
    print(f"Status Code: {response.status_code}")
    print(f"Response Content-Type: {response.headers.get('content-type', 'Unknown')}")
    print(f"Response Text (first 500 chars): {response.text[:500]}")
    print("-" * 80)

    # Check if request was successful
    if response.status_code != 200:
        raise Exception(f"API request failed with status code {response.status_code}: {response.text}")

    # Try to parse JSON with better error handling
    try:
        data = response.json()
    except requests.exceptions.JSONDecodeError as e:
        raise Exception(f"Failed to decode JSON response. Response text: {response.text[:1000]}") from e

    # Check if API returned an error message
    if isinstance(data, dict) and 'Error Message' in data:
        raise Exception(f"API Error: {data['Error Message']}")

    # Extract historical data (FMP API wraps data in 'historical' key)
    if isinstance(data, dict) and 'historical' in data:
        price_data = pd.DataFrame(data['historical'])
    else:
        price_data = pd.DataFrame(data)

    price_data["date"] = pd.to_datetime(price_data["date"])
    price_data = price_data.sort_values(by="date")
    return price_data


try:
    price_data = get_stock_data(symbol, price_url)
    print(price_data.info())
    print(price_data.shape)
    price_data.to_csv(price_data_path, index=False)
    print(f"\nData successfully saved to: {price_data_path}")
except Exception as e:
    print(f"\nError occurred: {e}")
    print("\nTroubleshooting steps:")
    print("1. Verify your API key is valid at https://financialmodelingprep.com")
    print("2. Check if you've exceeded your API rate limits")
    print("3. Ensure you have an active subscription if required")