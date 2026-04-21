"""Download historical daily OHLCV data from Alpha Vantage.

Secrets are loaded from the environment (``.env`` file in the project root
or an exported variable). Never hardcode API keys in source files.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from tiingo import TiingoClient
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "src" / "data" / "price"



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="AAPL", help="Ticker symbol (default: AAPL)")
    parser.add_argument(
        "--outputsize",
        default="full",
        choices=("compact", "full"),
        help="Alpha Vantage outputsize (default: full)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write the CSV (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def get_api_key() -> str:

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("TIINGO_API_KEY")
    if not api_key or api_key.strip().lower() in {"", "your_tiingo_key_here"}:
        sys.exit(
            "ERROR: TIINGO_API_KEY is not set.\n"
            "  1. Copy .env.example to .env\n"
            "  2. Set TIINGO_API_KEY=<your-key>\n"
            "  3. Re-run this script.\n"
            "Never commit the .env file to git."
        )
    return api_key


def main() -> None:
    args = parse_args()
    api_key = get_api_key()
    client = TiingoClient({"api_key": api_key, "session": True})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    price_data_path = args.output_dir / f"{args.symbol}.csv"

    historical_prices = client.get_dataframe(
        args.symbol,
        frequency="daily",
        startDate="2020-01-01",
        endDate="2025-01-01",
    )

    historical_prices.to_csv(price_data_path, index_label="date")
    print(f"Wrote {len(historical_prices)} rows to: {price_data_path}")


if __name__ == "__main__":
    main()
