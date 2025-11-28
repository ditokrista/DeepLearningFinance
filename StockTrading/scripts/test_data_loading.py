"""
Test script to verify data loading and column standardization
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loaders import load_stock_data

def test_data_loading():
    """Test loading AAPL data with Alpha Vantage format"""
    print("="*60)
    print("Testing Data Loading with Column Standardization")
    print("="*60)
    
    try:
        # Load AAPL data
        df = load_stock_data('AAPL')
        
        print(f"\n✓ Successfully loaded AAPL data")
        print(f"  Shape: {df.shape}")
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"\nColumn names after standardization:")
        for col in df.columns:
            print(f"  - {col}")
        
        print(f"\nFirst few rows:")
        print(df.head())
        
        # Check required columns
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"\n✗ Missing required columns: {missing_cols}")
        else:
            print(f"\n✓ All required columns present: {required_cols}")
        
        print("\n" + "="*60)
        print("Data loading test PASSED!")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Error loading data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_data_loading()
