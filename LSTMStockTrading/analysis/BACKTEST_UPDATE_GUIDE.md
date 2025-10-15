# Backtesting Update Guide

## Summary
The `Backtesting.py` file needs to be updated to support multi-feature LSTM models with technical indicators, matching the `PyTorchOptimized.py` implementation.

## Key Changes Required

### 1. Add Technical Indicators Function (After line 13)
Add the `calculate_technical_indicators()` function to compute the same features used during training:
- Returns, SMA (5, 20), RSI, MACD, Bollinger Bands, Momentum, Volatility, ROC

### 2. Add ImprovedLSTM Model Class (After line 13)
Add the `ImprovedLSTM` class that matches the architecture in `PyTorchOptimized.py`:
- 3 LSTM layers with 256 hidden units
- Layer normalization
- Batch normalization
- 3 fully connected layers (256 → 128 → 64 → 1)

### 3. Update `generate_trading_signals()` Method (Lines 65-122)
**Current**: Uses only close price (single feature)
```python
sequence = historical_data['close'].iloc[i-sequence_length:i].values.reshape(-1, 1)
```

**Updated**: Use multi-feature input
```python
# Add technical indicators
data_with_indicators = calculate_technical_indicators(historical_data)
data_with_indicators = data_with_indicators.dropna().reset_index(drop=True)

# Define feature columns (must match training)
feature_columns = ['close', 'returns', 'sma_5', 'sma_20', 'rsi', 
                  'macd', 'macd_signal', 'bb_position', 'bb_width',
                  'momentum', 'volatility', 'roc']

# Extract feature values
feature_data = data_with_indicators[feature_columns].values

# In the loop:
sequence = feature_data[i-sequence_length:i]
scaled_sequence = scaler.transform(sequence)

# Inverse transform (close is first feature)
n_features = scaler.n_features_in_
dummy_pred = np.zeros((1, n_features))
dummy_pred[:, 0] = scaled_pred.flatten()
prediction = scaler.inverse_transform(dummy_pred)[0, 0]
```

### 4. Update Main Execution (Lines 591-600)
**Current**:
```python
stock = "AAPL"
data_directory = Path(__file__).parent.parent
```

**Updated**:
```python
stock = "MSFT"  # Change as needed
data_directory = Path(__file__).resolve().parent.parent

print(f"\nLoading model and data for {stock}...")
print(f"Model path: {model_path}")
print(f"Scaler features: {scaler.n_features_in_}")
```

## Implementation Steps

1. **Backup** the current `Backtesting.py` file
2. **Copy** `calculate_technical_indicators()` from `PyTorchOptimized.py` (lines 66-128)
3. **Copy** `ImprovedLSTM` class from `PyTorchOptimized.py` (lines 238-297)
4. **Update** the `generate_trading_signals()` method to use multi-feature inputs
5. **Update** the main execution section with better logging
6. **Test** with the MSFT model that was just trained

## Expected Behavior

After updates:
- ✅ Backtesting uses same 12 features as training
- ✅ Model architecture matches trained model
- ✅ Predictions are based on technical indicators, not just price
- ✅ More accurate trading signals

## Files to Reference
- **Training Model**: `LSTMStockTrading/models/PyTorchOptimized.py`
- **Current Backtest**: `LSTMStockTrading/analysis/Backtesting.py`
- **Trained Models**: `LSTMStockTrading/models/MSFT_optimized_model.pth`
- **Scaler**: `LSTMStockTrading/models/MSFT_scaler_optimized.pkl`

## Testing
After making changes, run:
```python
python LSTMStockTrading/analysis/Backtesting.py
```

Expected output should show:
- "Using 12 features for backtesting"
- Feature list matching training
- Scaler features: 12
