# Position Sizing - Kelly Criterion

Optimal position sizing for quantitative stock trading using the Kelly Criterion, integrated with LSTM predictions.

## 📊 Overview

The Kelly Criterion is a mathematically optimal position sizing formula that maximizes the expected geometric growth rate of wealth. This implementation integrates Kelly sizing with LSTM predictions, confidence scores, and volatility regime adjustments.

**Formula:**
```
f* = (p × b - q) / b

Where:
- f* = optimal fraction of capital to risk
- p = probability of winning
- q = probability of losing (1 - p)
- b = win/loss ratio (average_win / average_loss)
```

## 🎯 Features

- ✅ **Kelly Criterion** - Mathematically optimal position sizing
- ✅ **LSTM Integration** - Scales positions by prediction confidence
- ✅ **Volatility Regimes** - Adjusts sizing based on market volatility
- ✅ **Risk Constraints** - Maximum/minimum position limits
- ✅ **Fractional Kelly** - Conservative sizing (quarter Kelly, half Kelly)
- ✅ **Type-Safe Configuration** - Dataclass-based configuration
- ✅ **Clean Modular Design** - Easy to test and extend

## 🚀 Quick Start

### Basic Usage

```python
from src.strategies.position_sizing.kelly import KellyCriterion

# Initialize Kelly calculator (quarter Kelly for conservative sizing)
kelly = KellyCriterion(kelly_fraction=0.25)

# Calculate position size
# Example: 55% win rate, avg wins are 1.5x avg losses
position_size = kelly.calculate_kelly_fraction(
    win_probability=0.55,
    win_loss_ratio=1.5
)

print(f"Position size: {position_size:.2%} of capital")
# Output: Position size: 5.42% of capital
```

### With LSTM Predictions

```python
from src.strategies.position_sizing.lstm_integration import LSTMPositionSizer
from src.strategies.position_sizing.config import get_conservative_config

# Initialize with historical data for calibration
sizer = LSTMPositionSizer(
    config=get_conservative_config(),
    historical_predictions=hist_predictions,
    historical_returns=hist_returns
)

# Size position for new prediction
position_size = sizer.size_position(
    prediction=0.025,           # 2.5% expected return
    confidence=0.80,            # 80% confidence
    recent_returns=returns[-60:]  # Last 60 days for volatility
)

print(f"Position size: {position_size:.1%}")
# Output: Position size: 12.3%
```

## 📁 Module Structure

```
position_sizing/
├── kelly.py              # Core Kelly Criterion implementation
├── utils.py              # Statistics and volatility utilities
├── config.py             # Type-safe configuration
├── lstm_integration.py   # LSTM integration layer
├── __init__.py
└── README.md            # This file
```

## 🔧 Configuration

### Pre-built Configurations

```python
from src.strategies.position_sizing.config import (
    get_conservative_config,  # Quarter Kelly, max 20%
    get_moderate_config,       # Half Kelly, max 30%
    get_aggressive_config      # Full Kelly, max 50%
)

# Use conservative config (recommended)
config = get_conservative_config()
```

### Custom Configuration

```python
from src.strategies.position_sizing.config import PositionSizingConfig

config = PositionSizingConfig(
    kelly_fraction=0.25,          # Quarter Kelly
    max_position_size=0.30,       # 30% max
    min_position_size=0.05,       # 5% min
    use_confidence_scaling=True,  # Scale by LSTM confidence
    use_volatility_adjustment=True,  # Adjust for volatility regimes
    vol_threshold_low=0.15,       # 15% vol = low regime
    vol_threshold_medium=0.25,    # 25% vol = medium regime
    vol_threshold_high=0.40       # 40% vol = high regime
)
```

## 📖 Detailed Usage

### 1. Calculate Kelly from Historical Performance

```python
from src.strategies.position_sizing.kelly import KellyCriterion

kelly = KellyCriterion(kelly_fraction=0.25)

# Automatically calculate from historical predictions
position_size = kelly.calculate_from_returns(
    historical_returns=returns,
    predictions=predictions,
    actual_returns=actual_returns
)
```

### 2. Adjust for Prediction Confidence

```python
# Base Kelly fraction
base_kelly = 0.10  # 10% position

# Adjust based on confidence
adjusted_kelly = kelly.adjust_for_confidence(
    base_kelly=base_kelly,
    confidence_score=0.85,  # 85% confidence
    scaling_factor=1.0
)

print(f"Adjusted position: {adjusted_kelly:.1%}")
# Higher confidence → larger position
# Lower confidence → smaller position
```

### 3. Detect Volatility Regime

```python
from src.strategies.position_sizing.utils import (
    calculate_rolling_volatility,
    detect_volatility_regime,
    get_regime_scaling_factor
)

# Calculate current volatility
vol = calculate_rolling_volatility(recent_returns, window=20)[-1]

# Detect regime
regime = detect_volatility_regime(vol)
print(f"Current regime: {regime.value}")  # LOW, MEDIUM, HIGH, or EXTREME

# Get scaling factor
scale_factor = get_regime_scaling_factor(regime)
adjusted_position = base_position * scale_factor
```

### 4. Full LSTM Integration

```python
from src.strategies.position_sizing.lstm_integration import LSTMPositionSizer

# Initialize
sizer = LSTMPositionSizer(
    config=config,
    historical_predictions=hist_preds,
    historical_returns=hist_rets
)

# Get sizing details
result = sizer.size_position_from_prediction(
    prediction=152.5,        # Predicted price
    current_price=150.0,     # Current price
    recent_returns=returns,  # For volatility
    prediction_std=2.5       # Prediction uncertainty
)

print(f"Position: {result['position_size']:.1%}")
print(f"Confidence: {result['confidence']:.1%}")
print(f"Expected Return: {result['expected_return']:.2%}")
print(f"Regime: {result['regime']}")
```

### 5. Update Calibration

```python
# After trading for a period, update calibration
sizer.update_calibration(
    new_predictions=recent_predictions,
    new_returns=recent_returns
)

# Check updated parameters
summary = sizer.get_sizing_summary()
print(f"Updated Win Rate: {summary['win_probability']:.1%}")
print(f"Updated Win/Loss: {summary['win_loss_ratio']:.2f}")
```

## 🎨 Complete Example

```python
import numpy as np
from src.strategies.position_sizing.lstm_integration import LSTMPositionSizer
from src.strategies.position_sizing.config import get_conservative_config

# Step 1: Load your historical data
# hist_predictions: LSTM predictions from training/validation
# hist_returns: Actual returns from training/validation
# test_predictions: New LSTM predictions
# test_returns: Returns for testing period

# Step 2: Initialize position sizer
config = get_conservative_config()
sizer = LSTMPositionSizer(
    config=config,
    historical_predictions=hist_predictions,
    historical_returns=hist_returns
)

# Step 3: Size positions for test period
position_sizes = []
for i, pred in enumerate(test_predictions):
    size = sizer.size_position(
        prediction=pred,
        confidence=confidence_scores[i],
        recent_returns=hist_returns[-60:]  # Rolling window
    )
    position_sizes.append(size)

# Step 4: Backtest with sized positions
portfolio_value = 1.0  # Start with $1
for size, pred_return, actual_return in zip(position_sizes, test_predictions, test_returns):
    # Trade in direction of prediction
    trade_return = size * np.sign(pred_return) * actual_return
    portfolio_value *= (1 + trade_return)

print(f"Final Portfolio Value: ${portfolio_value:.2f}")
```

## 📊 Run Examples

```bash
# Run comprehensive examples
python scripts/position_sizing_example.py
```

This will demonstrate:
1. Basic Kelly Criterion usage
2. LSTM integration with confidence scaling
3. Backtest comparison: Fixed vs Kelly sizing
4. Volatility regime adjustments

## 🔬 Theory

### Why Kelly Criterion?

The Kelly Criterion answers: *"What fraction of my capital should I risk to maximize long-term growth?"*

**Advantages:**
- Mathematically optimal for long-term growth
- Prevents overbetting (bankruptcy risk)
- Accounts for win probability and risk/reward ratio
- Adapts to changing market conditions

**Fractional Kelly:**
In practice, fractional Kelly (e.g., 1/4 Kelly or 1/2 Kelly) is used because:
- More conservative / less volatile
- Reduces drawdowns
- More robust to estimation errors
- Still grows faster than fixed sizing

### Integration with LSTM

LSTM predictions provide:
1. **Expected return** → Used to estimate win probability
2. **Prediction confidence** → Scale position size
3. **Historical performance** → Calibrate Kelly parameters

### Volatility Regimes

Markets alternate between different volatility states:
- **Low Vol (< 15%)**: Calm markets → Increase position sizes
- **Medium Vol (15-25%)**: Normal → Standard position sizes
- **High Vol (25-40%)**: Stressed → Reduce position sizes
- **Extreme Vol (> 40%)**: Crisis → Minimal position sizes

## ⚠️ Important Considerations

1. **Conservative Sizing**: Start with quarter Kelly (0.25). Full Kelly can be volatile.

2. **Parameter Estimation**: Kelly is sensitive to win probability and win/loss ratio. Use sufficient historical data (> 100 trades).

3. **Risk Constraints**: Always enforce maximum position limits (e.g., 30% max).

4. **Rebalancing**: Recalibrate Kelly parameters periodically as market conditions change.

5. **Transaction Costs**: High Kelly fractions with frequent rebalancing can incur significant costs.

## 📚 Key Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `kelly_fraction` | Fraction of full Kelly | 0.25 (Quarter Kelly) |
| `max_position_size` | Maximum position limit | 0.20 - 0.30 (20-30%) |
| `min_position_size` | Minimum position to enter | 0.05 (5%) |
| `volatility_lookback` | Window for vol calculation | 20 days |
| `confidence_scaling` | Use prediction confidence | True (recommended) |
| `volatility_adjustment` | Adjust for vol regimes | True (recommended) |

## 🎯 Best Practices

1. **Start Conservative**: Use quarter Kelly (0.25) and max 20-30% position
2. **Calibrate Properly**: Use at least 100+ trades for parameter estimation
3. **Monitor Performance**: Track realized win rate and win/loss ratio
4. **Update Regularly**: Recalibrate every 1-3 months
5. **Respect Constraints**: Never exceed maximum position limits
6. **Test Thoroughly**: Backtest with realistic transaction costs

## 📖 References

- Kelly, J. L. (1956). "A New Interpretation of Information Rate"
- Thorp, E. O. (1969). "Optimal Gambling Systems for Favorable Games"
- Poundstone, W. (2005). "Fortune's Formula: The Untold Story of the Scientific Betting System"

## 🔗 Integration with Main Project

This position sizing module integrates seamlessly with:
- `src/models/architectures/lstm_clean.py` - Get predictions
- `src/data/loaders.py` - Load historical data
- `src/strategies/lstm_strategy.py` - Trading signals

See `scripts/position_sizing_example.py` for complete integration examples.
