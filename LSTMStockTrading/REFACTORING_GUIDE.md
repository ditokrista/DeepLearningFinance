# Refactoring Guide - Modular Architecture

## Overview

This project has been refactored from monolithic scripts into a clean, modular architecture following software engineering best practices.

## What Changed?

### Before (Monolithic)
```
❌ models/PyTorchOptimized.py (672 lines)
   - Model architecture
   - Feature engineering
   - Data loading
   - Training loops
   - Evaluation
   - Visualization
   - Everything mixed together!

❌ scripts/train_model.py (239 lines)
   - Similar monolithic structure
   - Hard to reuse code
   - Difficult to test
   - Not production-ready
```

### After (Modular)
```
✅ Clean separation of concerns
✅ Reusable modules
✅ Easy to test
✅ Production-ready
✅ Easy to extend
```

---

## New Project Structure

```
LSTMStockTrading/
│
├── scripts/                              # EXECUTABLE SCRIPTS (Entry Points)
│   ├── train_model_clean.py             # Clean training script with CLI
│   ├── predict.py                       # Prediction/inference script
│   ├── download_data.py                 # Data download utility
│   └── run_backtest.py                  # Backtesting script
│
├── src/                                  # CORE LIBRARY CODE
│   │
│   ├── data/                            # Data handling
│   │   ├── loaders.py                   # ✨ Data loading & preparation
│   │   └── features/
│   │       ├── technical.py             # ✨ Technical indicators
│   │       └── __init__.py
│   │
│   ├── models/                          # Model components
│   │   ├── architectures/
│   │   │   ├── lstm_clean.py            # ✨ Pure model definitions
│   │   │   ├── lstm.py                  # (old monolithic - keep for reference)
│   │   │   └── __init__.py
│   │   │
│   │   ├── training/
│   │   │   ├── trainer.py               # ✨ Training logic & early stopping
│   │   │   └── __init__.py
│   │   │
│   │   ├── evaluation.py                # ✨ Metrics & evaluation
│   │   └── __init__.py
│   │
│   └── utils/
│       ├── config.py                    # ✨ Configuration management
│       └── __init__.py
│
├── models/                               # Saved models
├── data/                                 # Stock data
├── config/                               # Configuration files
└── artifacts/                            # Experiment artifacts
```

---

## Key Modules Explained

### 1. **`src/models/architectures/lstm_clean.py`** - Model Definitions ONLY

**Purpose**: Pure PyTorch model architecture - no training, data, or config logic.

```python
from src.models.architectures.lstm_clean import LSTMModel, SimpleLSTM, get_model

# Create model
model = LSTMModel(
    input_dim=12,
    hidden_dim=256,
    num_layers=3,
    dropout=0.3
)

# Or use factory
model = get_model(model_type='enhanced', input_dim=12, hidden_dim=256)
```

**What's inside:**
- `LSTMModel` - Enhanced LSTM with batch norm
- `SimpleLSTM` - Baseline model
- `get_model()` - Factory function

---

### 2. **`src/data/loaders.py`** - Data Loading & Preparation

**Purpose**: Load stock data, create sequences, handle scaling.

```python
from src.data.loaders import prepare_data, create_data_loaders

# Prepare data
data = prepare_data(
    symbol='AAPL',
    look_back=60,
    train_ratio=0.7,
    use_technical_indicators=True
)

# Create PyTorch dataloaders
train_loader, val_loader = create_data_loaders(
    data['X_train'], data['y_train'],
    data['X_val'], data['y_val'],
    batch_size=32
)
```

**What's inside:**
- `load_stock_data()` - Load CSV files
- `create_sequences()` - Time series sequences
- `prepare_data()` - Full pipeline
- `create_data_loaders()` - PyTorch DataLoaders
- `inverse_transform_predictions()` - Scale back to original

---

### 3. **`src/data/features/technical.py`** - Feature Engineering

**Purpose**: Calculate technical indicators and alpha factors.

```python
from src.data.features.technical import calculate_technical_indicators

# Add technical indicators to dataframe
df = calculate_technical_indicators(df)

# Features added: returns, SMA, EMA, MACD, Bollinger Bands, RSI, etc.
```

**What's inside:**
- `calculate_technical_indicators()` - Full technical analysis
- `calculate_alpha_factors()` - Quantitative alpha factors
- `get_default_feature_columns()` - Feature selection
- `select_features()` - Feature set selection

---

### 4. **`src/models/training/trainer.py`** - Training Logic

**Purpose**: Training loop, early stopping, optimization.

```python
from src.models.training.trainer import ModelTrainer, train_model, set_seed

# Set reproducible seed
set_seed(42)

# Train model
train_losses, val_losses = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    learning_rate=0.001,
    num_epochs=300,
    patience=30
)
```

**What's inside:**
- `EarlyStopping` - Prevent overfitting
- `ModelTrainer` - Complete training pipeline
- `train_model()` - Convenience function
- `set_seed()` - Reproducibility

---

### 5. **`src/models/evaluation.py`** - Metrics & Evaluation

**Purpose**: Calculate metrics and evaluate models.

```python
from src.models.evaluation import evaluate_model, calculate_trading_metrics

# Evaluate model
predictions, actual, metrics = evaluate_model(
    model, X_test, y_test, scaler, device, phase="Test"
)

# Metrics: RMSE, MAE, Direction Accuracy, Sharpe Ratio, Max Drawdown
```

**What's inside:**
- `calculate_trading_metrics()` - Trading-specific metrics
- `evaluate_model()` - Full evaluation pipeline
- `print_model_summary()` - Model architecture summary

---

### 6. **`src/utils/config.py`** - Configuration Management

**Purpose**: Centralized configuration with dataclasses.

```python
from src.utils.config import Config, get_default_config

# Get default config
config = get_default_config(symbol="AAPL", model_type='enhanced')

# Modify config
config.model.hidden_dim = 512
config.training.learning_rate = 0.0005

# Save/load config
config.save('config/my_config.yaml')
config = Config.load('config/my_config.yaml')

# Print config
config.print_config()
```

**What's inside:**
- `ModelConfig` - Model architecture settings
- `TrainingConfig` - Training hyperparameters
- `DataConfig` - Data processing settings
- `PathConfig` - File paths
- `Config` - Master configuration

---

## Usage Examples

### Train a Model

```bash
# Basic training
python scripts/train_model_clean.py --symbol AAPL

# Custom hyperparameters
python scripts/train_model_clean.py \
    --symbol TSLA \
    --model-type enhanced \
    --hidden-dim 512 \
    --num-layers 4 \
    --batch-size 64 \
    --epochs 500 \
    --lr 0.0005 \
    --look-back 90 \
    --feature-set extended

# From config file
python scripts/train_model_clean.py --config config/my_config.yaml

# CPU only
python scripts/train_model_clean.py --symbol AAPL --no-cuda
```

### Make Predictions

```bash
python scripts/predict.py \
    --symbol AAPL \
    --model-path models/AAPL_best_model.pth \
    --scaler-path models/AAPL_scaler.pkl \
    --config-path models/AAPL_config.yaml \
    --output predictions.csv
```

---

## Programmatic Usage

You can also use the modules programmatically:

```python
from src.utils.config import get_default_config
from src.data.loaders import prepare_data, create_data_loaders
from src.models.architectures.lstm_clean import get_model
from src.models.training.trainer import train_model, set_seed
from src.models.evaluation import evaluate_model

# 1. Setup
set_seed(42)
config = get_default_config("AAPL")

# 2. Data
data = prepare_data(
    symbol=config.data.stock_symbol,
    look_back=config.data.look_back,
    train_ratio=config.data.train_ratio
)

# 3. DataLoaders
train_loader, val_loader = create_data_loaders(
    data['X_train'], data['y_train'],
    data['X_val'], data['y_val'],
    batch_size=config.training.batch_size
)

# 4. Model
model = get_model(
    model_type='enhanced',
    input_dim=data['n_features'],
    hidden_dim=256,
    num_layers=3
)

# 5. Train
train_losses, val_losses = train_model(
    model, train_loader, val_loader,
    device=config.device,
    learning_rate=0.001,
    num_epochs=300
)

# 6. Evaluate
pred, actual, metrics = evaluate_model(
    model, data['X_test'], data['y_test'],
    data['scaler'], config.device
)

print(f"Test RMSE: {metrics['RMSE']:.4f}")
print(f"Direction Accuracy: {metrics['Direction_Accuracy']:.2f}%")
```

---

## Benefits of New Structure

### 1. **Separation of Concerns**
- Each module has ONE responsibility
- Easy to understand and maintain
- Changes don't cascade through codebase

### 2. **Reusability**
```python
# Reuse model in different contexts
from src.models.architectures.lstm_clean import LSTMModel
model = LSTMModel(input_dim=10, hidden_dim=128)

# Reuse feature engineering
from src.data.features.technical import calculate_technical_indicators
df = calculate_technical_indicators(df)

# Reuse training logic
from src.models.training.trainer import train_model
train_model(model, train_loader, val_loader, device)
```

### 3. **Testability**
```python
# Easy to write unit tests
def test_calculate_indicators():
    df = create_sample_data()
    result = calculate_technical_indicators(df)
    assert 'rsi' in result.columns
    assert 'macd' in result.columns

def test_model_forward():
    model = LSTMModel(input_dim=5, hidden_dim=64)
    x = torch.randn(32, 60, 5)  # batch_size, seq_len, features
    output = model(x)
    assert output.shape == (32, 1)
```

### 4. **Extensibility**
```python
# Add new model architecture
# src/models/architectures/transformer.py
class TransformerModel(nn.Module):
    # New architecture

# Use immediately
from src.models.architectures.transformer import TransformerModel
model = TransformerModel(...)

# Add new features
# src/data/features/alternative.py
def calculate_sentiment_features(df, news_data):
    # New features from news sentiment
```

### 5. **Production Ready**
- Clear interfaces between modules
- Easy to deploy as API service
- Version control for configs
- Model registry ready

---

## Migration Guide

If you want to migrate existing code:

### Old Way (Monolithic)
```python
# models/PyTorchOptimized.py
from models.PyTorchOptimized import main, Config

config = Config()
config.stock_symbol = "AAPL"
main()  # Runs everything
```

### New Way (Modular)
```python
# scripts/train_model_clean.py
from src.utils.config import get_default_config
from src.data.loaders import prepare_data
# ... import other modules

config = get_default_config("AAPL")
data = prepare_data(...)
model = get_model(...)
train_model(...)
```

---

## Next Steps

1. **Keep old files for reference** - Don't delete `lstm.py` yet
2. **Test new structure** - Run `train_model_clean.py`
3. **Gradually migrate** - Update other scripts to use new modules
4. **Add tests** - Write unit tests for each module
5. **Add visualization** - Create separate plotting module
6. **Documentation** - Add docstrings and examples

---

## FAQ

**Q: Can I still use the old `PyTorchOptimized.py`?**
A: Yes! It's still there. The new structure is an alternative approach.

**Q: Which approach should I use?**
A: Use `train_model_clean.py` for new work. It's more maintainable and production-ready.

**Q: How do I add a new model architecture?**
A: Create a new file in `src/models/architectures/` with your model class. Import and use it!

**Q: Can I customize the configuration?**
A: Yes! Either use CLI arguments or create a YAML config file.

**Q: How do I add new features?**
A: Add functions to `src/data/features/technical.py` or create a new feature module.

---

## Summary

The refactoring separates monolithic code into clean, focused modules:

- **`architectures/`** - Pure model definitions
- **`data/`** - Data loading & features
- **`training/`** - Training logic
- **`evaluation.py`** - Metrics
- **`config.py`** - Configuration
- **`scripts/`** - Thin entry points

This structure is:
- ✅ Maintainable
- ✅ Testable
- ✅ Reusable
- ✅ Production-ready
- ✅ Easy to extend

Enjoy the cleaner codebase! 🚀
