# Quick Start - Modular Structure

## 🚀 Quick Training

```bash
# Train AAPL with default settings
cd StockTrading
python scripts/train_model_clean.py --symbol AAPL

# Train with custom parameters
python scripts/train_model_clean.py \
    --symbol TSLA \
    --hidden-dim 512 \
    --num-layers 4 \
    --epochs 500 \
    --batch-size 64
```

## 📊 Make Predictions

```bash
python scripts/predict.py \
    --symbol AAPL \
    --model-path models/AAPL_best_model.pth \
    --scaler-path models/AAPL_scaler.pkl \
    --config-path models/AAPL_config.yaml \
    --output predictions.csv
```

## 📁 New File Organization

```
scripts/
  ├── train_model_clean.py   ← Run this for training!
  └── predict.py              ← Run this for predictions!

src/
  ├── data/
  │   ├── loaders.py          ← Data loading functions
  │   └── features/
  │       └── technical.py    ← Feature engineering
  │
  ├── models/
  │   ├── architectures/
  │   │   └── lstm_clean.py   ← Pure model definitions
  │   ├── training/
  │   │   └── trainer.py      ← Training logic
  │   └── evaluation.py       ← Metrics & evaluation
  │
  └── utils/
      └── config.py           ← Configuration
```

## 🔧 Key Differences from Old Code

### OLD (Monolithic)
```python
# Everything in one file
from models.PyTorchOptimized import main
main()  # Does everything
```

### NEW (Modular)
```python
# Clean imports
from src.models.architectures.lstm_clean import get_model
from src.data.loaders import prepare_data
from src.models.training.trainer import train_model

# Compose your workflow
data = prepare_data('AAPL')
model = get_model('enhanced', input_dim=12)
train_model(model, ...)
```

## 💡 Why Better?

1. **Scripts = Entry Points** (thin, just CLI)
2. **src/ = Reusable Library** (import anywhere)
3. **Clean Separation** (architecture ≠ training ≠ data)
4. **Easy Testing** (each module tested independently)
5. **Production Ready** (deploy as API easily)

## 📚 Read More

- `REFACTORING_GUIDE.md` - Full documentation
- `README.md` - Project overview

## ✅ Verification

Test that everything works:

```bash
python -c "
from src.models.architectures.lstm_clean import get_model
import torch
model = get_model('enhanced', input_dim=12)
x = torch.randn(2, 60, 12)
output = model(x)
print(f'✓ Model working! Output shape: {output.shape}')
"
```

## 🎯 Next Steps

1. Try training a model: `python scripts/train_model_clean.py --symbol AAPL`
2. Check saved artifacts in `models/` directory
3. Make predictions with `scripts/predict.py`
4. Read `REFACTORING_GUIDE.md` for deep dive

Happy modeling! 🎉
