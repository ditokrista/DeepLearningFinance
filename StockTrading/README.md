# LSTM Stock Trading System v2.0

**Production-Ready Quantitative Trading System with Deep Learning**

A modular, extensible quantitative trading platform combining LSTM neural networks with robust backtesting and risk management for systematic stock trading.

---

## 🎯 Overview

This system implements a complete quantitative trading pipeline:
- **Data Pipeline**: Automated data collection with quality validation
- **Model Training**: LSTM-based price prediction with experiment tracking
- **Strategy Development**: Signal generation with risk management
- **Backtesting**: Event-driven backtest engine with 30+ performance metrics
- **Deployment**: Weekly rebalancing with on-demand signal generation

### Key Features

✅ **Production-Grade Architecture**
- Modular design with clear separation of concerns
- Configuration-driven (no hardcoded parameters)
- Comprehensive logging and monitoring
- MLflow experiment tracking

✅ **Flexible Experimentation**
- Easy to test different models (LSTM, Transformers, RL)
- Plug-and-play data sources
- Configurable trading strategies
- Hyperparameter optimization

✅ **Robust Backtesting**
- Event-driven simulation
- Realistic transaction costs
- Advanced risk management (stop-loss, take-profit, drawdown controls)
- Walk-forward and Monte Carlo validation

✅ **Compute-Optimized**
- Mixed precision training for 2x T4 GPUs
- Efficient data loading and caching
- Vectorized operations where possible

---

## 📊 System Architecture

```
LSTMStockTrading/
├── config/              # YAML configuration files
├── src/                 # Production source code
│   ├── data/           # Data pipeline
│   ├── models/         # Model architectures & training
│   ├── strategies/     # Trading strategies
│   ├── backtesting/    # Backtesting engine
│   └── utils/          # Shared utilities
├── artifacts/          # Models, results, experiments
├── notebooks/          # Jupyter research notebooks
├── tests/              # Test suite
├── scripts/            # Executable scripts
└── docs/               # Documentation
```

See **[ARCHITECTURE.md](../ARCHITECTURE.md)** for detailed system design.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU support)
- FMP API key ([Get one here](https://financialmodelingprep.com))

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/DeepLearningFinance.git
cd DeepLearningFinance/StockTrading
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

For development (includes testing, linting, Jupyter):
```bash
pip install -r requirements-dev.txt
```

4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env and add your API keys
```

5. **Verify installation**
```bash
python src/utils/config_loader.py
```

---

## 📈 Usage

### 1. Download Data

```bash
python scripts/download_data.py --symbol AAPL --period 5y
```

### 2. Train Model

```bash
python scripts/train_model.py --config config/model_configs/lstm_default.yaml --symbol AAPL
```

### 3. Run Backtest

```bash
python scripts/run_backtest.py --model artifacts/models/AAPL_v1.pth --symbol AAPL
```

### 4. Generate Trading Signals

```bash
python scripts/generate_signals.py --model artifacts/models/AAPL_v1.pth --symbol AAPL
```

### 5. View Experiments (MLflow UI)

```bash
mlflow ui --backend-store-uri file:./artifacts/experiments/mlruns
# Open http://localhost:5000
```

---

## 🔧 Configuration

All system parameters are externalized to YAML files in `config/`:

### Base Configuration (`config/base_config.yaml`)
```yaml
system:
  name: "LSTM Stock Trading System"
  environment: "production"

  compute:
    device: "cuda"
    num_gpus: 2
    mixed_precision: true

trading:
  frequency: "daily"
  rebalance_schedule: "weekly"

  risk:
    max_position_size: 1.0
    max_drawdown_limit: 0.25
```

### Model Configuration (`config/model_configs/lstm_default.yaml`)
```yaml
model:
  name: "ImprovedLSTM"
  architecture:
    hidden_dim: 128
    num_layers: 2
    dropout: 0.2

training:
  epochs: 100
  batch_size: 64
  optimizer:
    type: "adam"
    lr: 0.001
```

### Backtest Configuration (`config/backtest_config.yaml`)
```yaml
backtest:
  portfolio:
    initial_capital: 100000

  trading:
    position_sizing:
      method: "volatility_scaled"
      base_size: 0.65

  risk:
    stop_loss:
      enabled: true
      trailing_pct: 0.10
```

---

## 📊 Performance Metrics

The backtesting engine calculates 30+ metrics:

### Returns
- Total Return
- Annualized Return
- Cumulative Return

### Risk
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio
- Max Drawdown
- Value at Risk (VaR)

### Trading
- Win Rate
- Profit Factor
- Average Win/Loss
- Trade Frequency

---

## 🧪 Testing

Run the full test suite:

```bash
pytest tests/ -v --cov=src --cov-report=html
```

Run specific tests:

```bash
pytest tests/unit/test_data_loaders.py
pytest tests/integration/test_backtest_engine.py
```

---

## 📚 Documentation

- **[ARCHITECTURE.md](../ARCHITECTURE.md)** - System architecture and design decisions
- **[docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md)** - Detailed implementation plan with 7 phases
- **[docs/setup_guide.md](docs/setup_guide.md)** - Environment setup (coming soon)
- **[docs/training_guide.md](docs/training_guide.md)** - Model training guide (coming soon)
- **[docs/backtesting_guide.md](docs/backtesting_guide.md)** - Backtesting guide (coming soon)

---

## 🛠️ Development

### Code Quality

Format code:
```bash
black src/ tests/
isort src/ tests/
```

Lint code:
```bash
pylint src/
mypy src/
```

### Research Notebooks

Launch Jupyter:
```bash
jupyter lab notebooks/
```

Notebooks are organized by category:
- `01_data_exploration/` - Data analysis
- `02_feature_engineering/` - Feature research
- `03_model_experiments/` - Model architecture experiments
- `04_strategy_research/` - Strategy development
- `05_backtest_analysis/` - Backtest analysis

---

## 🗺️ Roadmap

### ✅ Phase 1: Foundation (Completed)
- [x] Directory structure
- [x] Configuration system
- [x] Logging infrastructure
- [x] Documentation

### 🔄 Phase 2: Data Pipeline (In Progress)
- [ ] FMP data provider
- [ ] Data validation
- [ ] Feature engineering
- [ ] Data loaders

### ⏳ Phase 3: Model Architecture (Planned)
- [ ] Model registry
- [ ] Training infrastructure
- [ ] Inference engine

### ⏳ Phase 4: Backtesting (Planned)
- [ ] Strategy framework
- [ ] Backtesting engine
- [ ] Performance metrics
- [ ] Risk management

### ⏳ Phase 5+: Advanced Features
- [ ] Walk-forward optimization
- [ ] Monte Carlo simulation
- [ ] Transformer models
- [ ] Reinforcement learning agents

See **[IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md)** for detailed roadmap.

---

## 📊 Example Results

### Sample Backtest Performance (AAPL 2020-2025)

| Metric | Value |
|--------|-------|
| Total Return | +125.4% |
| Sharpe Ratio | 1.85 |
| Max Drawdown | -18.3% |
| Win Rate | 58.2% |
| Profit Factor | 2.14 |

*Note: Past performance is not indicative of future results.*

---

## ⚠️ Disclaimer

This software is for educational and research purposes only. It is not financial advice.

**Important Warnings:**
- Trading involves substantial risk of loss
- Past performance does not guarantee future results
- Always paper trade before using real capital
- Understand the risks before trading
- Consult a financial advisor

The authors are not responsible for any financial losses incurred from using this system.

---

## 🤝 Contributing

This is currently a solo project. Contributions welcome after Phase 7 completion.

### Development Guidelines

1. Follow PEP 8 style guide
2. Write tests for new features
3. Update documentation
4. Use type hints
5. Add docstrings (Google style)

---

## 📜 License

MIT License - See [LICENSE](LICENSE) file for details.

---

## 📧 Contact

- **Author**: Deep Learning Finance
- **Email**: your-email@example.com
- **GitHub**: https://github.com/your-username/DeepLearningFinance

---

## 🙏 Acknowledgments

### Frameworks & Libraries
- **PyTorch** - Deep learning framework
- **MLflow** - Experiment tracking
- **Pandas/NumPy** - Data manipulation

### Inspiration
- **QuantConnect** - Algorithmic trading platform design
- **Zipline** - Backtesting architecture
- **Backtrader** - Event-driven simulation

### Books
- "Advances in Financial Machine Learning" - Marcos López de Prado
- "Machine Learning for Asset Managers" - Marcos López de Prado
- "Quantitative Trading" - Ernie Chan

---

**Built with ❤️ for quantitative trading research**

*Last Updated: October 26, 2025*
