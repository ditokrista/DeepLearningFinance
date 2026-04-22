# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LSTM-based quantitative trading system built with PyTorch. Combines deep learning price prediction with backtesting, risk management, and Kelly Criterion position sizing. Targets daily data with weekly rebalancing on single-stock universe (expandable). Optimized for 2x T4 GPUs.

## Development Setup & Commands

```bash
cd LSTMStockTrading
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
```

### Key Scripts

```bash
python scripts/download_data.py --symbol AAPL --period 5y
python scripts/train_model.py --config config/model_configs/lstm_default.yaml --symbol AAPL
python scripts/run_backtest.py --model artifacts/models/AAPL_v1.pth --symbol AAPL
python scripts/check_gpu.py
```

### Code Quality (run before committing)

```bash
black src scripts && isort src scripts && flake8 src scripts && pylint src && mypy src
```

### Testing

```bash
pytest                                    # fast local check
pytest --cov=src --cov-report=term-missing  # with coverage
```

Tests go in `tests/` mirroring `src/` structure (e.g., `tests/data/test_ingestor.py`). Use scenario-driven names like `test_backtest_rejects_negative_cash`.

## Architecture

All production code lives under `LSTMStockTrading/src/` with clear separation:

- **`data/`** — Data pipeline: loaders, validators, preprocessors, feature engineering (technical indicators), pluggable providers (via abstract base class)
- **`models/`** — LSTM architectures in `architectures/` (SimpleLSTM, LSTMModel, EnhancedLSTM via `get_model()` factory), training orchestrator in `training/trainer.py`, inference in `inference/`
- **`strategies/`** — Trading logic: LSTM strategy, signal generation, position sizing (Kelly Criterion in `position_sizing/kelly.py` with fractional Kelly and confidence scaling)
- **`backtesting/`** — Event-driven backtest engine (`engine.py`), risk management (`risk/`), performance metrics, walk-forward validation
- **`utils/`** — ConfigLoader (singleton, YAML with `${ENV_VAR}` interpolation), structured JSON logging with `@log_execution_time()` decorator

Configuration is fully externalized to YAML files in `config/`. Never hardcode hyperparameters.

## Conventions

- Conventional Commits: `feat:`, `fix:`, `refactor:`, `docs:`
- Type hints on all functions, Google-style docstrings
- API keys in `.env` only (never committed), referenced via config loaders
- Large model artifacts stay in `artifacts/` locally — don't commit them
- New automation scripts go in `scripts/` with argparse flags
- New components should mirror existing package structure in `src/`
