# Repository Guidelines

## Project Structure & Module Organization
- Root files (`ARCHITECTURE.md`, `LSTMStockTrading/README.md`) describe the system design—skim them before contributing.
- `LSTMStockTrading/src/` holds production modules: `data/`, `models/`, `strategies/`, `backtesting/`, and shared `utils/`; mirror these packages when adding new components.
- `config/` stores YAML configs that drive experiments; never hardcode hyperparameters—extend or add a config instead.
- `analysis/` keeps exploratory notebooks and diagnostics, while `docs/` captures narrative guides; prefer docs for long-form explanations over comments.
- `scripts/` contains runnable entrypoints (`download_data.py`, `train_model.py`, `run_backtest.py`, etc.); new automation should live here with argparse-friendly flags.

## Build, Test, and Development Commands
- Create an isolated toolchain: `python -m venv .venv && source .venv/bin/activate`.
- Install the full stack: `pip install -r requirements-dev.txt` (this pulls runtime, testing, and linting deps).
- Typical workflow:
  - `python scripts/download_data.py --symbol AAPL --period 5y` to fetch raw data with validation.
  - `python scripts/train_model.py --config config/model_configs/lstm_default.yaml --symbol AAPL` to launch a tracked experiment.
  - `python scripts/run_backtest.py --model artifacts/models/AAPL_v1.pth --symbol AAPL` to evaluate signals locally.

## Coding Style & Naming Conventions
- Python files use 4-space indentation, type-annotated functions, and descriptive snake_case names; classes stay in PascalCase.
- Format and lint before committing: `black src scripts` → `isort src scripts` → `flake8 src scripts` → `pylint src` → `mypy src`.
- Keep configuration keys lowercase with hyphen-separated filenames (e.g., `lstm_default.yaml`); constants belong in ALL_CAPS.

## Testing Guidelines
- Place tests in a future `tests/` tree that mirrors `src/` (e.g., `tests/data/test_ingestor.py`).
- Run fast checks locally: `pytest`.
- For CI parity, include coverage: `pytest --cov=src --cov-report=term-missing`.
- Prefer scenario-driven names (`test_backtest_rejects_negative_cash`) and add fixtures for expensive setup.

## Commit & Pull Request Guidelines
- Follow the existing Conventional Commit style (`feat:`, `refactor:`, `fix:`, etc.) seen in `git log`.
- Commits should be scoped to one concern with passing lint/tests.
- PRs must explain *what* changed, *why*, and *how* to verify; link issues, paste key metrics, and attach logs or screenshots for model/backtest updates.

## Security & Configuration Tips
- Store API keys in `.env` (copied from `.env.example`) and reference them via the config loaders—never commit secrets.
- When sharing artifacts, only check in lightweight metadata; large models stay in `artifacts/` locally or via your preferred storage bucket.
