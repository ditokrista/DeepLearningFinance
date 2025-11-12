"""
Clean Training Script - Modular Approach

This script orchestrates the training workflow using modular components.
No business logic here - just imports and orchestration.
"""

import sys
from pathlib import Path
import argparse
import joblib
import pandas as pd
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import Config, get_default_config
from src.data.loaders import prepare_data, create_data_loaders
from src.models.architectures.lstm_clean import get_model
from src.models.training.trainer import train_model, set_seed
from src.models.evaluation import evaluate_model, print_model_summary


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Train LSTM model for stock price prediction')

    parser.add_argument('--symbol', type=str, default='AAPL',
                        help='Stock symbol (default: AAPL)')
    parser.add_argument('--model-type', type=str, default='enhanced',
                        choices=['enhanced', 'simple'],
                        help='Model type (default: enhanced)')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Hidden dimension (default: 256)')
    parser.add_argument('--num-layers', type=int, default=3,
                        help='Number of LSTM layers (default: 3)')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate (default: 0.3)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of epochs (default: 300)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--patience', type=int, default=30,
                        help='Early stopping patience (default: 30)')
    parser.add_argument('--look-back', type=int, default=60,
                        help='Look-back period (default: 60)')
    parser.add_argument('--feature-set', type=str, default='default',
                        choices=['minimal', 'default', 'extended', 'alpha'],
                        help='Feature set to use (default: default)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (optional)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA even if available')

    return parser.parse_args()


def main():
    """Main training workflow"""

    # Parse arguments
    args = parse_arguments()

    # Set random seed
    set_seed(args.seed)

    # Load or create config
    if args.config:
        config = Config.load(args.config)
    else:
        config = get_default_config(args.symbol, args.model_type)

        # Update config from command-line arguments
        config.data.stock_symbol = args.symbol
        config.data.look_back = args.look_back
        config.data.feature_set = args.feature_set

        config.model.model_type = args.model_type
        config.model.hidden_dim = args.hidden_dim
        config.model.num_layers = args.num_layers
        config.model.dropout = args.dropout

        config.training.batch_size = args.batch_size
        config.training.num_epochs = args.epochs
        config.training.learning_rate = args.lr
        config.training.patience = args.patience
        config.training.seed = args.seed

    # Override device if --no-cuda
    if args.no_cuda:
        config.device = torch.device('cpu')

    # Print configuration
    print("\n" + "="*60)
    print("LSTM STOCK PRICE PREDICTION - MODULAR TRAINING")
    print("="*60)
    config.print_config()

    # Step 1: Prepare data
    print("Step 1: Preparing data...")
    data = prepare_data(
        symbol=config.data.stock_symbol,
        look_back=config.data.look_back,
        train_ratio=config.data.train_ratio,
        validation_ratio=config.data.validation_ratio,
        use_technical_indicators=config.data.use_technical_indicators,
        scaler_type=config.data.scaler_type
    )

    # Update input dimension based on actual features
    config.model.input_dim = data['n_features']

    # Step 2: Create data loaders
    print("Step 2: Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        data['X_train'],
        data['y_train'],
        data['X_val'],
        data['y_val'],
        batch_size=config.training.batch_size
    )

    # Step 3: Initialize model
    print("Step 3: Initializing model...")
    model = get_model(
        model_type=config.model.model_type,
        input_dim=config.model.input_dim,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
        output_dim=config.model.output_dim
    )
    model = model.to(config.device)
    print_model_summary(model)

    # Step 4: Train model
    print("Step 4: Training model...")
    save_path = config.paths.models_dir / f"{config.data.stock_symbol}_best_model.pth"
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config.device,
        learning_rate=config.training.learning_rate,
        num_epochs=config.training.num_epochs,
        patience=config.training.patience,
        gradient_clip=config.training.gradient_clip,
        save_path=save_path,
        verbose=True
    )

    # Step 5: Evaluate on all sets
    print("Step 5: Evaluating model...")
    train_pred, train_actual, train_metrics = evaluate_model(
        model, data['X_train'], data['y_train'],
        data['scaler'], config.device, "Training"
    )
    val_pred, val_actual, val_metrics = evaluate_model(
        model, data['X_val'], data['y_val'],
        data['scaler'], config.device, "Validation"
    )
    test_pred, test_actual, test_metrics = evaluate_model(
        model, data['X_test'], data['y_test'],
        data['scaler'], config.device, "Test"
    )

    # Step 6: Save artifacts
    print("Step 6: Saving artifacts...")

    # Save final model
    model_path = config.paths.models_dir / f"{config.data.stock_symbol}_final_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

    # Save scaler
    scaler_path = config.paths.models_dir / f"{config.data.stock_symbol}_scaler.pkl"
    joblib.dump(data['scaler'], scaler_path)
    print(f"Scaler saved to: {scaler_path}")

    # Save metrics
    metrics_df = pd.DataFrame({
        'Set': ['Train', 'Validation', 'Test'],
        'RMSE': [train_metrics['RMSE'], val_metrics['RMSE'], test_metrics['RMSE']],
        'MAE': [train_metrics['MAE'], val_metrics['MAE'], test_metrics['MAE']],
        'Direction_Accuracy': [
            train_metrics['Direction_Accuracy'],
            val_metrics['Direction_Accuracy'],
            test_metrics['Direction_Accuracy']
        ],
        'R2': [train_metrics['R2'], val_metrics['R2'], test_metrics['R2']],
        'Sharpe_Ratio': [
            train_metrics['Sharpe_Ratio'],
            val_metrics['Sharpe_Ratio'],
            test_metrics['Sharpe_Ratio']
        ]
    })

    metrics_path = config.paths.models_dir / f"{config.data.stock_symbol}_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to: {metrics_path}")

    # Save configuration
    config_path = config.paths.models_dir / f"{config.data.stock_symbol}_config.yaml"
    config.save(config_path)
    print(f"Configuration saved to: {config_path}")

    # Save training history
    history_df = pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'val_loss': val_losses
    })
    history_path = config.paths.models_dir / f"{config.data.stock_symbol}_training_history.csv"
    history_df.to_csv(history_path, index=False)
    print(f"Training history saved to: {history_path}")

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60 + "\n")

    # Print final summary
    print("Final Test Set Performance:")
    print(f"  RMSE: {test_metrics['RMSE']:.4f}")
    print(f"  Direction Accuracy: {test_metrics['Direction_Accuracy']:.2f}%")
    print(f"  Sharpe Ratio: {test_metrics['Sharpe_Ratio']:.4f}")
    print()


if __name__ == "__main__":
    main()
