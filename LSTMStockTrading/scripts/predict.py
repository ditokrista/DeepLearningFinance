"""
Prediction Script - Make predictions with trained model

Load a trained model and make predictions on new data.
"""

import sys
from pathlib import Path
import argparse
import joblib
import torch
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.data.loaders import load_stock_data, create_sequences
from src.data.features.technical import calculate_technical_indicators, get_default_feature_columns
from src.models.architectures.lstm_clean import get_model


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Make predictions with trained LSTM model')

    parser.add_argument('--symbol', type=str, required=True,
                        help='Stock symbol')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model file (.pth)')
    parser.add_argument('--scaler-path', type=str, required=True,
                        help='Path to scaler file (.pkl)')
    parser.add_argument('--config-path', type=str, required=True,
                        help='Path to config file (.yaml)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for predictions (optional)')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA even if available')

    return parser.parse_args()


def main():
    """Main prediction workflow"""

    args = parse_arguments()

    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config_path)

    # Set device
    if args.no_cuda:
        device = torch.device('cpu')
    else:
        device = config.device

    print(f"Using device: {device}")

    # Load scaler
    print("Loading scaler...")
    scaler = joblib.load(args.scaler_path)

    # Load data
    print(f"Loading data for {args.symbol}...")
    df = load_stock_data(args.symbol)

    # Add technical indicators
    if config.data.use_technical_indicators:
        df = calculate_technical_indicators(df)
        feature_columns = get_default_feature_columns(df)
    else:
        feature_columns = ['close']

    # Prepare data
    df = df.dropna().reset_index(drop=True)
    values = df[feature_columns].values
    dates = df['date'].values

    # Scale data
    values_scaled = scaler.transform(values)

    # Create sequences
    X, y = create_sequences(values_scaled, config.data.look_back)
    dates_adjusted = dates[config.data.look_back:]

    print(f"Created {len(X)} sequences")

    # Load model
    print("Loading model...")
    model = get_model(
        model_type=config.model.model_type,
        input_dim=config.model.input_dim,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
        output_dim=config.model.output_dim
    )
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Make predictions
    print("Making predictions...")
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        predictions = model(X_tensor).cpu().numpy()

    # Inverse transform predictions
    n_features = scaler.n_features_in_
    dummy_pred = np.zeros((len(predictions), n_features))
    dummy_pred[:, 0] = predictions.flatten()
    predictions_rescaled = scaler.inverse_transform(dummy_pred)[:, 0]

    # Inverse transform actual
    dummy_actual = np.zeros((len(y), n_features))
    dummy_actual[:, 0] = y
    actual_rescaled = scaler.inverse_transform(dummy_actual)[:, 0]

    # Create results dataframe
    results_df = pd.DataFrame({
        'date': dates_adjusted,
        'actual': actual_rescaled,
        'predicted': predictions_rescaled,
        'error': actual_rescaled - predictions_rescaled,
        'error_pct': ((actual_rescaled - predictions_rescaled) / actual_rescaled) * 100
    })

    # Print summary statistics
    print("\n" + "="*60)
    print("Prediction Summary")
    print("="*60)
    print(f"Total predictions: {len(predictions_rescaled)}")
    print(f"Mean error: ${results_df['error'].mean():.2f}")
    print(f"Mean absolute error: ${results_df['error'].abs().mean():.2f}")
    print(f"Mean percentage error: {results_df['error_pct'].abs().mean():.2f}%")
    print("="*60 + "\n")

    # Show latest predictions
    print("Latest 10 Predictions:")
    print(results_df.tail(10).to_string(index=False))
    print()

    # Save to file if specified
    if args.output:
        output_path = Path(args.output)
        results_df.to_csv(output_path, index=False)
        print(f"Predictions saved to: {output_path}")


if __name__ == "__main__":
    main()
