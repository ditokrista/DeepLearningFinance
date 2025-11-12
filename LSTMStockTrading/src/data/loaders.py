"""
Data Loading and Preparation Module

Functions for loading stock data, creating sequences, and preparing datasets.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset

from .features.technical import calculate_technical_indicators, get_default_feature_columns


def load_stock_data(symbol, data_dir=None):
    """
    Load stock price data from CSV file

    Args:
        symbol (str): Stock symbol (e.g., 'AAPL', 'MSFT')
        data_dir (Path or str, optional): Directory containing data files

    Returns:
        pd.DataFrame: Stock data with date column
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    else:
        data_dir = Path(data_dir)

    file_path = data_dir / f"{symbol}.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    return df


def create_sequences(data, look_back):
    """
    Create sequences for time series prediction

    Args:
        data (np.ndarray): Array of shape (n_samples, n_features)
        look_back (int): Number of time steps to look back

    Returns:
        tuple: (X, y) where X is (n_sequences, look_back, n_features) and y is (n_sequences,)
    """
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back, 0])  # Predict close price (first column)
    return np.array(X), np.array(y)


def prepare_data(
    symbol,
    look_back=60,
    train_ratio=0.7,
    validation_ratio=0.15,
    use_technical_indicators=True,
    feature_columns=None,
    scaler_type='minmax',
    data_dir=None
):
    """
    Prepare data with proper scaling and feature engineering

    Key approach to prevent data leakage:
    1. Fit scaler ONLY on training data
    2. Transform entire dataset with training scaler
    3. Create sequences from entire scaled dataset
    4. Split sequences into train/val/test

    This ensures:
    - No data leakage (scaler never sees val/test statistics)
    - Continuous timeline (no jumps between sets)

    Args:
        symbol (str): Stock symbol
        look_back (int): Sequence length
        train_ratio (float): Training data ratio
        validation_ratio (float): Validation data ratio
        use_technical_indicators (bool): Whether to add technical indicators
        feature_columns (list, optional): Specific features to use
        scaler_type (str): Type of scaler ('minmax' or 'standard')
        data_dir (Path or str, optional): Data directory

    Returns:
        dict: Dictionary containing:
            - X_train, y_train, X_val, y_val, X_test, y_test
            - scaler, train_dates, val_dates, test_dates, feature_columns
    """
    # Load data
    df = load_stock_data(symbol, data_dir)

    # Add technical indicators
    if use_technical_indicators:
        df = calculate_technical_indicators(df)

        # Select features (close price must be first for prediction target)
        if feature_columns is None:
            feature_columns = get_default_feature_columns(df)
    else:
        feature_columns = ['close']

    # Remove NaN values created by technical indicators
    df = df.dropna().reset_index(drop=True)

    # Extract values and dates
    values = df[feature_columns].values
    dates = df['date'].values

    # Split data BEFORE scaling to prevent data leakage
    train_size = int(len(values) * train_ratio)
    train_data = values[:train_size]

    # Fit scaler ONLY on training data
    if scaler_type == 'minmax':
        scaler = MinMaxScaler(feature_range=(-1, 1))
    elif scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")

    scaler.fit(train_data)

    # Transform the ENTIRE dataset with training statistics
    values_scaled = scaler.transform(values)

    # Create sequences from the entire scaled dataset
    X_all, y_all = create_sequences(values_scaled, look_back)

    # Adjust dates to account for sequences
    dates_adjusted = dates[look_back:]

    # Split sequences into train/val/test
    val_size = int(len(values) * validation_ratio)
    train_seq_size = train_size - look_back
    val_seq_size = val_size

    X_train = X_all[:train_seq_size]
    y_train = y_all[:train_seq_size]

    X_val = X_all[train_seq_size:train_seq_size + val_seq_size]
    y_val = y_all[train_seq_size:train_seq_size + val_seq_size]

    X_test = X_all[train_seq_size + val_seq_size:]
    y_test = y_all[train_seq_size + val_seq_size:]

    # Split dates accordingly
    train_dates = dates_adjusted[:train_seq_size]
    val_dates = dates_adjusted[train_seq_size:train_seq_size + val_seq_size]
    test_dates = dates_adjusted[train_seq_size + val_seq_size:]

    # Print summary
    print(f"\n{'='*60}")
    print(f"Data Preparation Summary")
    print(f"{'='*60}")
    print(f"Symbol: {symbol}")
    print(f"Features used: {len(feature_columns)}")
    print(f"Feature names: {feature_columns}")
    print(f"Look-back period: {look_back}")
    print(f"Train shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Validation shape: X={X_val.shape}, y={y_val.shape}")
    print(f"Test shape: X={X_test.shape}, y={y_test.shape}")
    print(f"{'='*60}\n")

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'scaler': scaler,
        'train_dates': train_dates,
        'val_dates': val_dates,
        'test_dates': test_dates,
        'feature_columns': feature_columns,
        'n_features': len(feature_columns)
    }


def create_data_loaders(X_train, y_train, X_val, y_val, batch_size=32, shuffle_train=True):
    """
    Create PyTorch DataLoaders from numpy arrays

    Args:
        X_train, y_train, X_val, y_val: Training and validation data
        batch_size (int): Batch size for data loaders
        shuffle_train (bool): Whether to shuffle training data

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader


def inverse_transform_predictions(predictions, actual, scaler):
    """
    Inverse transform predictions and actual values back to original scale

    Args:
        predictions (np.ndarray): Scaled predictions
        actual (np.ndarray): Scaled actual values
        scaler: Fitted scaler object

    Returns:
        tuple: (predictions_rescaled, actual_rescaled)
    """
    n_features = scaler.n_features_in_

    # Create dummy arrays for inverse transform
    dummy_pred = np.zeros((len(predictions), n_features))
    dummy_pred[:, 0] = predictions.flatten()
    predictions_rescaled = scaler.inverse_transform(dummy_pred)[:, 0]

    dummy_actual = np.zeros((len(actual), n_features))
    dummy_actual[:, 0] = actual
    actual_rescaled = scaler.inverse_transform(dummy_actual)[:, 0]

    return predictions_rescaled, actual_rescaled
