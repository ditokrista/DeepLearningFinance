"""
Model Evaluation Module

Functions for evaluating model predictions and calculating metrics.
"""

import numpy as np
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch


def calculate_trading_metrics(y_true, y_pred):
    """
    Calculate comprehensive trading-specific metrics

    Args:
        y_true (np.ndarray): Actual values
        y_pred (np.ndarray): Predicted values

    Returns:
        dict: Dictionary of metrics
    """
    # Direction accuracy (most important for trading)
    y_true_diff = np.diff(y_true)
    y_pred_diff = np.diff(y_pred)
    direction_accuracy = np.mean((y_true_diff > 0) == (y_pred_diff > 0)) * 100

    # Standard regression metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # Calculate returns-based metrics
    returns_true = np.diff(y_true) / y_true[:-1]
    returns_pred = np.diff(y_pred) / y_pred[:-1]

    # Sharpe Ratio (assuming 252 trading days, 0% risk-free rate)
    if np.std(returns_pred) > 0:
        sharpe_ratio = np.mean(returns_pred) / np.std(returns_pred) * np.sqrt(252)
    else:
        sharpe_ratio = 0

    # Max Drawdown
    cumulative_returns = np.cumprod(1 + returns_pred)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = np.min(drawdown) * 100

    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2,
        'Direction_Accuracy': direction_accuracy,
        'Sharpe_Ratio': sharpe_ratio,
        'Max_Drawdown': max_drawdown
    }

    return metrics


def evaluate_model(model, X, y, scaler, device, phase="Test"):
    """
    Evaluate model and return predictions with metrics

    Args:
        model (nn.Module): Trained model
        X (np.ndarray): Input data
        y (np.ndarray): Target data
        scaler: Fitted scaler for inverse transform
        device (torch.device): Device to run on
        phase (str): Phase name for printing

    Returns:
        tuple: (predictions_rescaled, actual_rescaled, metrics)
    """
    from ..data.loaders import inverse_transform_predictions

    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        predictions = model(X_tensor).cpu().numpy()

    # Inverse transform to original scale
    predictions_rescaled, actual_rescaled = inverse_transform_predictions(
        predictions, y, scaler
    )

    # Calculate metrics
    metrics = calculate_trading_metrics(actual_rescaled, predictions_rescaled)

    # Print metrics
    print(f"\n{'='*60}")
    print(f"{phase} Set Metrics")
    print(f"{'='*60}")
    for metric, value in metrics.items():
        print(f"{metric:.<30} {value:.4f}")
    print(f"{'='*60}\n")

    return predictions_rescaled, actual_rescaled, metrics


def print_model_summary(model):
    """
    Print model architecture summary

    Args:
        model (nn.Module): Model to summarize
    """
    print("\nModel Architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
