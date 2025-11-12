"""
Position Sizing Utilities Module

Helper functions for statistics, volatility calculation, and regime detection
used in position sizing algorithms.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from enum import Enum


class VolatilityRegime(Enum):
    """Volatility regime classification"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


def estimate_win_probability(
    predictions: np.ndarray,
    actual_returns: np.ndarray,
    min_samples: int = 30
) -> float:
    """
    Estimate win probability from historical predictions

    Args:
        predictions: Model predictions (directional or magnitude)
        actual_returns: Actual returns
        min_samples: Minimum number of samples required

    Returns:
        float: Estimated win probability (0 to 1)

    Example:
        >>> predictions = np.array([0.02, -0.01, 0.03, 0.01])
        >>> actuals = np.array([0.015, -0.012, 0.028, -0.005])
        >>> win_prob = estimate_win_probability(predictions, actuals)
    """
    if len(predictions) < min_samples:
        # Default to neutral if insufficient data
        return 0.50

    # Determine if prediction direction matches actual direction
    pred_directions = np.sign(predictions)
    actual_directions = np.sign(actual_returns)

    # Calculate directional accuracy
    correct_predictions = (pred_directions == actual_directions).sum()
    win_probability = correct_predictions / len(predictions)

    return win_probability


def estimate_win_loss_ratio(
    predictions: np.ndarray,
    actual_returns: np.ndarray,
    min_samples: int = 30
) -> float:
    """
    Estimate average win to average loss ratio

    Args:
        predictions: Model predictions
        actual_returns: Actual returns
        min_samples: Minimum number of samples required

    Returns:
        float: Win/loss ratio (average_win / average_loss)

    Example:
        >>> # Average wins are 1.5x average losses
        >>> ratio = estimate_win_loss_ratio(predictions, actuals)
        >>> print(f"Win/Loss Ratio: {ratio:.2f}")
    """
    if len(predictions) < min_samples:
        # Default conservative ratio
        return 1.0

    # Calculate trade returns
    directions = np.sign(predictions)
    trade_returns = directions * actual_returns

    # Separate wins and losses
    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]

    if len(wins) == 0 or len(losses) == 0:
        return 1.0

    # Calculate averages
    avg_win = np.mean(wins)
    avg_loss = np.abs(np.mean(losses))

    # Calculate ratio
    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0

    return win_loss_ratio


def calculate_rolling_volatility(
    returns: np.ndarray,
    window: int = 20,
    annualize: bool = True,
    trading_days: int = 252
) -> np.ndarray:
    """
    Calculate rolling volatility (standard deviation of returns)

    Args:
        returns: Array of returns
        window: Rolling window size (default: 20 days)
        annualize: Whether to annualize volatility
        trading_days: Number of trading days per year

    Returns:
        np.ndarray: Rolling volatility

    Example:
        >>> returns = np.random.normal(0.001, 0.02, 100)
        >>> vol = calculate_rolling_volatility(returns, window=20)
    """
    returns_series = pd.Series(returns)
    rolling_vol = returns_series.rolling(window=window).std()

    if annualize:
        rolling_vol = rolling_vol * np.sqrt(trading_days)

    return rolling_vol.values


def detect_volatility_regime(
    current_volatility: float,
    low_threshold: float = 0.15,
    medium_threshold: float = 0.25,
    high_threshold: float = 0.40
) -> VolatilityRegime:
    """
    Detect current volatility regime

    Classifies market volatility into regimes for position size adjustment.

    Args:
        current_volatility: Current annualized volatility (e.g., 0.20 for 20%)
        low_threshold: Threshold for low volatility regime
        medium_threshold: Threshold for medium volatility regime
        high_threshold: Threshold for high volatility regime

    Returns:
        VolatilityRegime: Current regime (LOW, MEDIUM, HIGH, EXTREME)

    Example:
        >>> vol = 0.18
        >>> regime = detect_volatility_regime(vol)
        >>> print(f"Regime: {regime.value}")
        Regime: medium

    Typical annualized volatility levels:
    - Low: < 15% (calm markets)
    - Medium: 15-25% (normal markets)
    - High: 25-40% (stressed markets)
    - Extreme: > 40% (crisis markets)
    """
    if current_volatility < low_threshold:
        return VolatilityRegime.LOW
    elif current_volatility < medium_threshold:
        return VolatilityRegime.MEDIUM
    elif current_volatility < high_threshold:
        return VolatilityRegime.HIGH
    else:
        return VolatilityRegime.EXTREME


def get_regime_scaling_factor(regime: VolatilityRegime) -> float:
    """
    Get position size scaling factor based on volatility regime

    In low volatility, we can size larger positions.
    In high volatility, reduce position sizes for risk management.

    Args:
        regime: Volatility regime

    Returns:
        float: Scaling factor (multiplier for position size)

    Example:
        >>> regime = VolatilityRegime.LOW
        >>> factor = get_regime_scaling_factor(regime)
        >>> print(f"Scale factor: {factor}")
        Scale factor: 1.2
    """
    regime_factors = {
        VolatilityRegime.LOW: 1.2,      # Increase size by 20%
        VolatilityRegime.MEDIUM: 1.0,   # Normal size
        VolatilityRegime.HIGH: 0.6,     # Reduce size by 40%
        VolatilityRegime.EXTREME: 0.3   # Reduce size by 70%
    }

    return regime_factors.get(regime, 1.0)


def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe ratio for risk-adjusted returns

    Args:
        returns: Array of returns
        risk_free_rate: Annualized risk-free rate (default: 0)
        periods_per_year: Number of periods per year (252 for daily)

    Returns:
        float: Sharpe ratio

    Example:
        >>> returns = np.random.normal(0.001, 0.02, 252)
        >>> sharpe = calculate_sharpe_ratio(returns)
        >>> print(f"Sharpe: {sharpe:.2f}")
    """
    if len(returns) == 0:
        return 0.0

    mean_return = np.mean(returns)
    std_return = np.std(returns)

    if std_return == 0:
        return 0.0

    # Annualize
    annualized_return = mean_return * periods_per_year
    annualized_vol = std_return * np.sqrt(periods_per_year)

    sharpe = (annualized_return - risk_free_rate) / annualized_vol

    return sharpe


def calculate_max_drawdown(returns: np.ndarray) -> float:
    """
    Calculate maximum drawdown from returns

    Args:
        returns: Array of returns

    Returns:
        float: Maximum drawdown as decimal (e.g., 0.20 for 20% drawdown)

    Example:
        >>> returns = np.array([0.01, -0.02, -0.03, 0.02, 0.01])
        >>> mdd = calculate_max_drawdown(returns)
        >>> print(f"Max Drawdown: {mdd:.1%}")
    """
    cumulative_returns = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = np.min(drawdown)

    return abs(max_drawdown)


def calculate_expectancy(
    win_probability: float,
    avg_win: float,
    avg_loss: float
) -> float:
    """
    Calculate trading expectancy (average expected profit per trade)

    Args:
        win_probability: Probability of winning trade
        avg_win: Average winning trade size
        avg_loss: Average losing trade size (positive number)

    Returns:
        float: Expectancy per trade

    Example:
        >>> # 55% win rate, avg win $150, avg loss $100
        >>> expectancy = calculate_expectancy(0.55, 150, 100)
        >>> print(f"Expected profit per trade: ${expectancy:.2f}")
        Expected profit per trade: $37.50
    """
    loss_probability = 1 - win_probability
    expectancy = (win_probability * avg_win) - (loss_probability * avg_loss)

    return expectancy


def calculate_prediction_confidence(
    prediction: float,
    prediction_std: float,
    historical_std: float
) -> float:
    """
    Calculate confidence score for a prediction

    Confidence is higher when:
    1. Prediction magnitude is large relative to historical moves
    2. Prediction uncertainty (std) is low

    Args:
        prediction: Model prediction (expected return)
        prediction_std: Standard deviation of prediction
        historical_std: Historical standard deviation of returns

    Returns:
        float: Confidence score (0 to 1)

    Example:
        >>> confidence = calculate_prediction_confidence(
        ...     prediction=0.03,        # 3% expected return
        ...     prediction_std=0.005,   # Low uncertainty
        ...     historical_std=0.02     # 2% historical volatility
        ... )
        >>> print(f"Confidence: {confidence:.1%}")
    """
    # Signal strength: prediction magnitude relative to historical volatility
    if historical_std == 0:
        signal_strength = 0
    else:
        signal_strength = abs(prediction) / historical_std
        signal_strength = np.clip(signal_strength, 0, 3)  # Cap at 3 sigma
        signal_strength = signal_strength / 3  # Normalize to [0, 1]

    # Uncertainty penalty: high prediction std reduces confidence
    if prediction_std == 0 or historical_std == 0:
        uncertainty_factor = 1.0
    else:
        uncertainty_ratio = prediction_std / historical_std
        uncertainty_factor = 1 / (1 + uncertainty_ratio)  # Lower std = higher confidence

    # Combine factors
    confidence = signal_strength * uncertainty_factor

    return np.clip(confidence, 0, 1)


def apply_position_constraints(
    position_size: float,
    min_position: float = 0.05,
    max_position: float = 0.30
) -> float:
    """
    Apply minimum and maximum position size constraints

    Args:
        position_size: Calculated position size
        min_position: Minimum allowed position (default: 5%)
        max_position: Maximum allowed position (default: 30%)

    Returns:
        float: Constrained position size

    Example:
        >>> size = apply_position_constraints(0.45, min_position=0.05, max_position=0.30)
        >>> print(f"Constrained size: {size:.1%}")
        Constrained size: 30.0%
    """
    # Apply constraints
    constrained_size = np.clip(position_size, min_position, max_position)

    # If calculated size is below minimum, set to zero (no trade)
    if position_size < min_position:
        constrained_size = 0.0

    return constrained_size
