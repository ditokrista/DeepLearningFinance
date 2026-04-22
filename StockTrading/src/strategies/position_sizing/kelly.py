"""
Kelly Criterion Position Sizing Module

Implements the Kelly Criterion for optimal position sizing in quantitative trading.
The Kelly Criterion maximizes the expected geometric growth rate of wealth.

Formula: f* = (p * b - q) / b
Where:
- f* = fraction of capital to risk
- p = probability of winning
- q = probability of losing (1-p)
- b = win/loss ratio (average_win / average_loss)
"""

import numpy as np
from typing import Optional, Tuple


class KellyCriterion:
    """
    Kelly Criterion position sizing calculator

    The Kelly Criterion provides mathematically optimal position sizing
    that maximizes long-term growth rate. In practice, fractional Kelly
    (e.g., 1/4 Kelly or 1/2 Kelly) is often used for more conservative sizing.

    Args:
        kelly_fraction (float): Fraction of full Kelly to use (default: 0.25)
            0.25 = Quarter Kelly (conservative, recommended)
            0.50 = Half Kelly (moderate)
            1.00 = Full Kelly (aggressive, can be volatile)
        max_leverage (float): Maximum allowed leverage (default: 1.0 = no leverage)
    """

    def __init__(self, kelly_fraction: float = 0.25, max_leverage: float = 1.0):
        if not 0 < kelly_fraction <= 1.0:
            raise ValueError("kelly_fraction must be between 0 and 1")
        if max_leverage < 1.0:
            raise ValueError("max_leverage must be >= 1.0")

        self.kelly_fraction = kelly_fraction
        self.max_leverage = max_leverage

    def calculate_kelly_fraction(
        self,
        win_probability: float,
        win_loss_ratio: float
    ) -> float:
        """
        Calculate optimal Kelly fraction

        Args:
            win_probability (float): Probability of winning trade (0 to 1)
            win_loss_ratio (float): Ratio of average win to average loss (e.g., 1.5)

        Returns:
            float: Optimal position size as fraction of capital

        Example:
            >>> kelly = KellyCriterion(kelly_fraction=0.25)
            >>> # 55% win rate, average wins are 1.5x average losses
            >>> size = kelly.calculate_kelly_fraction(0.55, 1.5)
            >>> print(f"Position size: {size:.2%}")
            Position size: 5.42%
        """
        # Validate inputs
        if not 0 < win_probability < 1:
            raise ValueError("win_probability must be between 0 and 1")
        if win_loss_ratio <= 0:
            raise ValueError("win_loss_ratio must be positive")

        # Calculate Kelly formula: f* = (p * b - q) / b
        p = win_probability
        q = 1 - win_probability
        b = win_loss_ratio

        # Full Kelly fraction
        kelly_full = (p * b - q) / b

        # Apply fractional Kelly for risk control
        kelly_fractional = kelly_full * self.kelly_fraction

        # Clip to [0, max_leverage]
        kelly_fractional = np.clip(kelly_fractional, 0, self.max_leverage)

        return kelly_fractional

    def calculate_from_returns(
        self,
        historical_returns: np.ndarray,
        predictions: np.ndarray,
        actual_returns: np.ndarray
    ) -> float:
        """
        Calculate Kelly fraction from historical prediction performance

        Automatically estimates win probability and win/loss ratio from
        historical predictions and outcomes.

        Args:
            historical_returns (np.ndarray): Historical returns
            predictions (np.ndarray): Model predictions (directional or magnitude)
            actual_returns (np.ndarray): Actual returns

        Returns:
            float: Optimal position size as fraction of capital

        Example:
            >>> kelly = KellyCriterion()
            >>> size = kelly.calculate_from_returns(returns, predictions, actuals)
        """
        # Estimate win probability and win/loss ratio
        win_prob, win_loss_ratio = self._estimate_parameters(
            predictions, actual_returns
        )

        # Calculate Kelly fraction
        return self.calculate_kelly_fraction(win_prob, win_loss_ratio)

    def _estimate_parameters(
        self,
        predictions: np.ndarray,
        actual_returns: np.ndarray
    ) -> Tuple[float, float]:
        """
        Estimate win probability and win/loss ratio from historical data

        Args:
            predictions: Model predictions
            actual_returns: Actual returns

        Returns:
            tuple: (win_probability, win_loss_ratio)
        """
        # Determine trade direction (1 for long, -1 for short)
        directions = np.sign(predictions)

        # Calculate trade returns (prediction direction * actual return)
        trade_returns = directions * actual_returns

        # Separate winning and losing trades
        wins = trade_returns[trade_returns > 0]
        losses = trade_returns[trade_returns < 0]

        # Handle edge cases
        if len(wins) == 0 or len(losses) == 0:
            # Default conservative values if insufficient data
            return 0.5, 1.0

        # Calculate win probability
        win_probability = len(wins) / len(trade_returns)

        # Calculate average win and average loss
        avg_win = np.mean(wins)
        avg_loss = np.abs(np.mean(losses))  # Make positive

        # Calculate win/loss ratio
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0

        return win_probability, win_loss_ratio

    def calculate_with_expected_return(
        self,
        expected_return: float,
        return_volatility: float,
        sharpe_approximation: bool = True
    ) -> float:
        """
        Calculate Kelly fraction using expected return and volatility

        Alternative Kelly formula for continuous returns:
        f* = (mu - r) / sigma^2

        Where:
        - mu = expected return
        - r = risk-free rate (assumed 0)
        - sigma^2 = variance of returns

        Args:
            expected_return (float): Expected return (e.g., 0.10 for 10%)
            return_volatility (float): Standard deviation of returns
            sharpe_approximation (bool): Use Sharpe ratio approximation

        Returns:
            float: Optimal position size as fraction of capital
        """
        if return_volatility <= 0:
            return 0.0

        if sharpe_approximation:
            # Simplified: f* ≈ Sharpe Ratio / sigma
            # Assumes Sharpe = expected_return / volatility
            kelly_full = expected_return / (return_volatility ** 2)
        else:
            # Standard formula
            kelly_full = expected_return / (return_volatility ** 2)

        # Apply fractional Kelly
        kelly_fractional = kelly_full * self.kelly_fraction

        # Clip to valid range
        kelly_fractional = np.clip(kelly_fractional, 0, self.max_leverage)

        return kelly_fractional

    def adjust_for_confidence(
        self,
        base_kelly: float,
        confidence_score: float,
        scaling_factor: float = 1.0
    ) -> float:
        """
        Adjust Kelly fraction based on prediction confidence

        Higher confidence → larger position
        Lower confidence → smaller position

        Args:
            base_kelly (float): Base Kelly fraction
            confidence_score (float): Confidence in prediction (0 to 1)
            scaling_factor (float): How aggressively to scale (default: 1.0)

        Returns:
            float: Confidence-adjusted Kelly fraction

        Example:
            >>> kelly = KellyCriterion()
            >>> base = 0.10  # 10% position
            >>> adjusted = kelly.adjust_for_confidence(base, confidence=0.8)
            >>> # With 80% confidence, might increase to 12-15%
        """
        if not 0 <= confidence_score <= 1:
            confidence_score = np.clip(confidence_score, 0, 1)

        # Scale Kelly by confidence
        # confidence=1.0 → no change
        # confidence=0.5 → reduce by 50%
        confidence_multiplier = 0.5 + (confidence_score * 0.5 * scaling_factor)

        adjusted_kelly = base_kelly * confidence_multiplier

        return adjusted_kelly


def calculate_optimal_position_size(
    win_probability: float,
    win_loss_ratio: float,
    kelly_fraction: float = 0.25,
    max_position: float = 1.0
) -> float:
    """
    Convenience function to calculate optimal position size

    Args:
        win_probability: Probability of winning trade (0 to 1)
        win_loss_ratio: Average win / average loss ratio
        kelly_fraction: Fraction of full Kelly to use
        max_position: Maximum allowed position size

    Returns:
        float: Optimal position size as fraction of capital

    Example:
        >>> size = calculate_optimal_position_size(0.55, 1.5, kelly_fraction=0.25)
        >>> print(f"Position: {size:.1%}")
        Position: 5.4%
    """
    kelly = KellyCriterion(kelly_fraction=kelly_fraction, max_leverage=max_position)
    return kelly.calculate_kelly_fraction(win_probability, win_loss_ratio)
