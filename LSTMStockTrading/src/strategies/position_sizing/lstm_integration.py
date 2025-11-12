"""
LSTM Position Sizing Integration Module

Integrates Kelly Criterion position sizing with LSTM predictions,
including confidence scaling and volatility regime adjustment.
"""

import numpy as np
from typing import Optional, Dict
import warnings

from .kelly import KellyCriterion
from .utils import (
    estimate_win_probability,
    estimate_win_loss_ratio,
    calculate_rolling_volatility,
    detect_volatility_regime,
    get_regime_scaling_factor,
    calculate_prediction_confidence,
    apply_position_constraints
)
from .config import PositionSizingConfig, get_default_config


class LSTMPositionSizer:
    """
    Position sizer that integrates LSTM predictions with Kelly Criterion

    Workflow:
    1. Calculate base Kelly fraction from historical performance
    2. Adjust for prediction confidence (if enabled)
    3. Detect volatility regime
    4. Apply regime scaling (if enabled)
    5. Enforce position constraints

    Args:
        config (PositionSizingConfig): Configuration parameters
        historical_predictions (np.ndarray, optional): Historical predictions for calibration
        historical_returns (np.ndarray, optional): Historical returns for calibration

    Example:
        >>> from src.strategies.position_sizing.lstm_integration import LSTMPositionSizer
        >>> from src.strategies.position_sizing.config import get_conservative_config
        >>>
        >>> # Initialize with historical data
        >>> sizer = LSTMPositionSizer(
        ...     config=get_conservative_config(),
        ...     historical_predictions=hist_preds,
        ...     historical_returns=hist_returns
        ... )
        >>>
        >>> # Size position for new prediction
        >>> position_size = sizer.size_position(
        ...     prediction=0.025,           # 2.5% expected return
        ...     confidence=0.80,            # 80% confidence
        ...     current_price=150.0,
        ...     recent_returns=returns[-60:]
        ... )
        >>> print(f"Position size: {position_size:.1%}")
    """

    def __init__(
        self,
        config: Optional[PositionSizingConfig] = None,
        historical_predictions: Optional[np.ndarray] = None,
        historical_returns: Optional[np.ndarray] = None
    ):
        self.config = config or get_default_config()

        # Initialize Kelly calculator
        self.kelly = KellyCriterion(
            kelly_fraction=self.config.kelly_fraction,
            max_leverage=self.config.max_leverage
        )

        # Store historical data for calibration
        self.historical_predictions = historical_predictions
        self.historical_returns = historical_returns

        # Pre-calculate win probability and win/loss ratio if data provided
        if historical_predictions is not None and historical_returns is not None:
            self.win_probability, self.win_loss_ratio = self._calibrate_parameters()
        else:
            # Use default conservative values
            self.win_probability = 0.50
            self.win_loss_ratio = 1.0

    def size_position(
        self,
        prediction: float,
        confidence: Optional[float] = None,
        current_price: Optional[float] = None,
        recent_returns: Optional[np.ndarray] = None,
        prediction_std: Optional[float] = None
    ) -> float:
        """
        Calculate optimal position size for a given prediction

        Args:
            prediction (float): LSTM prediction (expected return, e.g., 0.025 for 2.5%)
            confidence (float, optional): Prediction confidence (0 to 1)
            current_price (float, optional): Current asset price
            recent_returns (np.ndarray, optional): Recent returns for volatility calculation
            prediction_std (float, optional): Standard deviation of prediction

        Returns:
            float: Position size as fraction of capital (e.g., 0.15 for 15%)

        Example:
            >>> size = sizer.size_position(
            ...     prediction=0.03,         # 3% expected return
            ...     confidence=0.85,         # 85% confidence
            ...     recent_returns=returns
            ... )
        """
        # Step 1: Calculate base Kelly fraction
        base_kelly = self.kelly.calculate_kelly_fraction(
            self.win_probability,
            self.win_loss_ratio
        )

        # Step 2: Adjust for prediction confidence (if enabled and provided)
        if self.config.use_confidence_scaling and confidence is not None:
            adjusted_kelly = self.kelly.adjust_for_confidence(
                base_kelly,
                confidence,
                scaling_factor=self.config.confidence_scaling_factor
            )
        else:
            adjusted_kelly = base_kelly

        # Step 3: Detect volatility regime and adjust (if enabled)
        if self.config.use_volatility_adjustment and recent_returns is not None:
            regime_adjusted_kelly = self._apply_volatility_regime_adjustment(
                adjusted_kelly,
                recent_returns
            )
        else:
            regime_adjusted_kelly = adjusted_kelly

        # Step 4: Apply position constraints
        final_position_size = apply_position_constraints(
            regime_adjusted_kelly,
            min_position=self.config.min_position_size,
            max_position=self.config.max_position_size
        )

        return final_position_size

    def size_position_from_prediction(
        self,
        prediction: float,
        current_price: float,
        recent_returns: np.ndarray,
        prediction_std: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate position size with automatic confidence estimation

        Automatically estimates confidence from prediction magnitude and uncertainty.

        Args:
            prediction (float): Predicted price or return
            current_price (float): Current asset price
            recent_returns (np.ndarray): Recent historical returns
            prediction_std (float, optional): Prediction uncertainty

        Returns:
            dict: Contains 'position_size', 'confidence', 'regime', 'base_kelly'

        Example:
            >>> result = sizer.size_position_from_prediction(
            ...     prediction=152.5,
            ...     current_price=150.0,
            ...     recent_returns=returns
            ... )
            >>> print(f"Position: {result['position_size']:.1%}")
            >>> print(f"Confidence: {result['confidence']:.1%}")
        """
        # Calculate expected return
        if prediction > current_price or prediction < 0:
            # Prediction is a price
            expected_return = (prediction - current_price) / current_price
        else:
            # Prediction is already a return
            expected_return = prediction

        # Estimate confidence if not provided
        if prediction_std is not None and recent_returns is not None:
            historical_std = np.std(recent_returns)
            confidence = calculate_prediction_confidence(
                expected_return,
                prediction_std,
                historical_std
            )
        else:
            # Default moderate confidence
            confidence = 0.70

        # Calculate base Kelly
        base_kelly = self.kelly.calculate_kelly_fraction(
            self.win_probability,
            self.win_loss_ratio
        )

        # Calculate final position size
        position_size = self.size_position(
            prediction=expected_return,
            confidence=confidence,
            current_price=current_price,
            recent_returns=recent_returns,
            prediction_std=prediction_std
        )

        # Detect regime
        regime = None
        if recent_returns is not None:
            current_vol = calculate_rolling_volatility(
                recent_returns,
                window=self.config.volatility_lookback,
                annualize=True,
                trading_days=self.config.trading_days_per_year
            )[-1]
            regime = detect_volatility_regime(
                current_vol,
                self.config.vol_threshold_low,
                self.config.vol_threshold_medium,
                self.config.vol_threshold_high
            )

        return {
            'position_size': position_size,
            'confidence': confidence,
            'expected_return': expected_return,
            'base_kelly': base_kelly,
            'regime': regime.value if regime else None
        }

    def update_calibration(
        self,
        new_predictions: np.ndarray,
        new_returns: np.ndarray
    ):
        """
        Update position sizer calibration with new data

        Recalculates win probability and win/loss ratio with updated history.

        Args:
            new_predictions: New prediction data
            new_returns: New actual returns data

        Example:
            >>> # After trading for a period, update calibration
            >>> sizer.update_calibration(recent_predictions, recent_returns)
        """
        # Append to historical data
        if self.historical_predictions is None:
            self.historical_predictions = new_predictions
            self.historical_returns = new_returns
        else:
            self.historical_predictions = np.concatenate([
                self.historical_predictions, new_predictions
            ])
            self.historical_returns = np.concatenate([
                self.historical_returns, new_returns
            ])

        # Recalibrate parameters
        self.win_probability, self.win_loss_ratio = self._calibrate_parameters()

    def _calibrate_parameters(self) -> tuple:
        """
        Calibrate win probability and win/loss ratio from historical data

        Returns:
            tuple: (win_probability, win_loss_ratio)
        """
        if self.historical_predictions is None or self.historical_returns is None:
            return 0.50, 1.0

        win_prob = estimate_win_probability(
            self.historical_predictions,
            self.historical_returns,
            min_samples=self.config.min_samples_required
        )

        win_loss = estimate_win_loss_ratio(
            self.historical_predictions,
            self.historical_returns,
            min_samples=self.config.min_samples_required
        )

        return win_prob, win_loss

    def _apply_volatility_regime_adjustment(
        self,
        position_size: float,
        recent_returns: np.ndarray
    ) -> float:
        """
        Adjust position size based on current volatility regime

        Args:
            position_size: Base position size
            recent_returns: Recent returns for volatility calculation

        Returns:
            float: Regime-adjusted position size
        """
        # Calculate current volatility
        rolling_vol = calculate_rolling_volatility(
            recent_returns,
            window=self.config.volatility_lookback,
            annualize=True,
            trading_days=self.config.trading_days_per_year
        )

        # Get current volatility (last value)
        current_vol = rolling_vol[-1] if len(rolling_vol) > 0 else 0.20

        # Detect regime
        regime = detect_volatility_regime(
            current_vol,
            self.config.vol_threshold_low,
            self.config.vol_threshold_medium,
            self.config.vol_threshold_high
        )

        # Get scaling factor
        scaling_factor = get_regime_scaling_factor(regime)

        # Apply scaling
        adjusted_size = position_size * scaling_factor

        return adjusted_size

    def get_sizing_summary(self) -> Dict[str, float]:
        """
        Get summary of current sizing parameters

        Returns:
            dict: Summary of parameters

        Example:
            >>> summary = sizer.get_sizing_summary()
            >>> print(f"Win Rate: {summary['win_probability']:.1%}")
            >>> print(f"Win/Loss Ratio: {summary['win_loss_ratio']:.2f}")
        """
        return {
            'win_probability': self.win_probability,
            'win_loss_ratio': self.win_loss_ratio,
            'kelly_fraction': self.config.kelly_fraction,
            'max_position': self.config.max_position_size,
            'min_position': self.config.min_position_size,
            'use_confidence_scaling': self.config.use_confidence_scaling,
            'use_volatility_adjustment': self.config.use_volatility_adjustment
        }


def quick_size_position(
    prediction: float,
    current_price: float,
    historical_predictions: np.ndarray,
    historical_returns: np.ndarray,
    recent_returns: np.ndarray,
    confidence: Optional[float] = None,
    kelly_fraction: float = 0.25
) -> float:
    """
    Convenience function for quick position sizing

    Args:
        prediction: Predicted price or return
        current_price: Current asset price
        historical_predictions: Historical predictions for calibration
        historical_returns: Historical returns for calibration
        recent_returns: Recent returns for volatility
        confidence: Optional confidence score
        kelly_fraction: Fractional Kelly to use

    Returns:
        float: Position size as fraction of capital

    Example:
        >>> size = quick_size_position(
        ...     prediction=155.0,
        ...     current_price=150.0,
        ...     historical_predictions=hist_preds,
        ...     historical_returns=hist_rets,
        ...     recent_returns=recent_rets
        ... )
    """
    config = PositionSizingConfig(kelly_fraction=kelly_fraction)

    sizer = LSTMPositionSizer(
        config=config,
        historical_predictions=historical_predictions,
        historical_returns=historical_returns
    )

    return sizer.size_position(
        prediction=prediction,
        confidence=confidence,
        current_price=current_price,
        recent_returns=recent_returns
    )
