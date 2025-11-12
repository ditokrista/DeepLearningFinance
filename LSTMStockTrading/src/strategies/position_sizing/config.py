"""
Position Sizing Configuration Module

Type-safe configuration for position sizing algorithms.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PositionSizingConfig:
    """
    Configuration for position sizing algorithms

    Attributes:
        kelly_fraction (float): Fraction of full Kelly to use
            - 0.25 = Quarter Kelly (conservative, recommended)
            - 0.50 = Half Kelly (moderate)
            - 1.00 = Full Kelly (aggressive)

        max_position_size (float): Maximum position size as fraction of capital
            - 0.30 = 30% (default, conservative)
            - 0.50 = 50% (moderate)
            - 1.00 = 100% (aggressive, all-in)

        min_position_size (float): Minimum position size to enter trade
            - 0.05 = 5% (default)
            - Below this, don't enter trade

        use_confidence_scaling (bool): Adjust size based on prediction confidence

        use_volatility_adjustment (bool): Adjust size based on volatility regime

        volatility_lookback (int): Window for volatility calculation (days)

        vol_threshold_low (float): Annualized volatility for low regime
        vol_threshold_medium (float): Annualized volatility for medium regime
        vol_threshold_high (float): Annualized volatility for high regime

        confidence_scaling_factor (float): How aggressively to scale by confidence
            - 1.0 = Full scaling
            - 0.5 = Moderate scaling

        max_leverage (float): Maximum allowed leverage
            - 1.0 = No leverage (default)
            - 2.0 = 2x leverage

        min_samples_required (int): Minimum samples for parameter estimation
    """

    # Kelly Criterion parameters
    kelly_fraction: float = 0.25  # Quarter Kelly (conservative)
    max_leverage: float = 1.0  # No leverage by default

    # Position size constraints
    max_position_size: float = 0.30  # 30% maximum
    min_position_size: float = 0.05  # 5% minimum

    # Feature flags
    use_confidence_scaling: bool = True
    use_volatility_adjustment: bool = True

    # Volatility parameters
    volatility_lookback: int = 20  # 20 days
    vol_threshold_low: float = 0.15  # 15% annualized
    vol_threshold_medium: float = 0.25  # 25% annualized
    vol_threshold_high: float = 0.40  # 40% annualized

    # Confidence parameters
    confidence_scaling_factor: float = 1.0  # Full scaling

    # Statistical parameters
    min_samples_required: int = 30  # Minimum for parameter estimation
    trading_days_per_year: int = 252  # For annualization

    def __post_init__(self):
        """Validate configuration parameters"""
        # Validate kelly_fraction
        if not 0 < self.kelly_fraction <= 1.0:
            raise ValueError("kelly_fraction must be between 0 and 1")

        # Validate position sizes
        if not 0 < self.max_position_size <= 1.0:
            raise ValueError("max_position_size must be between 0 and 1")

        if not 0 < self.min_position_size <= self.max_position_size:
            raise ValueError(
                f"min_position_size ({self.min_position_size}) must be <= "
                f"max_position_size ({self.max_position_size})"
            )

        # Validate leverage
        if self.max_leverage < 1.0:
            raise ValueError("max_leverage must be >= 1.0")

        # Validate volatility thresholds
        if not (self.vol_threshold_low < self.vol_threshold_medium < self.vol_threshold_high):
            raise ValueError("Volatility thresholds must be in ascending order")

    def to_dict(self):
        """Convert config to dictionary"""
        return {
            'kelly_fraction': self.kelly_fraction,
            'max_leverage': self.max_leverage,
            'max_position_size': self.max_position_size,
            'min_position_size': self.min_position_size,
            'use_confidence_scaling': self.use_confidence_scaling,
            'use_volatility_adjustment': self.use_volatility_adjustment,
            'volatility_lookback': self.volatility_lookback,
            'vol_threshold_low': self.vol_threshold_low,
            'vol_threshold_medium': self.vol_threshold_medium,
            'vol_threshold_high': self.vol_threshold_high,
            'confidence_scaling_factor': self.confidence_scaling_factor,
            'min_samples_required': self.min_samples_required,
            'trading_days_per_year': self.trading_days_per_year
        }

    def __repr__(self):
        """String representation"""
        return (
            f"PositionSizingConfig(\n"
            f"  Kelly: {self.kelly_fraction:.2f} (Fractional Kelly)\n"
            f"  Position: {self.min_position_size:.1%} - {self.max_position_size:.1%}\n"
            f"  Confidence Scaling: {self.use_confidence_scaling}\n"
            f"  Volatility Adjustment: {self.use_volatility_adjustment}\n"
            f")"
        )


def get_conservative_config() -> PositionSizingConfig:
    """
    Get conservative position sizing configuration

    - Quarter Kelly (0.25)
    - Max 20% position
    - Strict volatility scaling

    Returns:
        PositionSizingConfig: Conservative configuration
    """
    return PositionSizingConfig(
        kelly_fraction=0.25,
        max_position_size=0.20,
        min_position_size=0.05,
        use_confidence_scaling=True,
        use_volatility_adjustment=True
    )


def get_moderate_config() -> PositionSizingConfig:
    """
    Get moderate position sizing configuration

    - Half Kelly (0.50)
    - Max 30% position
    - Standard volatility scaling

    Returns:
        PositionSizingConfig: Moderate configuration
    """
    return PositionSizingConfig(
        kelly_fraction=0.50,
        max_position_size=0.30,
        min_position_size=0.05,
        use_confidence_scaling=True,
        use_volatility_adjustment=True
    )


def get_aggressive_config() -> PositionSizingConfig:
    """
    Get aggressive position sizing configuration

    - Full Kelly (1.00)
    - Max 50% position
    - Less conservative scaling

    WARNING: Can be volatile. Use with caution.

    Returns:
        PositionSizingConfig: Aggressive configuration
    """
    return PositionSizingConfig(
        kelly_fraction=1.00,
        max_position_size=0.50,
        min_position_size=0.10,
        use_confidence_scaling=False,  # Don't scale, use full Kelly
        use_volatility_adjustment=True
    )


def get_default_config() -> PositionSizingConfig:
    """
    Get default position sizing configuration

    Returns:
        PositionSizingConfig: Default configuration (moderate)
    """
    return PositionSizingConfig()
