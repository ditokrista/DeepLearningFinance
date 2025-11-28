"""Position limit enforcement and capital allocation helpers."""

from __future__ import annotations

from typing import Tuple

from .base import RiskConfig


class PositionLimitControl:
    """
    Bounds desired position sizes by the configured leverage and sizing limits.

    The control returns both the clipped position fraction and the number of
    shares to purchase given current capital and price.
    """

    def __init__(self, config: RiskConfig):
        self.config = config

    def allocate(self, capital: float, price: float, desired_fraction: float) -> Tuple[float, int]:
        if price <= 0 or capital <= 0:
            return 0.0, 0

        fraction = min(desired_fraction, self.config.max_position_size)
        fraction = max(fraction, self.config.min_position_size)

        notional = capital * fraction

        if self.config.max_leverage > 0:
            notional = min(notional, capital * self.config.max_leverage)

        shares = int(notional / price)

        return fraction, max(shares, 0)
