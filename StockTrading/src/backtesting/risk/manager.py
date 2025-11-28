"""Aggregates individual risk controls into a single facade."""

from __future__ import annotations

from typing import Optional

from .base import PositionSnapshot, RiskAction, RiskConfig, RiskDecision
from .drawdown import DrawdownControl
from .position_limits import PositionLimitControl
from .stop_loss import StopLossControl


class RiskManager:
    """Coordinates drawdown, stop-loss, and position limit controls."""

    def __init__(self, config: RiskConfig):
        self.config = config
        self.drawdown = DrawdownControl(config=config)
        self.stop_loss = StopLossControl(config=config)
        self.position_limits = PositionLimitControl(config=config)

    def reset(self) -> None:
        self.drawdown.reset(self.config.initial_capital)

    def update_portfolio_value(self, value: float) -> None:
        self.drawdown.update_portfolio_value(value)

    def evaluate_position(self, snapshot: PositionSnapshot) -> RiskDecision:
        decision = self.drawdown.evaluate_position(snapshot)
        if decision:
            return decision

        decision = self.stop_loss.evaluate_position(snapshot)
        if decision:
            return decision

        return RiskDecision.hold()

    def allocate_position(self, capital: float, price: float, desired_fraction: float):
        """
        Clamp desired position size to limits and return (applied_fraction, shares).
        """
        return self.position_limits.allocate(capital, price, desired_fraction)

    @property
    def drawdown_active(self) -> bool:
        return self.drawdown.in_protection

    @property
    def current_drawdown(self) -> float:
        return self.drawdown.current_drawdown

    @property
    def peak_value(self) -> float:
        return self.drawdown.peak_value or self.config.initial_capital
