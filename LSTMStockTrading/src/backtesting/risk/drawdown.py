"""Drawdown monitoring and circuit-breaker enforcement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .base import PositionSnapshot, RiskAction, RiskConfig, RiskDecision


@dataclass
class DrawdownControl:
    """
    Tracks portfolio drawdowns and enforces circuit breakers.

    The control is stateful: call ``update_portfolio_value`` every bar to feed the
    latest NAV, then ``evaluate_position`` to determine whether open positions
    must be unwound.
    """

    config: RiskConfig
    peak_value: float = None
    current_drawdown: float = 0.0
    protection_active: bool = False

    def reset(self, initial_capital: float) -> None:
        self.peak_value = initial_capital
        self.current_drawdown = 0.0
        self.protection_active = False

    def update_portfolio_value(self, value: float) -> float:
        if self.peak_value is None:
            self.peak_value = value

        if value > self.peak_value:
            self.peak_value = value
            self.protection_active = False

        if self.peak_value == 0:
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_value - value) / self.peak_value

        if self.current_drawdown > self.config.max_drawdown_limit:
            self.protection_active = True

        return self.current_drawdown

    def evaluate_position(self, snapshot: PositionSnapshot) -> Optional[RiskDecision]:
        if not snapshot.has_position:
            return None

        if not self.protection_active:
            return None

        return RiskDecision(
            action=RiskAction.CLOSE_POSITION,
            reason="DRAWDOWN_LIMIT",
            metadata={
                "drawdown_pct": round(self.current_drawdown * 100, 4),
                "peak_value": self.peak_value,
            },
        )

    @property
    def in_protection(self) -> bool:
        return self.protection_active
