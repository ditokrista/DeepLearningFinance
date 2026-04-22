"""Position-level risk controls (stop loss, take profit, holding limits)."""

from __future__ import annotations

from typing import Optional

from .base import PositionSnapshot, RiskAction, RiskConfig, RiskDecision


class StopLossControl:
    """Encapsulates exit rules tied to the open position."""

    def __init__(self, config: RiskConfig):
        self.config = config

    def evaluate_position(self, snapshot: PositionSnapshot) -> Optional[RiskDecision]:
        if not snapshot.has_position:
            return None

        reason = None

        if snapshot.return_pct <= -self.config.stop_loss_pct:
            reason = "STOP_LOSS"
        elif snapshot.return_pct >= self.config.take_profit_pct:
            reason = "TAKE_PROFIT"
        elif snapshot.trailing_from_peak >= self.config.trailing_stop_pct:
            reason = "TRAILING_STOP"
        elif snapshot.days_held >= self.config.max_holding_period:
            reason = "MAX_HOLDING"

        if reason is None:
            return None

        return RiskDecision(
            action=RiskAction.CLOSE_POSITION,
            reason=reason,
            metadata={
                "return_pct": round(snapshot.return_pct * 100, 4),
                "trailing_from_peak": round(snapshot.trailing_from_peak * 100, 4),
                "days_held": snapshot.days_held,
            },
        )
