"""
Core risk-management data structures.

These light-weight dataclasses isolate the information flowing between
the backtest engine and the individual risk controls so we can reason
about risk decisions in a unit-testable way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class RiskAction(Enum):
    """Possible outcomes of a risk evaluation."""

    HOLD = "hold"
    CLOSE_POSITION = "close_position"
    HALT_TRADING = "halt_trading"
    BLOCK_ENTRY = "block_entry"


@dataclass
class PositionSnapshot:
    """
    Immutable view of the current position/portfolio state used by the controls.
    """

    symbol: str
    timestamp: Any
    position: int
    shares: int
    entry_price: float
    current_price: float
    days_held: int
    capital: float
    portfolio_value: float
    position_peak_price: float

    @property
    def has_position(self) -> bool:
        return self.position != 0 and self.shares > 0

    @property
    def return_pct(self) -> float:
        if not self.has_position or self.entry_price == 0:
            return 0.0
        return (self.current_price / self.entry_price) - 1

    @property
    def trailing_from_peak(self) -> float:
        if self.position_peak_price <= 0:
            return 0.0
        return (self.position_peak_price - self.current_price) / self.position_peak_price


@dataclass
class RiskDecision:
    """Result returned by each risk control."""

    action: RiskAction
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def hold(cls) -> "RiskDecision":
        return cls(action=RiskAction.HOLD, reason="NONE")


@dataclass
class RiskConfig:
    """Normalized risk configuration consumed by the manager."""

    initial_capital: float
    max_position_size: float
    min_position_size: float
    max_leverage: float
    stop_loss_pct: float
    take_profit_pct: float
    trailing_stop_pct: float
    partial_profit_pct: float
    max_holding_period: int
    transaction_cost: float
    max_drawdown_limit: float
    drawdown_scale_factor: float

    @classmethod
    def from_backtest_config(cls, backtest_config: Any) -> "RiskConfig":
        """
        Build a RiskConfig from the BacktestConfig defined in the engine.
        Uses duck typing to avoid circular imports.
        """

        return cls(
            initial_capital=getattr(backtest_config, "initial_capital", 1_000_000),
            max_position_size=getattr(backtest_config, "max_position_size", 1.0),
            min_position_size=getattr(backtest_config, "min_position_size", 0.0),
            max_leverage=getattr(backtest_config, "max_leverage", 1.0),
            stop_loss_pct=getattr(backtest_config, "stop_loss_pct", 0.05),
            take_profit_pct=getattr(backtest_config, "take_profit_pct", 0.1),
            trailing_stop_pct=getattr(backtest_config, "trailing_stop_pct", 0.1),
            partial_profit_pct=getattr(backtest_config, "partial_profit_pct", 0.0),
            max_holding_period=getattr(backtest_config, "max_holding_period", 60),
            transaction_cost=getattr(backtest_config, "transaction_cost", 0.0),
            max_drawdown_limit=getattr(backtest_config, "max_drawdown_limit", 0.2),
            drawdown_scale_factor=getattr(backtest_config, "drawdown_scale_factor", 0.5),
        )
