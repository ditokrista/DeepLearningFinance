"""
LSTM Stock Trading System
Production-ready quantitative trading system with deep learning
"""

__version__ = "2.0.0"
__author__ = "Deep Learning Finance"

from .utils.config_loader import load_config, get_config_value, config_loader
from .utils.logger import setup_logger, get_logger, get_logger_with_context

__all__ = [
    "load_config",
    "get_config_value",
    "config_loader",
    "setup_logger",
    "get_logger",
    "get_logger_with_context",
]
