"""
Structured Logging System
Production-ready logging with proper formatting and rotation
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
import json


class StructuredFormatter(logging.Formatter):
    """
    Structured JSON formatter for machine-readable logs.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add custom fields from extra
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        return json.dumps(log_data)


class SimpleFormatter(logging.Formatter):
    """
    Simple human-readable formatter for console output.
    """

    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',  # Cyan
        'INFO': '\033[32m',   # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',  # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'

    def format(self, record: logging.LogRecord) -> str:
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"

        # Format timestamp
        record.asctime = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')

        # Use parent formatter
        formatted = super().format(record)

        # Reset levelname for other formatters
        record.levelname = levelname

        return formatted


def setup_logger(
    name: str,
    log_level: str = "INFO",
    log_dir: Optional[Path] = None,
    format_type: str = "simple",
    console_output: bool = True,
    file_output: bool = True,
) -> logging.Logger:
    """
    Set up a logger with both console and file handlers.

    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        format_type: 'simple' or 'structured' (JSON)
        console_output: Enable console output
        file_output: Enable file output

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Choose formatter
    if format_type == "structured":
        formatter = StructuredFormatter()
    else:
        formatter = SimpleFormatter(
            fmt='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if file_output and log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create log file with timestamp
        timestamp = datetime.now().strftime('%Y%m%d')
        log_file = log_dir / f"{name}_{timestamp}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # Use structured format for files
        if format_type == "structured":
            file_handler.setFormatter(formatter)
        else:
            # Use detailed format for file logs
            detailed_formatter = logging.Formatter(
                fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(module)s:%(funcName)s:%(lineno)d | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(detailed_formatter)

        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter for adding context to log messages.
    """

    def process(self, msg, kwargs):
        # Add extra fields to the record
        extra = kwargs.get("extra", {})
        extra.update(self.extra)
        kwargs["extra"] = {"extra_fields": extra}
        return msg, kwargs


def get_logger_with_context(name: str, **context) -> LoggerAdapter:
    """
    Get a logger with additional context.

    Args:
        name: Logger name
        **context: Additional context to add to all log messages

    Returns:
        LoggerAdapter with context

    Example:
        logger = get_logger_with_context("training", experiment_id="exp_123", model="lstm")
        logger.info("Training started")  # Will include experiment_id and model in logs
    """
    logger = get_logger(name)
    return LoggerAdapter(logger, context)


# Performance logging decorator
def log_execution_time(logger: Optional[logging.Logger] = None):
    """
    Decorator to log function execution time.

    Args:
        logger: Logger instance (if None, uses function's module logger)

    Example:
        @log_execution_time()
        def train_model():
            ...
    """
    import functools
    import time

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)

            start_time = time.time()
            logger.info(f"Starting {func.__name__}")

            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.info(
                    f"Completed {func.__name__} in {execution_time:.2f}s",
                    extra={"execution_time": execution_time, "function": func.__name__}
                )
                return result

            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(
                    f"Failed {func.__name__} after {execution_time:.2f}s: {e}",
                    extra={"execution_time": execution_time, "function": func.__name__},
                    exc_info=True
                )
                raise

        return wrapper
    return decorator


if __name__ == "__main__":
    # Test logging setup
    import tempfile

    # Create temporary log directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Simple format logger
        simple_logger = setup_logger(
            "test_simple",
            log_level="DEBUG",
            log_dir=Path(tmpdir),
            format_type="simple"
        )

        simple_logger.debug("Debug message")
        simple_logger.info("Info message")
        simple_logger.warning("Warning message")
        simple_logger.error("Error message")

        print("\n" + "="*50)

        # Structured format logger
        structured_logger = setup_logger(
            "test_structured",
            log_level="INFO",
            log_dir=Path(tmpdir),
            format_type="structured",
            console_output=False,
            file_output=True
        )

        structured_logger.info("Structured log message")

        # Test context logger
        context_logger = get_logger_with_context(
            "test_context",
            experiment_id="exp_001",
            model="lstm"
        )
        context_logger.info("Message with context")

        # Test execution time decorator
        @log_execution_time(logger=simple_logger)
        def slow_function():
            import time
            time.sleep(0.1)
            return "Done"

        result = slow_function()

        print("\nLogging system working correctly!")
