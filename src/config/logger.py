"""
Logging Infrastructure for Wakeword Training Platform using structlog
"""

import logging
import sys
from typing import List

import structlog


def setup_logging(log_level: str = "INFO") -> None:
    """Set up structlog logging for the entire application."""
    # Create logs directory
    from datetime import datetime
    from pathlib import Path

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Use timestamp for unique log files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"wakeword_training_{timestamp}.log"

    handlers: List[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(log_file), encoding="utf-8"),
    ]

    logging.basicConfig(level=log_level, format="%(message)s", handlers=handlers, force=True)

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_data_logger(name: str = "data") -> structlog.stdlib.BoundLogger:
    """
    Get a structlog logger instance for dataset or data operations.

    Args:
        name: Logger name (default: "data").

    Returns:
        A structlog logger instance.
    """
    return structlog.get_logger(name)  # type: ignore


get_logger = structlog.get_logger

if __name__ == "__main__":
    setup_logging()
    logger = get_logger("test")
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")
