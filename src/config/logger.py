"""
Logging Infrastructure for Wakeword Training Platform using structlog
"""
import logging
import sys
from datetime import datetime
from pathlib import Path

import structlog


def setup_logging(log_level: str = "INFO") -> None:
    """Set up structlog logging for the entire application."""
    logging.basicConfig(level=log_level, stream=sys.stdout, format="%(message)s")

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
    return structlog.get_logger(name)


get_logger = structlog.get_logger

if __name__ == "__main__":
    setup_logging()
    logger = get_logger("test")
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")
