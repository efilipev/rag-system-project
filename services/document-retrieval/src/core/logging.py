"""
Logging configuration for the service.
Re-exports from shared logging module for backward compatibility.
"""
import logging
from shared.logging import setup_logging as _setup_logging
from src.core.config import settings


def setup_logging() -> logging.Logger:
    """
    Setup structured logging with JSON formatter.
    Wrapper around shared logging for backward compatibility.
    """
    return _setup_logging(
        log_level=settings.LOG_LEVEL,
        service_name=settings.SERVICE_NAME
    )


# Create logger instance
logger = setup_logging()
