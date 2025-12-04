"""
Logging configuration for the API Gateway.
Re-exports from shared logging module for backward compatibility.
"""
import logging
from shared.logging import setup_logging as _setup_logging


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Setup structured JSON logging.

    Args:
        log_level: Logging level

    Returns:
        Configured logger
    """
    return _setup_logging(
        log_level=log_level,
        service_name="api-gateway"
    )
