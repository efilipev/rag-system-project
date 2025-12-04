"""
Shared JSON logging configuration for all RAG System microservices.
Provides consistent structured logging with JSON output.
"""
import logging
import sys
from typing import Optional

from pythonjsonlogger import jsonlogger


def setup_logging(
    log_level: str = "INFO",
    service_name: Optional[str] = None
) -> logging.Logger:
    """
    Setup structured logging with JSON formatter.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        service_name: Name of the service (used as logger name)

    Returns:
        Configured logger instance
    """
    logger_name = service_name or "rag-service"
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)

    # Create JSON formatter with standardized field names
    formatter = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
        rename_fields={
            "asctime": "timestamp",
            "name": "service",
            "levelname": "level",
        },
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
