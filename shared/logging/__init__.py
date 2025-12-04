"""
Shared logging utilities for RAG System microservices.
Provides consistent JSON-formatted logging across all services.
"""
from shared.logging.json_logging import setup_logging

__all__ = ["setup_logging"]
