"""
FastAPI dependencies for Query Analysis Service.
"""
from fastapi import Request, HTTPException, status
from redis.asyncio import Redis

from src.services.query_analyzer import QueryAnalyzerService
from src.services.cache_service import CacheService
from src.services.message_queue import MessageQueueService
from src.services.consumers import ConsumerManager


def get_analyzer_service(request: Request) -> QueryAnalyzerService:
    """
    Dependency to get the query analyzer service instance.

    Args:
        request: FastAPI request object.

    Returns:
        QueryAnalyzerService instance.

    Raises:
        HTTPException: If the service is not initialized.
    """
    service = getattr(request.app.state, "analyzer_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Query analyzer service not initialized"
        )
    return service


def get_cache_service(request: Request) -> CacheService:
    """
    Dependency to get the cache service instance.

    Args:
        request: FastAPI request object.

    Returns:
        CacheService instance or None if not available.
    """
    return getattr(request.app.state, "cache_service", None)


def get_mq_service(request: Request) -> MessageQueueService:
    """
    Dependency to get the message queue service instance.

    Args:
        request: FastAPI request object.

    Returns:
        MessageQueueService instance.

    Raises:
        HTTPException: If the service is not initialized.
    """
    service = getattr(request.app.state, "mq_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Message queue service not initialized"
        )
    return service


def get_redis_client(request: Request) -> Redis:
    """
    Dependency to get the Redis client.

    Args:
        request: FastAPI request object.

    Returns:
        Redis client or None if not available.
    """
    return getattr(request.app.state, "redis_client", None)


def get_consumer_manager(request: Request) -> ConsumerManager:
    """
    Dependency to get the consumer manager.

    Args:
        request: FastAPI request object.

    Returns:
        ConsumerManager instance or None if not available.
    """
    return getattr(request.app.state, "consumer_manager", None)
