"""
API routes for Query Analysis Service.
"""
import time
import uuid
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request, Query, Depends

from src.core.config import settings
from src.core.logging import logger
from src.models.schemas import QueryRequest, QueryResponse
from src.services.query_analyzer import QueryAnalyzerService
from src.services.cache_service import CacheService
from src.api.dependencies import (
    get_analyzer_service,
    get_cache_service,
    get_mq_service,
    get_redis_client,
    get_consumer_manager,
)

router = APIRouter()


@router.post("/analyze", response_model=QueryResponse)
async def analyze_query(
    query_request: QueryRequest,
    analyzer_service: QueryAnalyzerService = Depends(get_analyzer_service),
    mq_service = Depends(get_mq_service),
) -> QueryResponse:
    """
    Analyze a user query.

    :param query_request: Query request with query text and options.
    :param analyzer_service: Injected QueryAnalyzerService instance.
    :param mq_service: Injected MessageQueueService instance (optional).
    :return: Query analysis response.
    :raises HTTPException: If query analysis fails.
    """
    start_time = time.time()
    query_id = str(uuid.uuid4())

    try:
        logger.info(
            f"Received query analysis request for: {query_request.query[:100]}",
            extra={"query_id": query_id, "user_id": query_request.user_id}
        )

        # Perform analysis
        analysis = await analyzer_service.analyze_query(query_request.query)

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000

        # Publish to message queue for downstream processing (consumers will pick this up)
        if mq_service:
            try:
                await mq_service.publish_message(
                    routing_key="query.analyzed",
                    message={
                        "query_id": query_id,
                        "query": query_request.query,
                        "analysis": analysis.model_dump(),
                        "user_id": query_request.user_id or "anonymous",
                        "session_id": query_request.session_id,
                        "timestamp": datetime.utcnow().isoformat(),
                        "processing_time_ms": processing_time,
                    },
                    correlation_id=query_id,
                )
                logger.debug(f"Published query analysis event for query {query_id}")
            except Exception as e:
                logger.warning(f"Failed to publish to message queue: {e}")

        return QueryResponse(
            success=True,
            analysis=analysis,
            processing_time_ms=processing_time,
        )

    except ValueError as e:
        logger.warning(f"Invalid query: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Error analyzing query: {e}", exc_info=True)
        processing_time = (time.time() - start_time) * 1000
        return QueryResponse(
            success=False,
            error=str(e),
            processing_time_ms=processing_time,
        )


@router.get("/analytics")
async def get_analytics(
    date: Optional[str] = Query(None, description="Date in YYYY-MM-DD format (defaults to today)"),
    consumer_manager = Depends(get_consumer_manager),
) -> dict[str, Any]:
    """
    Get query analytics summary for a specific date.

    :param date: Date in YYYY-MM-DD format (defaults to today).
    :param consumer_manager: Injected ConsumerManager instance (optional).
    :return: Analytics summary dictionary.
    :raises HTTPException: If analytics service is not available or retrieval fails.
    """
    try:
        # Check if consumer manager is available
        if not consumer_manager:
            raise HTTPException(
                status_code=503,
                detail="Analytics service not available (Redis required)"
            )

        # Get analytics summary
        summary = await consumer_manager.get_analytics_summary(date)

        return {
            "success": True,
            "analytics": summary
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving analytics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve analytics: {str(e)}")


@router.get("/history/{user_id}")
async def get_user_history(
    user_id: str,
    limit: int = Query(10, ge=1, le=100, description="Number of queries to return"),
    redis_client = Depends(get_redis_client),
) -> dict[str, Any]:
    """
    Get query history for a specific user.

    :param user_id: User ID to get history for.
    :param limit: Number of queries to return (1-100, default 10).
    :param redis_client: Injected Redis client (optional).
    :return: User history dictionary.
    :raises HTTPException: If history service is not available or retrieval fails.
    """
    try:
        # Check if Redis is available
        if not redis_client:
            raise HTTPException(
                status_code=503,
                detail="History service not available (Redis required)"
            )

        # Get user's query history
        user_history_key = f"query:history:user:{user_id}"
        query_ids = await redis_client.lrange(user_history_key, 0, limit - 1)

        # Fetch details for each query
        history_items = []
        for query_id_bytes in query_ids:
            query_id = query_id_bytes.decode() if isinstance(query_id_bytes, bytes) else query_id_bytes
            history_key = f"query:history:{user_id}:{query_id}"

            # Get query data
            query_data = await redis_client.hgetall(history_key)
            if query_data:
                # Decode bytes to strings
                decoded_data = {
                    k.decode() if isinstance(k, bytes) else k: v.decode() if isinstance(v, bytes) else v
                    for k, v in query_data.items()
                }
                history_items.append(decoded_data)

        return {
            "success": True,
            "user_id": user_id,
            "count": len(history_items),
            "history": history_items
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving user history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve history: {str(e)}")


@router.post("/expand")
async def expand_query(
    query_request: QueryRequest,
    analyzer_service: QueryAnalyzerService = Depends(get_analyzer_service),
) -> dict[str, Any]:
    """
    Expand a query into multiple variations for better retrieval.

    :param query_request: Query request with query text.
    :param analyzer_service: Injected QueryAnalyzerService instance.
    :return: Query expansion result dictionary.
    :raises HTTPException: If query expansion fails.
    """
    try:
        logger.info(f"Received query expansion request for: {query_request.query[:100]}")

        # Expand query
        expansion_result = await analyzer_service.expand_query(
            query=query_request.query,
            max_expansions=settings.QUERY_EXPANSION_COUNT
        )

        return {
            "success": True,
            "expansion": expansion_result
        }

    except Exception as e:
        logger.error(f"Error expanding query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to expand query: {str(e)}")


@router.get("/cache/stats")
async def get_cache_stats(
    cache_service: CacheService = Depends(get_cache_service),
) -> dict[str, Any]:
    """
    Get cache statistics.

    :param cache_service: Injected CacheService instance (optional).
    :return: Cache statistics dictionary.
    :raises HTTPException: If cache service is not available or retrieval fails.
    """
    try:
        if not cache_service:
            raise HTTPException(
                status_code=503,
                detail="Cache service not available (Redis required)"
            )

        stats = await cache_service.get_cache_stats()

        return {
            "success": True,
            "stats": stats
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving cache stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve cache stats: {str(e)}")


@router.delete("/cache/invalidate")
async def invalidate_cache(
    query: str = Query(..., description="Query to invalidate from cache"),
    cache_service: CacheService = Depends(get_cache_service),
) -> dict[str, Any]:
    """
    Invalidate cache for a specific query.

    :param query: Query string to invalidate from cache.
    :param cache_service: Injected CacheService instance (optional).
    :return: Invalidation result dictionary.
    :raises HTTPException: If cache service is not available or invalidation fails.
    """
    try:
        if not cache_service:
            raise HTTPException(
                status_code=503,
                detail="Cache service not available (Redis required)"
            )

        success = await cache_service.invalidate_query(query)

        return {
            "success": success,
            "message": f"Cache invalidated for query: {query[:50]}"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error invalidating cache: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to invalidate cache: {str(e)}")


@router.delete("/cache/clear")
async def clear_cache(
    cache_service: CacheService = Depends(get_cache_service),
) -> dict[str, Any]:
    """
    Clear all query analysis cache (use with caution!).

    :param cache_service: Injected CacheService instance (optional).
    :return: Clear result dictionary.
    :raises HTTPException: If cache service is not available or clearing fails.
    """
    try:
        if not cache_service:
            raise HTTPException(
                status_code=503,
                detail="Cache service not available (Redis required)"
            )

        success = await cache_service.clear_all_cache()

        return {
            "success": success,
            "message": "All query analysis cache cleared"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing cache: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")


@router.get("/health")
async def health_check() -> dict[str, Any]:
    """
    Health check endpoint.

    :return: Health status dictionary.
    """
    return {"status": "healthy", "service": "query-analysis"}
