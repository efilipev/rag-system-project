"""
Redis Cache Service for Query Analysis
Caches embeddings and analysis results to improve performance
"""
import json
import hashlib
from typing import Optional, Dict, Any, List
from datetime import datetime

from redis.asyncio import Redis

from src.core.config import settings
from src.core.logging import logger


class CacheService:
    """
    Service for caching query analysis results and embeddings
    """

    def __init__(self, redis_client: Optional[Redis] = None):
        self.redis_client = redis_client
        self.enabled = settings.CACHE_ENABLED and redis_client is not None

        # Cache key prefixes
        self.embedding_prefix = "qa:embedding"
        self.analysis_prefix = "qa:analysis"
        self.expansion_prefix = "qa:expansion"

        # Default TTLs
        self.embedding_ttl = settings.CACHE_TTL  # 1 hour
        self.analysis_ttl = settings.CACHE_TTL  # 1 hour
        self.expansion_ttl = 7200  # 2 hours

    def _generate_cache_key(self, prefix: str, query: str, **kwargs) -> str:
        """
        Generate a cache key based on query and optional parameters

        Args:
            prefix: Cache key prefix
            query: Query text
            **kwargs: Additional parameters to include in key

        Returns:
            Cache key string
        """
        # Create a deterministic hash of the query and parameters
        content = f"{query}:{json.dumps(kwargs, sort_keys=True)}"
        query_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

        return f"{prefix}:{query_hash}"

    async def get_embedding(self, query: str) -> Optional[List[float]]:
        """
        Get cached embedding for a query

        Args:
            query: Query text

        Returns:
            Embedding vector or None if not cached
        """
        if not self.enabled:
            return None

        try:
            cache_key = self._generate_cache_key(self.embedding_prefix, query)
            cached_data = await self.redis_client.get(cache_key)

            if cached_data:
                # Deserialize embedding
                embedding = json.loads(cached_data.decode() if isinstance(cached_data, bytes) else cached_data)
                logger.debug(f"Cache HIT for embedding: {query[:50]}")
                return embedding
            else:
                logger.debug(f"Cache MISS for embedding: {query[:50]}")
                return None

        except Exception as e:
            logger.error(f"Error retrieving embedding from cache: {e}", exc_info=True)
            return None

    async def set_embedding(self, query: str, embedding: List[float]) -> bool:
        """
        Cache an embedding vector

        Args:
            query: Query text
            embedding: Embedding vector

        Returns:
            True if cached successfully
        """
        if not self.enabled:
            return False

        try:
            cache_key = self._generate_cache_key(self.embedding_prefix, query)

            # Serialize embedding
            embedding_json = json.dumps(embedding)

            # Store with TTL
            await self.redis_client.setex(
                cache_key,
                self.embedding_ttl,
                embedding_json
            )

            logger.debug(f"Cached embedding for: {query[:50]}")
            return True

        except Exception as e:
            logger.error(f"Error caching embedding: {e}", exc_info=True)
            return False

    async def get_analysis(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Get cached query analysis result

        Args:
            query: Query text

        Returns:
            Analysis result dictionary or None if not cached
        """
        if not self.enabled:
            return None

        try:
            cache_key = self._generate_cache_key(self.analysis_prefix, query)
            cached_data = await self.redis_client.get(cache_key)

            if cached_data:
                # Deserialize analysis
                analysis = json.loads(cached_data.decode() if isinstance(cached_data, bytes) else cached_data)
                logger.debug(f"Cache HIT for analysis: {query[:50]}")
                return analysis
            else:
                logger.debug(f"Cache MISS for analysis: {query[:50]}")
                return None

        except Exception as e:
            logger.error(f"Error retrieving analysis from cache: {e}", exc_info=True)
            return None

    async def set_analysis(self, query: str, analysis: Dict[str, Any]) -> bool:
        """
        Cache a query analysis result

        Args:
            query: Query text
            analysis: Analysis result dictionary

        Returns:
            True if cached successfully
        """
        if not self.enabled:
            return False

        try:
            cache_key = self._generate_cache_key(self.analysis_prefix, query)

            # Add cache metadata
            cache_data = {
                "analysis": analysis,
                "cached_at": datetime.utcnow().isoformat(),
            }

            # Serialize analysis
            analysis_json = json.dumps(cache_data)

            # Store with TTL
            await self.redis_client.setex(
                cache_key,
                self.analysis_ttl,
                analysis_json
            )

            logger.debug(f"Cached analysis for: {query[:50]}")
            return True

        except Exception as e:
            logger.error(f"Error caching analysis: {e}", exc_info=True)
            return False

    async def get_expansion(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Get cached query expansion result

        Args:
            query: Query text

        Returns:
            Expansion result dictionary or None if not cached
        """
        if not self.enabled:
            return None

        try:
            cache_key = self._generate_cache_key(self.expansion_prefix, query)
            cached_data = await self.redis_client.get(cache_key)

            if cached_data:
                expansion = json.loads(cached_data.decode() if isinstance(cached_data, bytes) else cached_data)
                logger.debug(f"Cache HIT for expansion: {query[:50]}")
                return expansion
            else:
                logger.debug(f"Cache MISS for expansion: {query[:50]}")
                return None

        except Exception as e:
            logger.error(f"Error retrieving expansion from cache: {e}", exc_info=True)
            return None

    async def set_expansion(self, query: str, expansion: Dict[str, Any]) -> bool:
        """
        Cache a query expansion result

        Args:
            query: Query text
            expansion: Expansion result dictionary

        Returns:
            True if cached successfully
        """
        if not self.enabled:
            return False

        try:
            cache_key = self._generate_cache_key(self.expansion_prefix, query)

            # Add cache metadata
            cache_data = {
                "expansion": expansion,
                "cached_at": datetime.utcnow().isoformat(),
            }

            # Serialize expansion
            expansion_json = json.dumps(cache_data)

            # Store with TTL
            await self.redis_client.setex(
                cache_key,
                self.expansion_ttl,
                expansion_json
            )

            logger.debug(f"Cached expansion for: {query[:50]}")
            return True

        except Exception as e:
            logger.error(f"Error caching expansion: {e}", exc_info=True)
            return False

    async def invalidate_query(self, query: str) -> bool:
        """
        Invalidate all cached data for a specific query

        Args:
            query: Query text

        Returns:
            True if invalidated successfully
        """
        if not self.enabled:
            return False

        try:
            keys_to_delete = [
                self._generate_cache_key(self.embedding_prefix, query),
                self._generate_cache_key(self.analysis_prefix, query),
                self._generate_cache_key(self.expansion_prefix, query),
            ]

            deleted_count = await self.redis_client.delete(*keys_to_delete)

            logger.info(f"Invalidated {deleted_count} cache entries for query: {query[:50]}")
            return deleted_count > 0

        except Exception as e:
            logger.error(f"Error invalidating cache: {e}", exc_info=True)
            return False

    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics

        Returns:
            Dictionary with cache statistics
        """
        if not self.enabled:
            return {
                "enabled": False,
                "message": "Cache is disabled"
            }

        try:
            # Get info from Redis
            info = await self.redis_client.info("stats")

            # Count keys for each prefix
            embedding_count = 0
            analysis_count = 0
            expansion_count = 0

            # Use scan to count keys (non-blocking)
            cursor = 0
            async for key in self.redis_client.scan_iter(match=f"{self.embedding_prefix}:*", count=100):
                embedding_count += 1

            async for key in self.redis_client.scan_iter(match=f"{self.analysis_prefix}:*", count=100):
                analysis_count += 1

            async for key in self.redis_client.scan_iter(match=f"{self.expansion_prefix}:*", count=100):
                expansion_count += 1

            return {
                "enabled": True,
                "embedding_cache_count": embedding_count,
                "analysis_cache_count": analysis_count,
                "expansion_cache_count": expansion_count,
                "total_cached_items": embedding_count + analysis_count + expansion_count,
                "redis_keyspace_hits": info.get("keyspace_hits", 0),
                "redis_keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": (
                    info.get("keyspace_hits", 0) /
                    (info.get("keyspace_hits", 0) + info.get("keyspace_misses", 1))
                    if info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0) > 0
                    else 0
                ),
            }

        except Exception as e:
            logger.error(f"Error getting cache stats: {e}", exc_info=True)
            return {
                "enabled": True,
                "error": str(e)
            }

    async def clear_all_cache(self) -> bool:
        """
        Clear all query analysis cache (use with caution!)

        Returns:
            True if cleared successfully
        """
        if not self.enabled:
            return False

        try:
            deleted_count = 0

            # Delete all keys with our prefixes
            for prefix in [self.embedding_prefix, self.analysis_prefix, self.expansion_prefix]:
                async for key in self.redis_client.scan_iter(match=f"{prefix}:*", count=100):
                    await self.redis_client.delete(key)
                    deleted_count += 1

            logger.warning(f"Cleared {deleted_count} cache entries (all query analysis cache)")
            return True

        except Exception as e:
            logger.error(f"Error clearing cache: {e}", exc_info=True)
            return False


# Global instance
_cache_service: Optional[CacheService] = None


def get_cache_service(redis_client: Optional[Redis] = None) -> CacheService:
    """Get or create the cache service singleton"""
    global _cache_service

    if _cache_service is None:
        _cache_service = CacheService(redis_client=redis_client)

    return _cache_service
