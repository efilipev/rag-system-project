"""
RabbitMQ Consumer Services for Query Analysis Events
"""
import json
from datetime import datetime
from typing import Dict, Any, List
from collections import defaultdict

import aio_pika
from redis.asyncio import Redis

from src.core.config import settings
from src.core.logging import logger


class QueryHistoryConsumer:
    """
    Consumer for logging query analysis history
    Stores query analysis results for audit and debugging
    """

    def __init__(self, redis_client: Redis):
        self.redis_client = redis_client
        self.history_key_prefix = "query:history"

    async def process_message(self, message: aio_pika.IncomingMessage) -> None:
        """
        Process incoming query.analyzed messages and store in Redis
        """
        async with message.process():
            try:
                # Parse message body
                body = json.loads(message.body.decode())

                # Extract query data
                query_id = body.get("query_id", message.correlation_id)
                timestamp = body.get("timestamp", datetime.utcnow().isoformat())
                user_id = body.get("user_id", "anonymous")
                session_id = body.get("session_id")
                analysis = body.get("analysis", {})

                # Store in Redis with TTL (30 days)
                history_key = f"{self.history_key_prefix}:{user_id}:{query_id}"
                history_data = {
                    "query_id": query_id,
                    "user_id": user_id,
                    "session_id": session_id or "",  # Convert None to empty string
                    "timestamp": timestamp,
                    "original_query": analysis.get("original_query", ""),
                    "normalized_query": analysis.get("normalized_query", ""),
                    "keywords": json.dumps(analysis.get("keywords", [])),
                    "entities": json.dumps(analysis.get("entities", [])),
                    "intent": json.dumps(analysis.get("intent", {})),
                    "processing_time_ms": str(body.get("processing_time_ms", 0)),  # Convert to string
                }

                # Store as hash
                await self.redis_client.hset(history_key, mapping=history_data)
                await self.redis_client.expire(history_key, 2592000)  # 30 days

                # Add to user's query history list (most recent first)
                user_history_key = f"{self.history_key_prefix}:user:{user_id}"
                await self.redis_client.lpush(user_history_key, query_id)
                await self.redis_client.ltrim(user_history_key, 0, 999)  # Keep last 1000
                await self.redis_client.expire(user_history_key, 2592000)

                logger.info(
                    f"Stored query history for user {user_id}, query {query_id}",
                    extra={"query_id": query_id, "user_id": user_id}
                )

            except Exception as e:
                logger.error(f"Error processing query history message: {e}", exc_info=True)
                # Message will be requeued due to failed processing


class QueryAnalyticsConsumer:
    """
    Consumer for aggregating query analytics
    Tracks query patterns, popular searches, and performance metrics
    """

    def __init__(self, redis_client: Redis):
        self.redis_client = redis_client
        self.analytics_key_prefix = "query:analytics"

        # In-memory buffers for batching (would use proper analytics DB in production)
        self.query_counts: Dict[str, int] = defaultdict(int)
        self.intent_counts: Dict[str, int] = defaultdict(int)
        self.entity_counts: Dict[str, int] = defaultdict(int)
        self.keyword_counts: Dict[str, int] = defaultdict(int)
        self.processing_times: List[float] = []

    async def process_message(self, message: aio_pika.IncomingMessage) -> None:
        """
        Process incoming query.analyzed messages and aggregate analytics
        """
        async with message.process():
            try:
                # Parse message body
                body = json.loads(message.body.decode())
                analysis = body.get("analysis", {})

                # Get current date for daily aggregation
                today = datetime.utcnow().strftime("%Y-%m-%d")

                # Increment query count
                await self._increment_counter(f"queries:total:{today}")

                # Track normalized query popularity
                normalized_query = analysis.get("normalized_query", "").lower()
                if normalized_query:
                    await self._increment_sorted_set(
                        f"queries:popular:{today}",
                        normalized_query
                    )

                # Track intent distribution
                intent_data = analysis.get("intent", {})
                if intent_data and isinstance(intent_data, dict):
                    intent = intent_data.get("intent")
                    if intent:
                        await self._increment_counter(f"intent:{intent}:{today}")
                        await self._increment_sorted_set(f"intent:popular:{today}", intent)

                # Track entity types
                entities = analysis.get("entities", [])
                if entities and isinstance(entities, list):
                    for entity in entities:
                        if isinstance(entity, dict):
                            entity_label = entity.get("label")
                            if entity_label:
                                await self._increment_counter(f"entity:{entity_label}:{today}")

                # Track keywords
                keywords = analysis.get("keywords", [])
                if keywords and isinstance(keywords, list):
                    for keyword in keywords[:10]:  # Limit to top 10
                        if keyword:
                            await self._increment_sorted_set(
                                f"keywords:popular:{today}",
                                str(keyword).lower()
                            )

                # Track processing time percentiles
                processing_time = body.get("processing_time_ms", 0)
                if processing_time > 0:
                    await self.redis_client.zadd(
                        f"performance:processing_time:{today}",
                        {str(processing_time): processing_time}
                    )

                # Track language distribution
                language = analysis.get("language", "unknown")
                await self._increment_counter(f"language:{language}:{today}")

                # Track user activity (if user_id provided)
                user_id = body.get("user_id")
                if user_id and user_id != "anonymous":
                    await self._increment_sorted_set(f"users:active:{today}", user_id)

                logger.debug(
                    f"Aggregated analytics for query",
                    extra={"date": today}
                )

            except Exception as e:
                logger.error(f"Error processing analytics message: {e}", exc_info=True)

    async def _increment_counter(self, key: str) -> None:
        """Increment a Redis counter"""
        try:
            await self.redis_client.incr(key)
            await self.redis_client.expire(key, 7776000)  # 90 days
        except Exception as e:
            logger.error(f"Error incrementing counter {key}: {e}")

    async def _increment_sorted_set(self, key: str, member: str) -> None:
        """Increment a member's score in a sorted set"""
        try:
            await self.redis_client.zincrby(key, 1, member)
            await self.redis_client.expire(key, 7776000)  # 90 days
        except Exception as e:
            logger.error(f"Error incrementing sorted set {key}: {e}")

    async def get_analytics_summary(self, date: str) -> Dict[str, Any]:
        """
        Get analytics summary for a specific date
        """
        try:
            # Get total queries
            total_queries = await self.redis_client.get(f"queries:total:{date}")

            # Get popular queries (top 10)
            popular_queries = await self.redis_client.zrevrange(
                f"queries:popular:{date}",
                0, 9,
                withscores=True
            )

            # Get intent distribution (top 5)
            popular_intents = await self.redis_client.zrevrange(
                f"intent:popular:{date}",
                0, 4,
                withscores=True
            )

            # Get popular keywords (top 20)
            popular_keywords = await self.redis_client.zrevrange(
                f"keywords:popular:{date}",
                0, 19,
                withscores=True
            )

            # Get active users count
            active_users = await self.redis_client.zcard(f"users:active:{date}")

            return {
                "date": date,
                "total_queries": int(total_queries) if total_queries else 0,
                "popular_queries": [
                    {"query": q.decode() if isinstance(q, bytes) else q, "count": int(s)}
                    for q, s in popular_queries
                ] if popular_queries else [],
                "intent_distribution": [
                    {"intent": i.decode() if isinstance(i, bytes) else i, "count": int(s)}
                    for i, s in popular_intents
                ] if popular_intents else [],
                "popular_keywords": [
                    {"keyword": k.decode() if isinstance(k, bytes) else k, "count": int(s)}
                    for k, s in popular_keywords
                ] if popular_keywords else [],
                "active_users": active_users or 0,
            }

        except Exception as e:
            logger.error(f"Error getting analytics summary: {e}", exc_info=True)
            return {
                "date": date,
                "error": str(e)
            }


class ConsumerManager:
    """
    Manager for all consumer services
    Handles initialization and lifecycle management
    """

    def __init__(self, message_queue_service, redis_client: Redis):
        self.mq_service = message_queue_service
        self.redis_client = redis_client

        # Initialize consumers
        self.history_consumer = QueryHistoryConsumer(redis_client)
        self.analytics_consumer = QueryAnalyticsConsumer(redis_client)

    async def start_consumers(self) -> None:
        """
        Start all consumer services
        """
        try:
            logger.info("Starting consumer services...")

            # Start query history consumer
            await self.mq_service.consume_messages(
                queue_name=settings.RABBITMQ_HISTORY_QUEUE,
                callback=self.history_consumer.process_message,
                routing_key="query.analyzed"
            )

            # Start query analytics consumer
            await self.mq_service.consume_messages(
                queue_name=settings.RABBITMQ_ANALYTICS_QUEUE,
                callback=self.analytics_consumer.process_message,
                routing_key="query.analyzed"
            )

            logger.info("All consumer services started successfully")

        except Exception as e:
            logger.error(f"Failed to start consumer services: {e}", exc_info=True)
            raise

    async def get_analytics_summary(self, date: str = None) -> Dict[str, Any]:
        """
        Get analytics summary for a date (defaults to today)
        """
        if date is None:
            date = datetime.utcnow().strftime("%Y-%m-%d")

        return await self.analytics_consumer.get_analytics_summary(date)
