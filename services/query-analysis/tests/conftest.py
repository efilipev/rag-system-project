"""
Pytest configuration and shared fixtures for Query Analysis Service tests
"""
import pytest
import pytest_asyncio
from unittest.mock import Mock, AsyncMock, MagicMock
from redis.asyncio import Redis
import aio_pika


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing"""
    redis = AsyncMock(spec=Redis)
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock(return_value=True)
    redis.setex = AsyncMock(return_value=True)
    redis.hset = AsyncMock(return_value=1)
    redis.hgetall = AsyncMock(return_value={})
    redis.lpush = AsyncMock(return_value=1)
    redis.ltrim = AsyncMock(return_value=True)
    redis.lrange = AsyncMock(return_value=[])
    redis.expire = AsyncMock(return_value=True)
    redis.incr = AsyncMock(return_value=1)
    redis.zincrby = AsyncMock(return_value=1.0)
    redis.zadd = AsyncMock(return_value=1)
    redis.zrevrange = AsyncMock(return_value=[])
    redis.zcard = AsyncMock(return_value=0)
    redis.scan_iter = AsyncMock(return_value=[])
    redis.info = AsyncMock(return_value={
        'keyspace_hits': 10,
        'keyspace_misses': 5,
    })
    return redis


@pytest.fixture
def mock_rabbitmq_connection():
    """Mock RabbitMQ connection for testing"""
    connection = AsyncMock(spec=aio_pika.Connection)
    channel = AsyncMock(spec=aio_pika.Channel)
    exchange = AsyncMock(spec=aio_pika.Exchange)
    queue = AsyncMock(spec=aio_pika.Queue)

    connection.channel = AsyncMock(return_value=channel)
    channel.declare_exchange = AsyncMock(return_value=exchange)
    channel.declare_queue = AsyncMock(return_value=queue)
    queue.bind = AsyncMock()
    queue.consume = AsyncMock()

    return connection


@pytest.fixture
def mock_rabbitmq_message():
    """Mock RabbitMQ message for testing"""
    message = Mock(spec=aio_pika.IncomingMessage)
    message.body = b'{"query_id": "test-123", "user_id": "user1", "analysis": {}}'
    message.correlation_id = "test-123"
    message.process = MagicMock()
    message.process.return_value.__aenter__ = AsyncMock()
    message.process.return_value.__aexit__ = AsyncMock()
    return message


@pytest.fixture
def sample_query():
    """Sample query for testing"""
    return "How do I implement OAuth2 authentication in Python?"


@pytest.fixture
def sample_analysis_result():
    """Sample analysis result for testing"""
    return {
        "original_query": "How do I implement OAuth2 authentication in Python?",
        "normalized_query": "how do i implement oauth2 authentication in python?",
        "keywords": ["implement", "oauth2", "authentication", "python"],
        "entities": [
            {"text": "OAuth2", "label": "PRODUCT", "start": 19, "end": 25},
            {"text": "Python", "label": "PRODUCT", "start": 44, "end": 50}
        ],
        "intent": {
            "intent": "procedural",
            "confidence": 0.85
        },
        "language": "en",
        "embedding": [0.1] * 384,
        "query_type": "how_to"
    }


@pytest.fixture
def sample_expansion_result():
    """Sample query expansion result for testing"""
    return {
        "original_query": "What is machine learning?",
        "all_expansions": [
            "what is political machine learning?",
            "what is car learning?",
            "machine learning? definition"
        ],
        "expansion_methods": ["synonyms", "reformulation"],
        "total_expansions": 3
    }


@pytest_asyncio.fixture
async def mock_cache_service(mock_redis):
    """Mock cache service for testing"""
    from app.services.cache_service import CacheService
    cache_service = CacheService(mock_redis)
    return cache_service
