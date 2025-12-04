"""
Unit tests for RabbitMQ Consumer Services
"""
import pytest
import json
from datetime import datetime
from unittest.mock import AsyncMock, Mock, MagicMock
from app.services.consumers import QueryHistoryConsumer, QueryAnalyticsConsumer, ConsumerManager


class TestQueryHistoryConsumer:
    """Test suite for QueryHistoryConsumer"""

    @pytest_asyncio.fixture
    async def consumer(self, mock_redis):
        """Create QueryHistoryConsumer instance"""
        return QueryHistoryConsumer(mock_redis)

    @pytest.mark.asyncio
    async def test_initialization(self, consumer, mock_redis):
        """Test that QueryHistoryConsumer initializes correctly"""
        assert consumer is not None
        assert consumer.redis_client == mock_redis
        assert consumer.history_key_prefix == "query:history"

    @pytest.mark.asyncio
    async def test_process_message_success(self, consumer, mock_redis, mock_rabbitmq_message):
        """Test successful message processing"""
        message_data = {
            "query_id": "test-123",
            "user_id": "user1",
            "session_id": "session-abc",
            "timestamp": datetime.utcnow().isoformat(),
            "analysis": {
                "original_query": "What is Python?",
                "normalized_query": "what is python?",
                "keywords": ["python"],
                "entities": [],
                "intent": {"intent": "definitional", "confidence": 0.9}
            },
            "processing_time_ms": 123.45
        }

        mock_rabbitmq_message.body = json.dumps(message_data).encode()

        await consumer.process_message(mock_rabbitmq_message)

        # Verify Redis hset was called to store history
        assert mock_redis.hset.called
        # Verify TTL was set
        assert mock_redis.expire.called
        # Verify lpush was called to add to user history list
        assert mock_redis.lpush.called

    @pytest.mark.asyncio
    async def test_process_message_missing_optional_fields(self, consumer, mock_redis, mock_rabbitmq_message):
        """Test processing message with missing optional fields"""
        message_data = {
            "query_id": "test-123",
            # user_id missing - should default to "anonymous"
            # session_id missing - should be converted to empty string
            # timestamp missing - should use current time
            "analysis": {
                "original_query": "Test query",
                "normalized_query": "test query"
            }
            # processing_time_ms missing - should default to 0
        }

        mock_rabbitmq_message.body = json.dumps(message_data).encode()

        await consumer.process_message(mock_rabbitmq_message)

        # Should process without errors
        assert mock_redis.hset.called

        # Check that default values were used
        call_args = mock_redis.hset.call_args
        history_data = call_args[1]['mapping']
        assert history_data['user_id'] == 'anonymous'
        assert history_data['session_id'] == ''
        assert history_data['processing_time_ms'] == '0'

    @pytest.mark.asyncio
    async def test_process_message_with_none_session_id(self, consumer, mock_redis, mock_rabbitmq_message):
        """Test processing message with None session_id"""
        message_data = {
            "query_id": "test-123",
            "user_id": "user1",
            "session_id": None,  # Explicitly None
            "analysis": {
                "original_query": "Test query"
            }
        }

        mock_rabbitmq_message.body = json.dumps(message_data).encode()

        await consumer.process_message(mock_rabbitmq_message)

        # Should convert None to empty string
        call_args = mock_redis.hset.call_args
        history_data = call_args[1]['mapping']
        assert history_data['session_id'] == ''

    @pytest.mark.asyncio
    async def test_process_message_error_handling(self, consumer, mock_redis, mock_rabbitmq_message):
        """Test error handling in message processing"""
        # Make Redis raise an error
        mock_redis.hset.side_effect = Exception("Redis connection error")

        message_data = {
            "query_id": "test-123",
            "analysis": {}
        }
        mock_rabbitmq_message.body = json.dumps(message_data).encode()

        # Should not raise exception (should log error instead)
        await consumer.process_message(mock_rabbitmq_message)

    @pytest.mark.asyncio
    async def test_store_complex_analysis(self, consumer, mock_redis, mock_rabbitmq_message):
        """Test storing complex analysis with entities and intent"""
        message_data = {
            "query_id": "test-123",
            "user_id": "user1",
            "analysis": {
                "original_query": "Apple released iPhone 15 in California",
                "normalized_query": "apple released iphone 15 in california",
                "keywords": ["apple", "iphone", "15", "california"],
                "entities": [
                    {"text": "Apple", "label": "ORG", "start": 0, "end": 5},
                    {"text": "iPhone 15", "label": "PRODUCT", "start": 14, "end": 23},
                    {"text": "California", "label": "GPE", "start": 27, "end": 37}
                ],
                "intent": {
                    "intent": "informational",
                    "confidence": 0.87
                }
            },
            "processing_time_ms": 234.56
        }

        mock_rabbitmq_message.body = json.dumps(message_data).encode()

        await consumer.process_message(mock_rabbitmq_message)

        assert mock_redis.hset.called
        call_args = mock_redis.hset.call_args
        history_data = call_args[1]['mapping']

        # Verify complex data is JSON-serialized
        assert isinstance(history_data['keywords'], str)
        assert isinstance(history_data['entities'], str)
        assert isinstance(history_data['intent'], str)

        # Verify data can be deserialized
        keywords = json.loads(history_data['keywords'])
        assert len(keywords) == 4

    @pytest.mark.asyncio
    async def test_user_history_list_management(self, consumer, mock_redis, mock_rabbitmq_message):
        """Test that user history list is properly managed"""
        message_data = {
            "query_id": "test-123",
            "user_id": "user1",
            "analysis": {}
        }

        mock_rabbitmq_message.body = json.dumps(message_data).encode()

        await consumer.process_message(mock_rabbitmq_message)

        # Verify lpush was called to add query_id
        assert mock_redis.lpush.called
        lpush_args = mock_redis.lpush.call_args
        assert "query:history:user:user1" in lpush_args[0][0]
        assert lpush_args[0][1] == "test-123"

        # Verify ltrim was called to keep last 1000 queries
        assert mock_redis.ltrim.called
        ltrim_args = mock_redis.ltrim.call_args
        assert ltrim_args[0][1] == 0
        assert ltrim_args[0][2] == 999

    @pytest.mark.asyncio
    async def test_ttl_settings(self, consumer, mock_redis, mock_rabbitmq_message):
        """Test that TTL is set correctly (30 days)"""
        message_data = {
            "query_id": "test-123",
            "user_id": "user1",
            "analysis": {}
        }

        mock_rabbitmq_message.body = json.dumps(message_data).encode()

        await consumer.process_message(mock_rabbitmq_message)

        # Verify expire was called with 30 days TTL (2592000 seconds)
        assert mock_redis.expire.call_count == 2  # Once for hash, once for list
        expire_calls = mock_redis.expire.call_args_list
        assert any(call[0][1] == 2592000 for call in expire_calls)


class TestQueryAnalyticsConsumer:
    """Test suite for QueryAnalyticsConsumer"""

    @pytest_asyncio.fixture
    async def consumer(self, mock_redis):
        """Create QueryAnalyticsConsumer instance"""
        return QueryAnalyticsConsumer(mock_redis)

    @pytest.mark.asyncio
    async def test_initialization(self, consumer, mock_redis):
        """Test that QueryAnalyticsConsumer initializes correctly"""
        assert consumer is not None
        assert consumer.redis_client == mock_redis
        assert consumer.analytics_key_prefix == "query:analytics"

    @pytest.mark.asyncio
    async def test_process_message_success(self, consumer, mock_redis, mock_rabbitmq_message):
        """Test successful analytics message processing"""
        message_data = {
            "query_id": "test-123",
            "user_id": "user1",
            "analysis": {
                "normalized_query": "what is python?",
                "intent": {"intent": "definitional", "confidence": 0.9},
                "entities": [
                    {"text": "Python", "label": "LANGUAGE"}
                ],
                "keywords": ["python", "programming"],
                "language": "en"
            },
            "processing_time_ms": 123.45
        }

        mock_rabbitmq_message.body = json.dumps(message_data).encode()

        await consumer.process_message(mock_rabbitmq_message)

        # Verify various Redis operations were called
        assert mock_redis.incr.called  # For counters
        assert mock_redis.zincrby.called  # For sorted sets

    @pytest.mark.asyncio
    async def test_track_query_count(self, consumer, mock_redis, mock_rabbitmq_message):
        """Test that total query count is tracked"""
        message_data = {
            "analysis": {}
        }

        mock_rabbitmq_message.body = json.dumps(message_data).encode()

        await consumer.process_message(mock_rabbitmq_message)

        # Verify incr was called for total queries
        assert mock_redis.incr.called
        incr_calls = [call[0][0] for call in mock_redis.incr.call_args_list]
        today = datetime.utcnow().strftime("%Y-%m-%d")
        assert any(f"queries:total:{today}" in call for call in incr_calls)

    @pytest.mark.asyncio
    async def test_track_popular_queries(self, consumer, mock_redis, mock_rabbitmq_message):
        """Test that popular queries are tracked"""
        message_data = {
            "analysis": {
                "normalized_query": "what is python?"
            }
        }

        mock_rabbitmq_message.body = json.dumps(message_data).encode()

        await consumer.process_message(mock_rabbitmq_message)

        # Verify zincrby was called for popular queries
        assert mock_redis.zincrby.called

    @pytest.mark.asyncio
    async def test_track_intent_distribution(self, consumer, mock_redis, mock_rabbitmq_message):
        """Test that intent distribution is tracked"""
        message_data = {
            "analysis": {
                "intent": {
                    "intent": "procedural",
                    "confidence": 0.85
                }
            }
        }

        mock_rabbitmq_message.body = json.dumps(message_data).encode()

        await consumer.process_message(mock_rabbitmq_message)

        # Verify intent counter was incremented
        incr_calls = [call[0][0] for call in mock_redis.incr.call_args_list]
        today = datetime.utcnow().strftime("%Y-%m-%d")
        assert any(f"intent:procedural:{today}" in call for call in incr_calls)

    @pytest.mark.asyncio
    async def test_track_entity_types(self, consumer, mock_redis, mock_rabbitmq_message):
        """Test that entity types are tracked"""
        message_data = {
            "analysis": {
                "entities": [
                    {"text": "Apple", "label": "ORG"},
                    {"text": "California", "label": "GPE"}
                ]
            }
        }

        mock_rabbitmq_message.body = json.dumps(message_data).encode()

        await consumer.process_message(mock_rabbitmq_message)

        # Verify entity counters were incremented
        incr_calls = [call[0][0] for call in mock_redis.incr.call_args_list]
        today = datetime.utcnow().strftime("%Y-%m-%d")
        assert any(f"entity:ORG:{today}" in call for call in incr_calls)
        assert any(f"entity:GPE:{today}" in call for call in incr_calls)

    @pytest.mark.asyncio
    async def test_track_keywords(self, consumer, mock_redis, mock_rabbitmq_message):
        """Test that keywords are tracked"""
        message_data = {
            "analysis": {
                "keywords": ["python", "programming", "language"]
            }
        }

        mock_rabbitmq_message.body = json.dumps(message_data).encode()

        await consumer.process_message(mock_rabbitmq_message)

        # Verify keywords were added to sorted set
        zincrby_calls = [call[0][1] for call in mock_redis.zincrby.call_args_list]
        assert "python" in zincrby_calls or any("python" in str(call) for call in mock_redis.zincrby.call_args_list)

    @pytest.mark.asyncio
    async def test_track_processing_time(self, consumer, mock_redis, mock_rabbitmq_message):
        """Test that processing time is tracked"""
        message_data = {
            "processing_time_ms": 234.56
        }

        mock_rabbitmq_message.body = json.dumps(message_data).encode()

        await consumer.process_message(mock_rabbitmq_message)

        # Verify zadd was called for processing time
        assert mock_redis.zadd.called

    @pytest.mark.asyncio
    async def test_track_user_activity(self, consumer, mock_redis, mock_rabbitmq_message):
        """Test that user activity is tracked"""
        message_data = {
            "user_id": "user123",
            "analysis": {}
        }

        mock_rabbitmq_message.body = json.dumps(message_data).encode()

        await consumer.process_message(mock_rabbitmq_message)

        # Verify user was added to active users
        zincrby_calls = [call[0][1] for call in mock_redis.zincrby.call_args_list]
        assert "user123" in zincrby_calls or any("user123" in str(call) for call in mock_redis.zincrby.call_args_list)

    @pytest.mark.asyncio
    async def test_get_analytics_summary(self, consumer, mock_redis):
        """Test getting analytics summary"""
        date = "2025-11-12"

        # Mock Redis responses
        mock_redis.get.return_value = b"100"  # total queries
        mock_redis.zrevrange.return_value = [
            (b"what is python?", 50.0),
            (b"how to use asyncio?", 30.0)
        ]
        mock_redis.zcard.return_value = 25

        summary = await consumer.get_analytics_summary(date)

        assert summary is not None
        assert summary['date'] == date
        assert summary['total_queries'] == 100
        assert len(summary['popular_queries']) == 2
        assert summary['active_users'] == 25

    @pytest.mark.asyncio
    async def test_get_analytics_summary_error_handling(self, consumer, mock_redis):
        """Test error handling in get_analytics_summary"""
        # Make Redis raise an error
        mock_redis.get.side_effect = Exception("Redis error")

        date = "2025-11-12"
        summary = await consumer.get_analytics_summary(date)

        # Should return error in response
        assert 'error' in summary
        assert summary['date'] == date

    @pytest.mark.asyncio
    async def test_ttl_settings_analytics(self, consumer, mock_redis, mock_rabbitmq_message):
        """Test that TTL is set correctly for analytics (90 days)"""
        message_data = {
            "analysis": {
                "normalized_query": "test query"
            }
        }

        mock_rabbitmq_message.body = json.dumps(message_data).encode()

        await consumer.process_message(mock_rabbitmq_message)

        # Verify expire was called with 90 days TTL (7776000 seconds)
        assert mock_redis.expire.called
        expire_calls = [call[0][1] for call in mock_redis.expire.call_args_list]
        assert 7776000 in expire_calls


class TestConsumerManager:
    """Test suite for ConsumerManager"""

    @pytest_asyncio.fixture
    async def manager(self, mock_redis):
        """Create ConsumerManager instance"""
        mock_mq_service = AsyncMock()
        return ConsumerManager(mock_mq_service, mock_redis)

    @pytest.mark.asyncio
    async def test_initialization(self, manager, mock_redis):
        """Test that ConsumerManager initializes correctly"""
        assert manager is not None
        assert manager.history_consumer is not None
        assert manager.analytics_consumer is not None

    @pytest.mark.asyncio
    async def test_start_consumers(self, manager):
        """Test starting all consumer services"""
        await manager.start_consumers()

        # Verify consume_messages was called for both consumers
        assert manager.mq_service.consume_messages.call_count == 2

    @pytest.mark.asyncio
    async def test_get_analytics_summary_delegates(self, manager):
        """Test that get_analytics_summary delegates to analytics consumer"""
        manager.analytics_consumer.get_analytics_summary = AsyncMock(return_value={"date": "2025-11-12"})

        result = await manager.get_analytics_summary("2025-11-12")

        assert result['date'] == "2025-11-12"
        manager.analytics_consumer.get_analytics_summary.assert_called_once_with("2025-11-12")

    @pytest.mark.asyncio
    async def test_get_analytics_summary_defaults_to_today(self, manager):
        """Test that get_analytics_summary defaults to today if no date provided"""
        manager.analytics_consumer.get_analytics_summary = AsyncMock(return_value={})

        await manager.get_analytics_summary(None)

        # Should be called with today's date
        today = datetime.utcnow().strftime("%Y-%m-%d")
        manager.analytics_consumer.get_analytics_summary.assert_called_once_with(today)
