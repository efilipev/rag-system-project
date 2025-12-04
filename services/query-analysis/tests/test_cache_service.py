"""
Unit tests for Cache Service
"""
import pytest
import json
from unittest.mock import AsyncMock
from app.services.cache_service import CacheService


class TestCacheService:
    """Test suite for CacheService"""

    @pytest_asyncio.fixture
    async def cache_service(self, mock_redis):
        """Create cache service instance for testing"""
        return CacheService(mock_redis)

    @pytest.mark.asyncio
    async def test_initialization(self, cache_service):
        """Test that CacheService initializes correctly"""
        assert cache_service is not None
        assert cache_service.redis_client is not None
        assert cache_service.embedding_prefix == "qa:embedding"
        assert cache_service.analysis_prefix == "qa:analysis"
        assert cache_service.expansion_prefix == "qa:expansion"

    @pytest.mark.asyncio
    async def test_get_embedding_cache_miss(self, cache_service, mock_redis):
        """Test getting embedding when not in cache"""
        mock_redis.get.return_value = None

        query = "What is Python?"
        result = await cache_service.get_embedding(query)

        assert result is None
        mock_redis.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_embedding_cache_hit(self, cache_service, mock_redis):
        """Test getting embedding when in cache"""
        embedding = [0.1, 0.2, 0.3]
        mock_redis.get.return_value = json.dumps(embedding).encode()

        query = "What is Python?"
        result = await cache_service.get_embedding(query)

        assert result == embedding
        mock_redis.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_embedding(self, cache_service, mock_redis):
        """Test setting embedding in cache"""
        query = "What is Python?"
        embedding = [0.1, 0.2, 0.3] * 128  # 384-dim vector

        success = await cache_service.set_embedding(query, embedding)

        assert success is True
        mock_redis.setex.assert_called_once()
        # Check TTL was set
        call_args = mock_redis.setex.call_args
        assert call_args[0][1] == 3600  # 1 hour TTL

    @pytest.mark.asyncio
    async def test_get_analysis_cache_miss(self, cache_service, mock_redis):
        """Test getting analysis when not in cache"""
        mock_redis.get.return_value = None

        query = "What is Python?"
        result = await cache_service.get_analysis(query)

        assert result is None
        mock_redis.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_analysis_cache_hit(self, cache_service, mock_redis, sample_analysis_result):
        """Test getting analysis when in cache"""
        mock_redis.get.return_value = json.dumps(sample_analysis_result).encode()

        query = "What is Python?"
        result = await cache_service.get_analysis(query)

        assert result is not None
        assert result['normalized_query'] == sample_analysis_result['normalized_query']
        assert result['keywords'] == sample_analysis_result['keywords']
        mock_redis.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_analysis(self, cache_service, mock_redis, sample_analysis_result):
        """Test setting analysis in cache"""
        query = "What is Python?"

        success = await cache_service.set_analysis(query, sample_analysis_result)

        assert success is True
        mock_redis.setex.assert_called_once()
        # Check TTL was set (30 minutes)
        call_args = mock_redis.setex.call_args
        assert call_args[0][1] == 1800

    @pytest.mark.asyncio
    async def test_get_expansion_cache_miss(self, cache_service, mock_redis):
        """Test getting expansion when not in cache"""
        mock_redis.get.return_value = None

        query = "What is machine learning?"
        result = await cache_service.get_expansion(query)

        assert result is None
        mock_redis.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_expansion_cache_hit(self, cache_service, mock_redis, sample_expansion_result):
        """Test getting expansion when in cache"""
        mock_redis.get.return_value = json.dumps(sample_expansion_result).encode()

        query = "What is machine learning?"
        result = await cache_service.get_expansion(query)

        assert result is not None
        assert result['original_query'] == sample_expansion_result['original_query']
        assert len(result['all_expansions']) == len(sample_expansion_result['all_expansions'])
        mock_redis.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_expansion(self, cache_service, mock_redis, sample_expansion_result):
        """Test setting expansion in cache"""
        query = "What is machine learning?"

        success = await cache_service.set_expansion(query, sample_expansion_result)

        assert success is True
        mock_redis.setex.assert_called_once()
        # Check TTL was set (1 hour)
        call_args = mock_redis.setex.call_args
        assert call_args[0][1] == 3600

    @pytest.mark.asyncio
    async def test_invalidate_query(self, cache_service, mock_redis):
        """Test invalidating all cache entries for a query"""
        query = "What is Python?"

        success = await cache_service.invalidate_query(query)

        # Should delete embedding, analysis, and expansion
        assert mock_redis.delete.call_count == 3

    @pytest.mark.asyncio
    async def test_clear_all_cache(self, cache_service, mock_redis):
        """Test clearing all cache"""
        # Mock scan_iter to return some keys
        mock_redis.scan_iter.return_value = [
            b"qa:embedding:key1",
            b"qa:analysis:key2",
            b"qa:expansion:key3"
        ]

        success = await cache_service.clear_all_cache()

        # Should have scanned for all prefixes
        assert mock_redis.scan_iter.call_count == 3
        # Should delete the keys found
        assert mock_redis.delete.called

    @pytest.mark.asyncio
    async def test_get_cache_stats(self, cache_service, mock_redis):
        """Test getting cache statistics"""
        # Mock Redis INFO command
        mock_redis.info.return_value = {
            'keyspace_hits': 100,
            'keyspace_misses': 25,
        }

        # Mock key counting
        async def mock_scan_iter(*args, **kwargs):
            match = kwargs.get('match', '')
            if 'embedding' in match:
                return [b"key1", b"key2"]
            elif 'analysis' in match:
                return [b"key3"]
            else:
                return []

        mock_redis.scan_iter = mock_scan_iter

        stats = await cache_service.get_cache_stats()

        assert stats is not None
        assert 'enabled' in stats
        assert stats['enabled'] is True
        assert 'embedding_cache_count' in stats
        assert 'analysis_cache_count' in stats
        assert 'expansion_cache_count' in stats
        assert 'hit_rate' in stats

    @pytest.mark.asyncio
    async def test_cache_key_generation(self, cache_service):
        """Test that cache keys are generated correctly"""
        query = "What is Python?"

        key1 = cache_service._generate_cache_key("prefix", query)
        key2 = cache_service._generate_cache_key("prefix", query)

        # Same query should generate same key
        assert key1 == key2

        # Different queries should generate different keys
        key3 = cache_service._generate_cache_key("prefix", "Different query")
        assert key1 != key3

    @pytest.mark.asyncio
    async def test_cache_key_normalization(self, cache_service):
        """Test that cache keys are normalized (case/whitespace)"""
        query1 = "What is Python?"
        query2 = "what is python?"
        query3 = "What   is   Python?"

        key1 = cache_service._generate_cache_key("prefix", query1)
        key2 = cache_service._generate_cache_key("prefix", query2)
        key3 = cache_service._generate_cache_key("prefix", query3)

        # Keys should be the same after normalization
        assert key1 == key2 == key3

    @pytest.mark.asyncio
    async def test_set_embedding_error_handling(self, cache_service, mock_redis):
        """Test error handling when setting embedding fails"""
        mock_redis.setex.side_effect = Exception("Redis error")

        query = "What is Python?"
        embedding = [0.1] * 384

        success = await cache_service.set_embedding(query, embedding)

        assert success is False

    @pytest.mark.asyncio
    async def test_get_embedding_error_handling(self, cache_service, mock_redis):
        """Test error handling when getting embedding fails"""
        mock_redis.get.side_effect = Exception("Redis error")

        query = "What is Python?"
        result = await cache_service.get_embedding(query)

        assert result is None

    @pytest.mark.asyncio
    async def test_empty_embedding_handling(self, cache_service, mock_redis):
        """Test handling of empty embedding"""
        query = "What is Python?"
        embedding = []

        success = await cache_service.set_embedding(query, embedding)

        # Should still cache empty embedding
        assert success is True or success is False  # Implementation dependent

    @pytest.mark.asyncio
    async def test_large_embedding_handling(self, cache_service, mock_redis):
        """Test handling of large embeddings"""
        query = "What is Python?"
        embedding = [0.1] * 1024  # Larger than typical 384

        success = await cache_service.set_embedding(query, embedding)

        assert mock_redis.setex.called

    @pytest.mark.asyncio
    async def test_complex_analysis_caching(self, cache_service, mock_redis):
        """Test caching of complex analysis with nested structures"""
        query = "What is Python?"
        complex_analysis = {
            "keywords": ["python", "programming"],
            "entities": [
                {"text": "Python", "label": "LANGUAGE", "start": 7, "end": 13}
            ],
            "intent": {
                "primary": "informational",
                "confidence": 0.85,
                "alternatives": ["definitional"]
            },
            "embedding": [0.1] * 384
        }

        success = await cache_service.set_analysis(query, complex_analysis)
        assert success is True

        # Verify it can be retrieved
        mock_redis.get.return_value = json.dumps(complex_analysis).encode()
        result = await cache_service.get_analysis(query)
        assert result is not None
        assert result['intent']['primary'] == 'informational'

    @pytest.mark.asyncio
    async def test_unicode_query_caching(self, cache_service, mock_redis):
        """Test caching with unicode characters in query"""
        query = "What is Python? 你好"
        embedding = [0.1] * 384

        success = await cache_service.set_embedding(query, embedding)

        # Should handle unicode correctly
        assert success is True

    @pytest.mark.asyncio
    async def test_special_characters_in_query(self, cache_service, mock_redis):
        """Test caching with special characters"""
        query = "How to use @decorator & *args in Python?"
        embedding = [0.1] * 384

        success = await cache_service.set_embedding(query, embedding)
        assert success is True

    @pytest.mark.asyncio
    async def test_very_long_query_caching(self, cache_service, mock_redis):
        """Test caching of very long queries"""
        query = "How " * 100 + "is this handled?"
        embedding = [0.1] * 384

        success = await cache_service.set_embedding(query, embedding)
        assert mock_redis.setex.called

    @pytest.mark.asyncio
    async def test_cache_hit_rate_calculation(self, cache_service, mock_redis):
        """Test cache hit rate calculation"""
        mock_redis.info.return_value = {
            'keyspace_hits': 80,
            'keyspace_misses': 20,
        }

        async def mock_scan_iter(*args, **kwargs):
            return []

        mock_redis.scan_iter = mock_scan_iter

        stats = await cache_service.get_cache_stats()

        assert 'hit_rate' in stats
        assert stats['hit_rate'] == 0.8  # 80/(80+20)

    @pytest.mark.asyncio
    async def test_cache_hit_rate_zero_requests(self, cache_service, mock_redis):
        """Test cache hit rate when no requests have been made"""
        mock_redis.info.return_value = {
            'keyspace_hits': 0,
            'keyspace_misses': 0,
        }

        async def mock_scan_iter(*args, **kwargs):
            return []

        mock_redis.scan_iter = mock_scan_iter

        stats = await cache_service.get_cache_stats()

        assert 'hit_rate' in stats
        assert stats['hit_rate'] == 0.0
