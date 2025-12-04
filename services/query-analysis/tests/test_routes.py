"""
Unit tests for API Routes
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, Mock, patch
from app.main import app
from app.models.schemas import QueryAnalysis


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def mock_analyzer_service():
    """Mock analyzer service"""
    analyzer = AsyncMock()
    analyzer.analyze_query = AsyncMock(return_value=QueryAnalysis(
        original_query="What is Python?",
        normalized_query="what is python?",
        keywords=["python"],
        entities=[],
        intent={"intent": "definitional", "confidence": 0.9},
        language="en",
        embedding=[0.1] * 384,
        query_type="what_is"
    ))
    analyzer.expand_query = AsyncMock(return_value={
        "original_query": "What is Python?",
        "all_expansions": ["python definition", "what python language"],
        "expansion_methods": ["synonyms", "reformulation"],
        "total_expansions": 2
    })
    return analyzer


@pytest.fixture
def mock_cache_service():
    """Mock cache service"""
    cache = AsyncMock()
    cache.get_cache_stats = AsyncMock(return_value={
        "enabled": True,
        "embedding_cache_count": 10,
        "analysis_cache_count": 8,
        "expansion_cache_count": 3,
        "total_cached_items": 21,
        "hit_rate": 0.75
    })
    cache.invalidate_query = AsyncMock(return_value=True)
    cache.clear_all_cache = AsyncMock(return_value=True)
    return cache


@pytest.fixture
def mock_consumer_manager():
    """Mock consumer manager"""
    manager = AsyncMock()
    manager.get_analytics_summary = AsyncMock(return_value={
        "date": "2025-11-12",
        "total_queries": 100,
        "popular_queries": [
            {"query": "what is python?", "count": 25}
        ],
        "intent_distribution": [
            {"intent": "definitional", "count": 40}
        ],
        "popular_keywords": [
            {"keyword": "python", "count": 50}
        ],
        "active_users": 15
    })
    return manager


class TestHealthEndpoint:
    """Test health check endpoint"""

    def test_health_check(self, client):
        """Test health check returns success"""
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "query-analysis"


class TestAnalyzeEndpoint:
    """Test analyze query endpoint"""

    def test_analyze_query_success(self, client, mock_analyzer_service, monkeypatch):
        """Test successful query analysis"""
        # Mock the app state
        with patch.object(app.state, 'analyzer_service', mock_analyzer_service):
            with patch.object(app.state, 'mq_service', AsyncMock()):
                response = client.post(
                    "/api/v1/analyze",
                    json={"query": "What is Python?", "user_id": "user1"}
                )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "analysis" in data
        assert "processing_time_ms" in data

    def test_analyze_query_invalid_input(self, client):
        """Test analysis with invalid input"""
        response = client.post(
            "/api/v1/analyze",
            json={"invalid_field": "value"}
        )

        assert response.status_code == 422  # Validation error

    def test_analyze_query_empty_query(self, client, mock_analyzer_service):
        """Test analysis with empty query"""
        mock_analyzer_service.analyze_query.side_effect = ValueError("Query cannot be empty")

        with patch.object(app.state, 'analyzer_service', mock_analyzer_service):
            response = client.post(
                "/api/v1/analyze",
                json={"query": ""}
            )

        # Should return 400 for invalid query
        assert response.status_code in [400, 422]

    def test_analyze_query_without_user_id(self, client, mock_analyzer_service):
        """Test analysis without user_id (should default to anonymous)"""
        with patch.object(app.state, 'analyzer_service', mock_analyzer_service):
            with patch.object(app.state, 'mq_service', AsyncMock()):
                response = client.post(
                    "/api/v1/analyze",
                    json={"query": "What is Python?"}
                )

        assert response.status_code == 200


class TestExpandEndpoint:
    """Test query expansion endpoint"""

    def test_expand_query_success(self, client, mock_analyzer_service):
        """Test successful query expansion"""
        with patch.object(app.state, 'analyzer_service', mock_analyzer_service):
            response = client.post(
                "/api/v1/expand",
                json={"query": "What is Python?"}
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "expansion" in data
        assert "all_expansions" in data["expansion"]

    def test_expand_query_error(self, client, mock_analyzer_service):
        """Test query expansion error handling"""
        mock_analyzer_service.expand_query.side_effect = Exception("Expansion failed")

        with patch.object(app.state, 'analyzer_service', mock_analyzer_service):
            response = client.post(
                "/api/v1/expand",
                json={"query": "What is Python?"}
            )

        assert response.status_code == 500


class TestCacheEndpoints:
    """Test cache-related endpoints"""

    def test_get_cache_stats(self, client, mock_cache_service):
        """Test getting cache statistics"""
        with patch.object(app.state, 'cache_service', mock_cache_service):
            response = client.get("/api/v1/cache/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "stats" in data
        assert data["stats"]["hit_rate"] == 0.75

    def test_get_cache_stats_unavailable(self, client):
        """Test cache stats when cache service unavailable"""
        with patch.object(app.state, 'cache_service', None):
            response = client.get("/api/v1/cache/stats")

        assert response.status_code == 503

    def test_invalidate_cache(self, client, mock_cache_service):
        """Test cache invalidation"""
        with patch.object(app.state, 'cache_service', mock_cache_service):
            response = client.delete(
                "/api/v1/cache/invalidate",
                params={"query": "What is Python?"}
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_clear_cache(self, client, mock_cache_service):
        """Test clearing all cache"""
        with patch.object(app.state, 'cache_service', mock_cache_service):
            response = client.delete("/api/v1/cache/clear")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


class TestAnalyticsEndpoint:
    """Test analytics endpoint"""

    def test_get_analytics_with_date(self, client, mock_consumer_manager):
        """Test getting analytics for specific date"""
        with patch.object(app.state, 'consumer_manager', mock_consumer_manager):
            response = client.get(
                "/api/v1/analytics",
                params={"date": "2025-11-12"}
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "analytics" in data
        assert data["analytics"]["date"] == "2025-11-12"
        assert data["analytics"]["total_queries"] == 100

    def test_get_analytics_without_date(self, client, mock_consumer_manager):
        """Test getting analytics without date (defaults to today)"""
        with patch.object(app.state, 'consumer_manager', mock_consumer_manager):
            response = client.get("/api/v1/analytics")

        assert response.status_code == 200

    def test_get_analytics_unavailable(self, client):
        """Test analytics when consumer manager unavailable"""
        with patch.object(app.state, 'consumer_manager', None):
            response = client.get("/api/v1/analytics")

        assert response.status_code == 503


class TestHistoryEndpoint:
    """Test query history endpoint"""

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client for history tests"""
        redis = AsyncMock()
        redis.lrange = AsyncMock(return_value=[b"query-1", b"query-2"])
        redis.hgetall = AsyncMock(return_value={
            b"query_id": b"query-1",
            b"user_id": b"user1",
            b"original_query": b"What is Python?",
            b"timestamp": b"2025-11-12T10:00:00"
        })
        return redis

    def test_get_user_history(self, client, mock_redis):
        """Test getting user query history"""
        with patch.object(app.state, 'redis_client', mock_redis):
            response = client.get("/api/v1/history/user1")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["user_id"] == "user1"
        assert "history" in data

    def test_get_user_history_with_limit(self, client, mock_redis):
        """Test getting user history with custom limit"""
        with patch.object(app.state, 'redis_client', mock_redis):
            response = client.get(
                "/api/v1/history/user1",
                params={"limit": 5}
            )

        assert response.status_code == 200

    def test_get_user_history_invalid_limit(self, client, mock_redis):
        """Test history with invalid limit"""
        with patch.object(app.state, 'redis_client', mock_redis):
            response = client.get(
                "/api/v1/history/user1",
                params={"limit": 200}  # Exceeds max of 100
            )

        assert response.status_code == 422  # Validation error

    def test_get_user_history_unavailable(self, client):
        """Test history when Redis unavailable"""
        with patch.object(app.state, 'redis_client', None):
            response = client.get("/api/v1/history/user1")

        assert response.status_code == 503


class TestIntegrationScenarios:
    """Test integrated scenarios"""

    def test_full_query_analysis_workflow(self, client, mock_analyzer_service, mock_cache_service):
        """Test complete workflow: analyze -> cache -> stats"""
        with patch.object(app.state, 'analyzer_service', mock_analyzer_service):
            with patch.object(app.state, 'cache_service', mock_cache_service):
                with patch.object(app.state, 'mq_service', AsyncMock()):
                    # Step 1: Analyze query
                    analyze_response = client.post(
                        "/api/v1/analyze",
                        json={"query": "What is Python?", "user_id": "user1"}
                    )
                    assert analyze_response.status_code == 200

                    # Step 2: Check cache stats
                    stats_response = client.get("/api/v1/cache/stats")
                    assert stats_response.status_code == 200

    def test_concurrent_requests(self, client, mock_analyzer_service):
        """Test handling of concurrent requests"""
        with patch.object(app.state, 'analyzer_service', mock_analyzer_service):
            with patch.object(app.state, 'mq_service', AsyncMock()):
                # Simulate multiple concurrent requests
                responses = [
                    client.post("/api/v1/analyze", json={"query": f"Query {i}"})
                    for i in range(5)
                ]

                # All should succeed
                assert all(r.status_code == 200 for r in responses)
