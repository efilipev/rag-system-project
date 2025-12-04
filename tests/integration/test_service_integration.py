"""
Integration tests for service-to-service communication.
"""

import pytest
import asyncio
from httpx import AsyncClient


@pytest.mark.integration
class TestServiceCommunication:
    """Test inter-service communication."""

    @pytest.mark.asyncio
    async def test_query_to_retrieval_flow(self):
        """Test query analysis to document retrieval flow."""
        async with AsyncClient() as client:
            # Step 1: Analyze query
            analysis_response = await client.post(
                "http://localhost:8101/api/v1/analyze",
                json={
                    "query": "What is machine learning?",
                    "user_id": "test_user",
                    "session_id": "test_session"
                },
                timeout=30.0
            )

            assert analysis_response.status_code == 200
            analysis_data = analysis_response.json()

            # Step 2: Use analysis results for retrieval
            retrieval_response = await client.post(
                "http://localhost:8102/api/v1/retrieve",
                json={
                    "query": analysis_data.get("query", "What is machine learning?"),
                    "embedding": analysis_data.get("embedding"),
                    "top_k": 10
                },
                timeout=30.0
            )

            assert retrieval_response.status_code == 200
            retrieval_data = retrieval_response.json()
            assert "documents" in retrieval_data

    @pytest.mark.asyncio
    async def test_retrieval_to_ranking_flow(self):
        """Test document retrieval to ranking flow."""
        async with AsyncClient() as client:
            # Step 1: Retrieve documents
            retrieval_response = await client.post(
                "http://localhost:8102/api/v1/retrieve",
                json={
                    "query": "machine learning",
                    "top_k": 20
                },
                timeout=30.0
            )

            assert retrieval_response.status_code == 200
            documents = retrieval_response.json().get("documents", [])

            if documents:
                # Step 2: Rank retrieved documents
                ranking_response = await client.post(
                    "http://localhost:8103/api/v1/rank",
                    json={
                        "query": "machine learning",
                        "documents": documents,
                        "top_k": 10
                    },
                    timeout=30.0
                )

                assert ranking_response.status_code == 200
                ranked_data = ranking_response.json()
                assert "ranked_documents" in ranked_data

    @pytest.mark.asyncio
    async def test_full_pipeline(self):
        """Test complete pipeline from query to response."""
        async with AsyncClient() as client:
            # Use API Gateway orchestrator
            response = await client.post(
                "http://localhost:8000/api/v1/query",
                json={
                    "query": "Explain the Pythagorean theorem",
                    "output_format": "markdown",
                    "retrieval_top_k": 20,
                    "ranking_top_k": 10
                },
                timeout=120.0  # LLM can take time
            )

            assert response.status_code == 200
            data = response.json()

            assert data["success"] is True
            assert "response" in data
            assert "correlation_id" in data
            assert "metadata" in data
            assert data["metadata"]["documents_retrieved"] > 0


@pytest.mark.integration
class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens(self):
        """Test that circuit breaker opens after failures."""
        from shared.clients.base_client import BaseHTTPClient

        client = BaseHTTPClient(
            base_url="http://non-existent-service:9999",
            max_retries=3,
            circuit_breaker_threshold=3
        )

        # Make multiple failing requests
        for _ in range(5):
            try:
                await client.get("/test")
            except Exception:
                pass

        # Circuit should be open now
        assert client.circuit_breaker.state == "open"

    @pytest.mark.asyncio
    async def test_retry_mechanism(self):
        """Test retry logic with exponential backoff."""
        from shared.clients.base_client import BaseHTTPClient

        client = BaseHTTPClient(
            base_url="http://localhost:8101",
            max_retries=3
        )

        # This should succeed (eventually)
        response = await client.get("/health")
        assert response is not None


@pytest.mark.integration
class TestCaching:
    """Test caching behavior."""

    @pytest.mark.asyncio
    async def test_embedding_cache_hit(self, mock_redis):
        """Test that embeddings are cached."""
        from services.query_analysis.app.services.cache import get_cached_embedding, cache_embedding

        query = "test query"
        embedding = [0.1] * 384

        # Cache the embedding
        await cache_embedding(query, embedding, mock_redis)

        # Retrieve from cache
        cached = await get_cached_embedding(query, mock_redis)
        assert cached == embedding

    @pytest.mark.asyncio
    async def test_llm_response_cache(self, mock_redis):
        """Test LLM response caching."""
        from services.llm_generation.app.services.cache import get_cached_response, cache_response

        query = "test query"
        context_hash = "abc123"
        response = {"answer": "test answer", "tokens": 100}

        # Cache the response
        await cache_response(query, context_hash, response, mock_redis)

        # Retrieve from cache
        cached = await get_cached_response(query, context_hash, mock_redis)
        assert cached == response


@pytest.mark.integration
@pytest.mark.slow
class TestDatabaseIntegration:
    """Test database integration."""

    @pytest.mark.asyncio
    async def test_document_insertion(self):
        """Test inserting documents into PostgreSQL."""
        # This would require actual database connection
        # Should be run in integration environment
        pass

    @pytest.mark.asyncio
    async def test_vector_search(self):
        """Test vector similarity search."""
        # This would require Qdrant to be running
        pass


@pytest.mark.integration
class TestMessageQueue:
    """Test RabbitMQ message queue."""

    @pytest.mark.asyncio
    async def test_publish_message(self):
        """Test publishing message to RabbitMQ."""
        # This would require RabbitMQ to be running
        pass

    @pytest.mark.asyncio
    async def test_consume_message(self):
        """Test consuming message from RabbitMQ."""
        # This would require RabbitMQ to be running
        pass
