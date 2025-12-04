"""
End-to-end tests for complete RAG pipeline.
"""

import pytest
import asyncio
from httpx import AsyncClient
import time


@pytest.mark.e2e
class TestCompleteRAGPipeline:
    """End-to-end tests for the complete RAG system."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_simple_query_to_response(self):
        """Test complete flow from query to formatted response."""
        async with AsyncClient() as client:
            response = await client.post(
                "http://localhost:8000/api/v1/query",
                json={
                    "query": "What is artificial intelligence?",
                    "output_format": "markdown"
                },
                timeout=120.0
            )

            assert response.status_code == 200
            data = response.json()

            # Verify response structure
            assert data["success"] is True
            assert len(data["response"]) > 0
            assert data["correlation_id"] is not None
            assert data["output_format"] == "markdown"

            # Verify metadata
            metadata = data["metadata"]
            assert "documents_retrieved" in metadata
            assert "documents_used" in metadata
            assert "tokens_used" in metadata
            assert metadata["documents_retrieved"] > 0

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_latex_query_processing(self):
        """Test query containing LaTeX formulas."""
        async with AsyncClient() as client:
            response = await client.post(
                "http://localhost:8000/api/v1/query",
                json={
                    "query": "Explain the formula $E=mc^2$",
                    "output_format": "html",
                    "enable_latex_parsing": True
                },
                timeout=120.0
            )

            assert response.status_code == 200
            data = response.json()

            assert data["success"] is True
            # Response should contain processed LaTeX
            assert len(data["response"]) > 0

    @pytest.mark.asyncio
    async def test_multiple_output_formats(self):
        """Test different output formats."""
        query = {
            "query": "What is machine learning?",
            "retrieval_top_k": 5,
            "ranking_top_k": 3
        }

        formats = ["markdown", "html", "json", "plain_text"]

        async with AsyncClient() as client:
            for fmt in formats:
                response = await client.post(
                    "http://localhost:8000/api/v1/query",
                    json={**query, "output_format": fmt},
                    timeout=120.0
                )

                assert response.status_code == 200
                data = response.json()
                assert data["output_format"] == fmt
                assert len(data["response"]) > 0

    @pytest.mark.asyncio
    async def test_concurrent_queries(self):
        """Test handling multiple concurrent queries."""
        queries = [
            "What is machine learning?",
            "Explain neural networks",
            "What is deep learning?",
            "How does backpropagation work?",
            "What is gradient descent?"
        ]

        async with AsyncClient() as client:
            tasks = [
                client.post(
                    "http://localhost:8000/api/v1/query",
                    json={"query": q, "output_format": "markdown"},
                    timeout=120.0
                )
                for q in queries
            ]

            responses = await asyncio.gather(*tasks)

            # All should succeed
            assert all(r.status_code == 200 for r in responses)
            # All should have unique correlation IDs
            correlation_ids = [r.json()["correlation_id"] for r in responses]
            assert len(set(correlation_ids)) == len(queries)

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_performance_under_load(self):
        """Test system performance under load."""
        num_requests = 10
        query = {"query": "Test query", "output_format": "markdown"}

        start_time = time.time()

        async with AsyncClient() as client:
            tasks = [
                client.post(
                    "http://localhost:8000/api/v1/query",
                    json=query,
                    timeout=120.0
                )
                for _ in range(num_requests)
            ]

            responses = await asyncio.gather(*tasks, return_exceptions=True)

        end_time = time.time()
        total_time = end_time - start_time

        # Count successes
        successful = sum(1 for r in responses if not isinstance(r, Exception) and r.status_code == 200)
        success_rate = successful / num_requests

        print(f"\nPerformance Test Results:")
        print(f"Total requests: {num_requests}")
        print(f"Successful: {successful}")
        print(f"Success rate: {success_rate * 100}%")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average time per request: {total_time / num_requests:.2f}s")

        # Should have >90% success rate
        assert success_rate > 0.9


@pytest.mark.e2e
class TestErrorHandling:
    """Test error handling in the pipeline."""

    @pytest.mark.asyncio
    async def test_invalid_query(self):
        """Test handling of invalid query."""
        async with AsyncClient() as client:
            response = await client.post(
                "http://localhost:8000/api/v1/query",
                json={"query": "", "output_format": "markdown"},
                timeout=30.0
            )

            assert response.status_code in [400, 422]  # Bad request or validation error

    @pytest.mark.asyncio
    async def test_service_unavailable(self):
        """Test handling when a service is unavailable."""
        # This would require stopping a service
        # In a real test, we'd use chaos engineering
        pass

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test handling of service timeouts."""
        async with AsyncClient() as client:
            # Make request with very short timeout
            try:
                response = await client.post(
                    "http://localhost:8000/api/v1/query",
                    json={"query": "Complex query", "output_format": "markdown"},
                    timeout=0.001  # Impossibly short timeout
                )
            except Exception as e:
                # Should timeout gracefully
                assert "timeout" in str(e).lower()


@pytest.mark.e2e
@pytest.mark.smoke
class TestSmokeTests:
    """Quick smoke tests for deployment validation."""

    @pytest.mark.asyncio
    async def test_all_services_healthy(self):
        """Test that all services respond to health checks."""
        services = [
            ("Query Analysis", "http://localhost:8101/health"),
            ("Document Retrieval", "http://localhost:8102/health"),
            ("Document Ranking", "http://localhost:8103/api/v1/health"),
            ("LaTeX Parser", "http://localhost:8104/api/v1/health"),
            ("LLM Generation", "http://localhost:8105/api/v1/health"),
            ("Response Formatter", "http://localhost:8106/api/v1/health"),
            ("API Gateway", "http://localhost:8000/api/v1/health"),
        ]

        async with AsyncClient() as client:
            for name, url in services:
                try:
                    response = await client.get(url, timeout=10.0)
                    assert response.status_code == 200, f"{name} health check failed"
                    print(f"✓ {name} is healthy")
                except Exception as e:
                    pytest.fail(f"{name} health check failed: {e}")

    @pytest.mark.asyncio
    async def test_basic_query_works(self):
        """Test that basic query functionality works."""
        async with AsyncClient() as client:
            response = await client.post(
                "http://localhost:8000/api/v1/query",
                json={"query": "test", "output_format": "markdown"},
                timeout=60.0
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            print("✓ Basic query works")
