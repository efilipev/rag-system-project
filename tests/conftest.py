"""
Pytest configuration and shared fixtures for RAG System tests.
"""

import asyncio
import pytest
import pytest_asyncio
from typing import AsyncGenerator, Generator
from httpx import AsyncClient
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Create an async HTTP client for testing."""
    async with AsyncClient() as client:
        yield client


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    redis_mock = AsyncMock()
    redis_mock.get.return_value = None
    redis_mock.set.return_value = True
    redis_mock.setex.return_value = True
    redis_mock.delete.return_value = 1
    redis_mock.exists.return_value = 0
    redis_mock.incr.return_value = 1
    redis_mock.expire.return_value = True
    return redis_mock


@pytest.fixture
def mock_postgres():
    """Mock PostgreSQL connection."""
    db_mock = MagicMock()
    return db_mock


@pytest.fixture
def mock_qdrant():
    """Mock Qdrant client."""
    qdrant_mock = AsyncMock()
    qdrant_mock.search.return_value = []
    qdrant_mock.upsert.return_value = True
    return qdrant_mock


@pytest.fixture
def mock_openai():
    """Mock OpenAI client."""
    openai_mock = AsyncMock()
    openai_mock.chat.completions.create.return_value = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(
                    content="This is a test response from the LLM."
                )
            )
        ],
        usage=MagicMock(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150
        )
    )
    return openai_mock


@pytest.fixture
def sample_query():
    """Sample query for testing."""
    return {
        "query": "What is machine learning?",
        "user_id": "test_user_123",
        "session_id": "test_session_456"
    }


@pytest.fixture
def sample_document():
    """Sample document for testing."""
    return {
        "id": "doc_123",
        "title": "Introduction to Machine Learning",
        "content": "Machine learning is a subset of artificial intelligence...",
        "source": "textbook",
        "metadata": {
            "author": "John Doe",
            "year": 2024
        }
    }


@pytest.fixture
def sample_documents():
    """Sample list of documents for testing."""
    return [
        {
            "id": f"doc_{i}",
            "title": f"Document {i}",
            "content": f"Content for document {i}",
            "score": 0.9 - (i * 0.1)
        }
        for i in range(5)
    ]


@pytest.fixture
def sample_embedding():
    """Sample embedding vector for testing."""
    return [0.1] * 384


@pytest.fixture
def mock_jwt_token():
    """Mock JWT token for testing."""
    return "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0X3VzZXIiLCJleHAiOjk5OTk5OTk5OTl9.test"


@pytest.fixture
def mock_api_key():
    """Mock API key for testing."""
    return "rag_test_api_key_123456789"


@pytest.fixture
def test_config():
    """Test configuration."""
    return {
        "LOG_LEVEL": "DEBUG",
        "ENVIRONMENT": "test",
        "ENABLE_AUTHENTICATION": False,
        "ENABLE_RATE_LIMITING": False,
        "REDIS_URL": "redis://localhost:6379/1",  # Use separate DB for tests
        "POSTGRES_URL": "postgresql://test:test@localhost:5432/test_db",
    }


# Markers for test organization
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "smoke: Smoke tests for quick validation")
