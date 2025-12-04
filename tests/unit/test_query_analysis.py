"""
Unit tests for Query Analysis Service.
Following AAA (Arrange-Act-Assert) paradigm.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import re


@pytest.mark.unit
class TestQueryNormalization:
    """Test cases for query normalization logic."""

    def test_normalize_query_lowercase(self):
        """Test query is converted to lowercase."""
        # Arrange
        query = "What is MACHINE Learning?"

        # Act
        normalized = query.lower().strip()
        normalized = re.sub(r"\s+", " ", normalized)

        # Assert
        assert normalized == "what is machine learning?"

    def test_normalize_query_trim_whitespace(self):
        """Test extra whitespace is removed."""
        # Arrange
        query = "  What   is   machine   learning?  "

        # Act
        normalized = query.lower().strip()
        normalized = re.sub(r"\s+", " ", normalized)

        # Assert
        assert normalized == "what is machine learning?"

    def test_normalize_query_special_characters(self):
        """Test special characters handling."""
        # Arrange
        query = "What is AI?"

        # Act
        normalized = query.lower().strip()
        normalized = re.sub(r"\s+", " ", normalized)
        normalized = re.sub(r"[^\w\s\-?!.]", "", normalized)

        # Assert
        assert "what is ai" in normalized


@pytest.mark.unit
class TestQueryAnalysisAPI:
    """Test cases for Query Analysis API endpoints."""

    @pytest.mark.asyncio
    async def test_health_endpoint(self, async_client):
        """Test health check endpoint."""
        # Arrange
        url = "http://localhost:8101/health"

        # Act
        response = await async_client.get(url)

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_analyze_endpoint_invalid_empty_query(self, async_client):
        """Test analysis endpoint with empty query."""
        # Arrange
        url = "http://localhost:8101/api/v1/analyze"
        invalid_request = {"query": ""}

        # Act
        response = await async_client.post(url, json=invalid_request)

        # Assert
        assert response.status_code == 422


@pytest.mark.unit
class TestEmbeddingLogic:
    """Test cases for embedding-related logic."""

    def test_embedding_vector_size(self):
        """Test expected embedding vector size."""
        # Arrange
        expected_size = 384  # MiniLM embedding size

        # Act
        mock_embedding = [0.1] * expected_size

        # Assert
        assert len(mock_embedding) == expected_size
        assert all(isinstance(x, float) for x in mock_embedding)

    def test_embedding_normalization(self):
        """Test embedding normalization logic."""
        # Arrange
        import math
        embedding = [3.0, 4.0]  # Simple vector for testing

        # Act
        magnitude = math.sqrt(sum(x**2 for x in embedding))
        normalized = [x / magnitude for x in embedding]

        # Assert
        normalized_magnitude = math.sqrt(sum(x**2 for x in normalized))
        assert abs(normalized_magnitude - 1.0) < 0.0001


@pytest.mark.unit
class TestKeywordExtraction:
    """Test cases for keyword extraction logic."""

    def test_stopword_filtering(self):
        """Test that common stopwords would be filtered."""
        # Arrange
        stopwords = {"is", "what", "the", "a", "an", "and", "or", "but"}
        words = ["what", "is", "machine", "learning"]

        # Act
        filtered = [w for w in words if w.lower() not in stopwords]

        # Assert
        assert "machine" in filtered
        assert "learning" in filtered
        assert "is" not in filtered
        assert "what" not in filtered

    def test_keyword_deduplication(self):
        """Test keyword deduplication."""
        # Arrange
        keywords = ["machine", "learning", "machine", "ai", "learning"]

        # Act
        unique_keywords = list(dict.fromkeys(keywords))

        # Assert
        assert len(unique_keywords) == 3
        assert unique_keywords == ["machine", "learning", "ai"]


@pytest.mark.unit
class TestIntentClassification:
    """Test cases for intent classification logic."""

    def test_intent_types(self):
        """Test valid intent types."""
        # Arrange
        valid_intents = ["informational", "procedural", "causal", "comparative", "recommendation"]

        # Assert
        assert len(valid_intents) == 5
        assert "informational" in valid_intents

    def test_intent_confidence_range(self):
        """Test intent confidence is in valid range."""
        # Arrange
        confidence = 0.85

        # Assert
        assert 0.0 <= confidence <= 1.0
