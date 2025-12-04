"""
Unit tests for Document Ranking Service.
"""

import pytest
from unittest.mock import MagicMock, patch


@pytest.mark.unit
class TestCrossEncoderRanker:
    """Test cases for Cross-Encoder Ranker."""

    def test_rank_documents(self, sample_query, sample_documents):
        """Test document ranking with cross-encoder."""
        from services.document_ranking.app.services.cross_encoder_ranker import CrossEncoderRanker

        ranker = CrossEncoderRanker()
        ranked = ranker.rank(sample_query["query"], sample_documents, top_k=3)

        assert len(ranked) <= 3
        assert all("score" in doc for doc in ranked)
        # Scores should be in descending order
        scores = [doc["score"] for doc in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_rank_empty_documents(self, sample_query):
        """Test ranking with empty document list."""
        from services.document_ranking.app.services.cross_encoder_ranker import CrossEncoderRanker

        ranker = CrossEncoderRanker()
        ranked = ranker.rank(sample_query["query"], [], top_k=5)

        assert ranked == []

    def test_rank_top_k_larger_than_documents(self, sample_query, sample_documents):
        """Test when top_k is larger than number of documents."""
        from services.document_ranking.app.services.cross_encoder_ranker import CrossEncoderRanker

        ranker = CrossEncoderRanker()
        ranked = ranker.rank(sample_query["query"], sample_documents, top_k=100)

        assert len(ranked) == len(sample_documents)


@pytest.mark.unit
class TestBM25Ranker:
    """Test cases for BM25 Ranker."""

    def test_bm25_ranking(self, sample_query, sample_documents):
        """Test BM25 ranking."""
        from services.document_ranking.app.services.bm25_ranker import BM25Ranker

        ranker = BM25Ranker()
        ranked = ranker.rank(sample_query["query"], sample_documents, top_k=3)

        assert len(ranked) <= 3
        assert all("score" in doc for doc in ranked)

    def test_bm25_keyword_match(self):
        """Test BM25 with keyword matching."""
        from services.document_ranking.app.services.bm25_ranker import BM25Ranker

        documents = [
            {"id": "1", "content": "machine learning algorithms"},
            {"id": "2", "content": "deep neural networks"},
            {"id": "3", "content": "machine learning and deep learning"}
        ]

        ranker = BM25Ranker()
        ranked = ranker.rank("machine learning", documents, top_k=2)

        # Documents with "machine learning" should rank higher
        assert ranked[0]["id"] in ["1", "3"]


@pytest.mark.unit
class TestRankingAPI:
    """Test cases for Ranking API endpoints."""

    @pytest.mark.asyncio
    async def test_rank_endpoint(self, async_client, sample_query, sample_documents):
        """Test ranking endpoint."""
        response = await async_client.post(
            "http://localhost:8103/api/v1/rank",
            json={
                "query": sample_query["query"],
                "documents": sample_documents,
                "top_k": 3
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "ranked_documents" in data
        assert len(data["ranked_documents"]) <= 3

    @pytest.mark.asyncio
    async def test_rank_endpoint_invalid_input(self, async_client):
        """Test ranking endpoint with invalid input."""
        response = await async_client.post(
            "http://localhost:8103/api/v1/rank",
            json={
                "query": "",
                "documents": []
            }
        )

        assert response.status_code == 422


@pytest.mark.unit
class TestHybridRanking:
    """Test cases for hybrid ranking strategies."""

    def test_hybrid_ranking_combination(self, sample_query, sample_documents):
        """Test combining multiple ranking strategies."""
        from services.document_ranking.app.services.cross_encoder_ranker import CrossEncoderRanker
        from services.document_ranking.app.services.bm25_ranker import BM25Ranker

        ce_ranker = CrossEncoderRanker()
        bm25_ranker = BM25Ranker()

        ce_ranked = ce_ranker.rank(sample_query["query"], sample_documents, top_k=5)
        bm25_ranked = bm25_ranker.rank(sample_query["query"], sample_documents, top_k=5)

        # Both should return results
        assert len(ce_ranked) > 0
        assert len(bm25_ranked) > 0

    def test_score_normalization(self, sample_documents):
        """Test score normalization for combining rankers."""
        scores = [doc["score"] for doc in sample_documents]

        # Normalize to 0-1 range
        min_score = min(scores)
        max_score = max(scores)
        normalized = [(s - min_score) / (max_score - min_score) for s in scores]

        assert all(0 <= s <= 1 for s in normalized)
