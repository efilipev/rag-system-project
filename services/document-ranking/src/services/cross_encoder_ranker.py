"""
Cross-Encoder based document ranker implementation
"""
import logging
import asyncio
from typing import List, Tuple, Dict, Any
import torch
from sentence_transformers import CrossEncoder
import numpy as np

from src.models.schemas import Document
from src.services.base_ranker import BaseRanker

logger = logging.getLogger(__name__)


class CrossEncoderRanker(BaseRanker):
    """
    Cross-Encoder based ranker for semantic document ranking
    Follows Single Responsibility Principle - handles only cross-encoder ranking
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length: int = 512,
        batch_size: int = 32,
        device: str = None
    ):
        """
        Initialize Cross-Encoder ranker

        Args:
            model_name: Name of the cross-encoder model from HuggingFace
            max_length: Maximum sequence length
            batch_size: Batch size for processing
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size

        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Initializing CrossEncoderRanker with model: {model_name} on device: {self.device}")

        try:
            # Load cross-encoder model
            self.model = CrossEncoder(
                model_name,
                max_length=max_length,
                device=self.device
            )
            logger.info(f"CrossEncoderRanker initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize CrossEncoderRanker: {e}", exc_info=True)
            raise

    def _prepare_pairs(self, query: str, documents: List[Document]) -> List[List[str]]:
        """
        Prepare query-document pairs for scoring

        Args:
            query: Search query
            documents: List of documents

        Returns:
            List of [query, document_text] pairs
        """
        pairs = []
        for doc in documents:
            # Combine title and content for better ranking
            doc_text = doc.content
            if doc.title:
                doc_text = f"{doc.title}. {doc.content}"

            pairs.append([query, doc_text])

        return pairs

    def _score_pairs(self, pairs: List[List[str]], normalize: bool = True) -> np.ndarray:
        """
        Score query-document pairs using cross-encoder

        Args:
            pairs: List of [query, document] pairs
            normalize: Whether to normalize logits to 0-1 range using sigmoid

        Returns:
            Array of relevance scores (normalized to 0-1 if normalize=True)
        """
        try:
            # Predict scores in batches
            scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )

            # Normalize logits to 0-1 range using sigmoid
            # Cross-encoder returns raw logits (can be negative, typically -10 to +10)
            # Sigmoid converts to probability-like scores (0 to 1)
            if normalize:
                scores = 1 / (1 + np.exp(-scores))

            return scores

        except Exception as e:
            logger.error(f"Error scoring pairs: {e}", exc_info=True)
            raise

    async def rank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = None
    ) -> List[Tuple[Document, float]]:
        """
        Rank documents based on relevance to query

        Args:
            query: Search query
            documents: List of documents to rank
            top_k: Number of top documents to return

        Returns:
            List of (document, score) tuples sorted by relevance
        """
        if not documents:
            return []

        try:
            logger.info(f"Ranking {len(documents)} documents for query: {query[:100]}...")

            # Run scoring in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            pairs = self._prepare_pairs(query, documents)

            # Score pairs (run in executor to avoid blocking)
            scores = await loop.run_in_executor(None, self._score_pairs, pairs)

            # Create scored documents
            scored_docs = list(zip(documents, scores))

            # Sort by score (descending)
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            # Apply top_k if specified
            if top_k is not None and top_k > 0:
                scored_docs = scored_docs[:top_k]

            logger.info(f"Ranked {len(scored_docs)} documents. Top score: {scored_docs[0][1]:.4f}")

            return scored_docs

        except Exception as e:
            logger.error(f"Error ranking documents: {e}", exc_info=True)
            raise

    async def rank_batch(
        self,
        queries: List[str],
        documents: List[Document],
        top_k: int = None
    ) -> List[List[Tuple[Document, float]]]:
        """
        Rank documents for multiple queries in batch

        Args:
            queries: List of search queries
            documents: List of documents to rank
            top_k: Number of top documents to return per query

        Returns:
            List of ranking results, one per query
        """
        if not queries or not documents:
            return [[] for _ in queries]

        try:
            logger.info(f"Batch ranking {len(documents)} documents for {len(queries)} queries...")

            # Prepare all pairs
            all_pairs = []
            query_doc_counts = []

            for query in queries:
                pairs = self._prepare_pairs(query, documents)
                all_pairs.extend(pairs)
                query_doc_counts.append(len(pairs))

            # Run scoring in thread pool
            loop = asyncio.get_event_loop()
            all_scores = await loop.run_in_executor(None, self._score_pairs, all_pairs)

            # Split scores by query
            results = []
            score_idx = 0

            for i, count in enumerate(query_doc_counts):
                query_scores = all_scores[score_idx:score_idx + count]
                scored_docs = list(zip(documents, query_scores))

                # Sort by score
                scored_docs.sort(key=lambda x: x[1], reverse=True)

                # Apply top_k
                if top_k is not None and top_k > 0:
                    scored_docs = scored_docs[:top_k]

                results.append(scored_docs)
                score_idx += count

            logger.info(f"Batch ranking completed for {len(queries)} queries")

            return results

        except Exception as e:
            logger.error(f"Error in batch ranking: {e}", exc_info=True)
            raise

    def get_model_name(self) -> str:
        """Get the name of the ranking model"""
        return self.model_name

    async def health_check(self) -> bool:
        """
        Check if the ranker is healthy and ready

        Returns:
            True if healthy, False otherwise
        """
        try:
            # Try a simple prediction
            test_pairs = [["test query", "test document"]]
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._score_pairs, test_pairs)
            return True

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def close(self):
        """Clean up resources"""
        # Cross-encoder doesn't need explicit cleanup
        logger.info("CrossEncoderRanker closed")
