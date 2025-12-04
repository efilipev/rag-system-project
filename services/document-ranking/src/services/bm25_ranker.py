"""
BM25 based document ranker implementation (lightweight alternative)
"""
import logging
import asyncio
from typing import List, Tuple
from rank_bm25 import BM25Okapi
import re

from src.models.schemas import Document
from src.services.base_ranker import BaseRanker

logger = logging.getLogger(__name__)


class BM25Ranker(BaseRanker):
    """
    BM25 based ranker for keyword-based document ranking
    Follows Single Responsibility Principle - handles only BM25 ranking

    This is a lightweight alternative to cross-encoders, useful for:
    - Fast ranking without GPU requirements
    - Keyword-based queries
    - Systems with limited resources
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 ranker

        Args:
            k1: BM25 k1 parameter (term frequency saturation)
            b: BM25 b parameter (length normalization)
        """
        self.k1 = k1
        self.b = b
        self.model_name = f"BM25(k1={k1}, b={b})"

        logger.info(f"Initialized BM25Ranker with k1={k1}, b={b}")

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization (can be improved with spaCy or NLTK)

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        # Convert to lowercase and split by non-alphanumeric characters
        text = text.lower()
        tokens = re.findall(r'\w+', text)
        return tokens

    def _prepare_corpus(self, documents: List[Document]) -> Tuple[List[List[str]], List[str]]:
        """
        Prepare document corpus for BM25

        Args:
            documents: List of documents

        Returns:
            Tuple of (tokenized documents, document texts)
        """
        tokenized_corpus = []
        doc_texts = []

        for doc in documents:
            # Combine title and content
            doc_text = doc.content
            if doc.title:
                doc_text = f"{doc.title}. {doc.content}"

            doc_texts.append(doc_text)
            tokens = self._tokenize(doc_text)
            tokenized_corpus.append(tokens)

        return tokenized_corpus, doc_texts

    def _rank_documents(
        self,
        query: str,
        documents: List[Document],
        top_k: int = None
    ) -> List[Tuple[Document, float]]:
        """
        Internal ranking method (synchronous)

        Args:
            query: Search query
            documents: List of documents to rank
            top_k: Number of top documents to return

        Returns:
            List of (document, score) tuples
        """
        if not documents:
            return []

        # Prepare corpus
        tokenized_corpus, _ = self._prepare_corpus(documents)

        # Initialize BM25
        bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)

        # Tokenize query
        tokenized_query = self._tokenize(query)

        # Get scores
        scores = bm25.get_scores(tokenized_query)

        # Create scored documents
        scored_docs = list(zip(documents, scores))

        # Sort by score (descending)
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Apply top_k if specified
        if top_k is not None and top_k > 0:
            scored_docs = scored_docs[:top_k]

        return scored_docs

    async def rank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = None
    ) -> List[Tuple[Document, float]]:
        """
        Rank documents based on relevance to query using BM25

        Args:
            query: Search query
            documents: List of documents to rank
            top_k: Number of top documents to return

        Returns:
            List of (document, score) tuples sorted by relevance
        """
        try:
            logger.info(f"BM25 ranking {len(documents)} documents for query: {query[:100]}...")

            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            scored_docs = await loop.run_in_executor(
                None,
                self._rank_documents,
                query,
                documents,
                top_k
            )

            if scored_docs:
                logger.info(f"BM25 ranked {len(scored_docs)} documents. Top score: {scored_docs[0][1]:.4f}")

            return scored_docs

        except Exception as e:
            logger.error(f"Error in BM25 ranking: {e}", exc_info=True)
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
        try:
            logger.info(f"BM25 batch ranking {len(documents)} documents for {len(queries)} queries...")

            # Prepare corpus once for all queries
            tokenized_corpus, _ = self._prepare_corpus(documents)
            bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)

            def rank_all_queries():
                results = []
                for query in queries:
                    tokenized_query = self._tokenize(query)
                    scores = bm25.get_scores(tokenized_query)

                    scored_docs = list(zip(documents, scores))
                    scored_docs.sort(key=lambda x: x[1], reverse=True)

                    if top_k is not None and top_k > 0:
                        scored_docs = scored_docs[:top_k]

                    results.append(scored_docs)

                return results

            # Run in executor
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, rank_all_queries)

            logger.info(f"BM25 batch ranking completed for {len(queries)} queries")

            return results

        except Exception as e:
            logger.error(f"Error in BM25 batch ranking: {e}", exc_info=True)
            raise

    def get_model_name(self) -> str:
        """Get the name of the ranking model"""
        return self.model_name

    async def health_check(self) -> bool:
        """
        Check if the ranker is healthy

        Returns:
            True (BM25 is always ready)
        """
        return True

    async def close(self):
        """Clean up resources"""
        # BM25 doesn't need explicit cleanup
        logger.info("BM25Ranker closed")
