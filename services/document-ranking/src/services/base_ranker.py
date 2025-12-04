"""
Abstract base class for document rankers following SOLID principles
"""
from abc import ABC, abstractmethod
from typing import List, Tuple
from src.models.schemas import Document


class BaseRanker(ABC):
    """
    Abstract base class for document rankers (Dependency Inversion Principle)

    This interface defines the contract that all rankers must implement,
    allowing the system to work with any ranking backend without knowing the specifics.
    """

    @abstractmethod
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
            top_k: Number of top documents to return (None = return all)

        Returns:
            List of (document, score) tuples sorted by relevance (highest first)

        Raises:
            Exception: If ranking fails
        """
        pass

    @abstractmethod
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

        Raises:
            Exception: If ranking fails
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """
        Get the name of the ranking model

        Returns:
            Model name string
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the ranker is healthy and ready

        Returns:
            True if healthy, False otherwise
        """
        pass

    @abstractmethod
    async def close(self):
        """Clean up resources"""
        pass
