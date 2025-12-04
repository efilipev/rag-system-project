"""
Ranker Factory - Factory Pattern with Dependency Injection
"""
import logging
from typing import Optional

from src.services.base_ranker import BaseRanker
from src.services.cross_encoder_ranker import CrossEncoderRanker
from src.services.bm25_ranker import BM25Ranker

logger = logging.getLogger(__name__)


class RankerFactory:
    """
    Factory for creating document rankers (Factory Pattern)

    This class follows the Open/Closed Principle - open for extension
    (can add new rankers) but closed for modification (existing code
    doesn't need to change).
    """

    _instance: Optional['RankerFactory'] = None
    _ranker: Optional[BaseRanker] = None

    def __new__(cls):
        """Singleton pattern to ensure single factory instance"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def create_ranker(
        cls,
        ranker_type: str,
        **kwargs
    ) -> BaseRanker:
        """
        Create a ranker based on configuration

        Args:
            ranker_type: Type of ranker ('cross-encoder', 'bm25')
            **kwargs: Ranker-specific configuration

        Returns:
            Ranker instance

        Raises:
            ValueError: If ranker type is not supported
        """
        ranker_type = ranker_type.lower()

        try:
            if ranker_type == "cross-encoder":
                model_name = kwargs.get(
                    "model_name",
                    "cross-encoder/ms-marco-MiniLM-L-6-v2"
                )
                max_length = kwargs.get("max_length", 512)
                batch_size = kwargs.get("batch_size", 32)
                device = kwargs.get("device", None)

                logger.info(f"Creating CrossEncoderRanker with model: {model_name}")

                return CrossEncoderRanker(
                    model_name=model_name,
                    max_length=max_length,
                    batch_size=batch_size,
                    device=device
                )

            elif ranker_type == "bm25":
                k1 = kwargs.get("k1", 1.5)
                b = kwargs.get("b", 0.75)

                logger.info(f"Creating BM25Ranker with k1={k1}, b={b}")

                return BM25Ranker(k1=k1, b=b)

            else:
                raise ValueError(
                    f"Unsupported ranker type: {ranker_type}. "
                    f"Supported types: cross-encoder, bm25"
                )

        except Exception as e:
            logger.error(f"Failed to create ranker {ranker_type}: {e}")
            raise

    @classmethod
    def get_ranker(cls) -> Optional[BaseRanker]:
        """
        Get the current ranker instance

        Returns:
            Current ranker or None if not initialized
        """
        return cls._ranker

    @classmethod
    def set_ranker(cls, ranker: BaseRanker):
        """
        Set the ranker instance (Dependency Injection)

        Args:
            ranker: Ranker instance to set
        """
        cls._ranker = ranker
        logger.info(f"Ranker set to: {ranker.__class__.__name__}")

    @classmethod
    async def close_ranker(cls):
        """Close the current ranker and cleanup resources"""
        if cls._ranker:
            await cls._ranker.close()
            cls._ranker = None
            logger.info("Ranker closed and cleaned up")


class RankingService:
    """
    High-level service class that uses dependency injection
    to work with any ranker (Dependency Inversion Principle)

    This class depends on the BaseRanker abstraction, not on
    concrete implementations, allowing easy swapping of rankers.
    """

    def __init__(self, ranker: BaseRanker):
        """
        Initialize service with a ranker

        Args:
            ranker: Ranker instance (injected dependency)
        """
        self.ranker = ranker
        logger.info(f"RankingService initialized with {ranker.__class__.__name__}")

    async def rank(self, query: str, documents: list, top_k: int = None):
        """Delegate to ranker"""
        return await self.ranker.rank(query, documents, top_k)

    async def rank_batch(self, queries: list, documents: list, top_k: int = None):
        """Delegate to ranker"""
        return await self.ranker.rank_batch(queries, documents, top_k)

    def get_model_name(self) -> str:
        """Delegate to ranker"""
        return self.ranker.get_model_name()

    async def health_check(self) -> bool:
        """Delegate to ranker"""
        return await self.ranker.health_check()
