"""
FastAPI dependencies for Document Ranking Service.
"""
from fastapi import Request, HTTPException, status

from src.services.ranker_factory import RankingService


def get_ranking_service(request: Request) -> RankingService:
    """
    Dependency to get the ranking service instance.

    Args:
        request: FastAPI request object.

    Returns:
        RankingService instance with the configured ranker.

    Raises:
        HTTPException: If the ranker is not initialized.
    """
    ranker = getattr(request.app.state, "ranker", None)
    if ranker is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ranker not initialized"
        )
    return RankingService(ranker)
