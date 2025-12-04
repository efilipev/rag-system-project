"""
API routes for Document Ranking Service.
"""
import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, status, Depends

from src.models.schemas import (
    RankingRequest,
    RankingResponse,
    RankedDocument,
    BatchRankingRequest,
    BatchRankingResponse,
    ErrorResponse
)
from src.services.ranker_factory import RankingService
from src.api.dependencies import get_ranking_service
from src.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
async def health_check(
    service: RankingService = Depends(get_ranking_service)
) -> Dict[str, Any]:
    """
    Health check endpoint.

    :param service: Injected RankingService instance.
    :return: Health status dictionary.
    """
    try:
        is_healthy = await service.health_check()

        return {
            "status": "healthy" if is_healthy else "degraded",
            "ranker_type": settings.RANKER_TYPE,
            "model": service.get_model_name()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@router.get("/info")
async def get_info(
    service: RankingService = Depends(get_ranking_service)
) -> Dict[str, Any]:
    """
    Get service information.

    :param service: Injected RankingService instance.
    :return: Service configuration and capabilities dictionary.
    :raises HTTPException: If failed to get info.
    """
    try:
        return {
            "service": settings.SERVICE_NAME,
            "ranker_type": settings.RANKER_TYPE,
            "model": service.get_model_name(),
            "max_documents": settings.MAX_DOCUMENTS_PER_REQUEST,
            "default_top_k": settings.DEFAULT_TOP_K,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get info: {str(e)}"
        )


@router.post(
    "/rank",
    response_model=RankingResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
        503: {"model": ErrorResponse, "description": "Service Unavailable"}
    }
)
async def rank_documents(
    request: RankingRequest,
    service: RankingService = Depends(get_ranking_service)
) -> RankingResponse:
    """
    Rank documents based on relevance to query.

    :param request: Ranking request with query and documents.
    :param service: Injected RankingService instance.
    :return: Ranked documents with scores.
    :raises HTTPException: If validation fails or ranking fails.
    """
    try:
        # Validate request
        if len(request.documents) > settings.MAX_DOCUMENTS_PER_REQUEST:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Maximum {settings.MAX_DOCUMENTS_PER_REQUEST} documents allowed"
            )

        if len(request.query) > settings.MAX_QUERY_LENGTH:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Query exceeds maximum length of {settings.MAX_QUERY_LENGTH} characters"
            )

        logger.info(f"Ranking {len(request.documents)} documents for query: {request.query[:100]}...")

        top_k = request.top_k or settings.DEFAULT_TOP_K

        scored_docs = await service.rank(
            query=request.query,
            documents=request.documents,
            top_k=top_k
        )

        # Build response
        ranked_documents = []
        for idx, (doc, score) in enumerate(scored_docs, 1):
            ranked_doc = RankedDocument(
                id=doc.id,
                content=doc.content,
                title=doc.title,
                source=doc.source,
                relevance_score=float(score),
                rank_position=idx,
                original_score=doc.score,
                metadata=doc.metadata
            )
            ranked_documents.append(ranked_doc)

        logger.info(f"Successfully ranked {len(ranked_documents)} documents")

        return RankingResponse(
            query=request.query,
            ranked_documents=ranked_documents,
            total_documents=len(request.documents),
            model_used=service.get_model_name(),
            metadata={
                "top_k": top_k,
                "ranker_type": settings.RANKER_TYPE
            }
        )

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Ranking failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ranking failed: {str(e)}"
        )


@router.post(
    "/rank/batch",
    response_model=BatchRankingResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
        503: {"model": ErrorResponse, "description": "Service Unavailable"}
    }
)
async def rank_documents_batch(
    request: BatchRankingRequest,
    service: RankingService = Depends(get_ranking_service)
) -> BatchRankingResponse:
    """
    Rank documents for multiple queries in batch.

    :param request: Batch ranking request with queries and documents.
    :param service: Injected RankingService instance.
    :return: Ranked documents for each query.
    :raises HTTPException: If validation fails or ranking fails.
    """
    try:
        # Validate request
        if len(request.documents) > settings.MAX_DOCUMENTS_PER_REQUEST:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Maximum {settings.MAX_DOCUMENTS_PER_REQUEST} documents allowed"
            )

        for query in request.queries:
            if len(query) > settings.MAX_QUERY_LENGTH:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Query exceeds maximum length of {settings.MAX_QUERY_LENGTH} characters"
                )

        logger.info(f"Batch ranking {len(request.documents)} documents for {len(request.queries)} queries...")

        top_k = request.top_k or settings.DEFAULT_TOP_K

        batch_results = await service.rank_batch(
            queries=request.queries,
            documents=request.documents,
            top_k=top_k
        )

        # Build responses
        responses = []
        for query, scored_docs in zip(request.queries, batch_results):
            ranked_documents = []
            for idx, (doc, score) in enumerate(scored_docs, 1):
                ranked_doc = RankedDocument(
                    id=doc.id,
                    content=doc.content,
                    title=doc.title,
                    source=doc.source,
                    relevance_score=float(score),
                    rank_position=idx,
                    original_score=doc.score,
                    metadata=doc.metadata
                )
                ranked_documents.append(ranked_doc)

            response = RankingResponse(
                query=query,
                ranked_documents=ranked_documents,
                total_documents=len(request.documents),
                model_used=service.get_model_name(),
                metadata={
                    "top_k": top_k,
                    "ranker_type": settings.RANKER_TYPE
                }
            )
            responses.append(response)

        logger.info(f"Successfully batch ranked for {len(request.queries)} queries")

        return BatchRankingResponse(
            results=responses,
            model_used=service.get_model_name()
        )

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Batch ranking failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch ranking failed: {str(e)}"
        )
