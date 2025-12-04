"""
API routes for API Gateway.
"""
import logging
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, status, Depends, Request

from src.models.schemas import (
    QueryRequest,
    QueryResponse,
    HealthCheckResponse,
    ErrorResponse
)
from src.services.orchestrator import RAGOrchestrator
from src.api.dependencies import get_orchestrator
from src.core.config import settings
from shared.validation.validators import InputValidator

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
async def health_check(
    orchestrator: RAGOrchestrator = Depends(get_orchestrator)
) -> Dict[str, Any]:
    """
    Health check endpoint.

    :param orchestrator: Injected RAGOrchestrator instance.
    :return: Health status of gateway and all services.
    """
    try:
        service_health = await orchestrator.health_check_all_services()

        # Determine overall status
        all_healthy = all(service_health.values())
        some_healthy = any(service_health.values())

        if all_healthy:
            overall_status = "healthy"
        elif some_healthy:
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"

        return {
            "status": overall_status,
            "services": service_health,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@router.get("/")
async def root() -> Dict[str, Any]:
    """
    Root endpoint with API information.

    :return: API information dictionary with available endpoints.
    """
    return {
        "service": "RAG System API Gateway",
        "version": "0.1.0",
        "status": "running",
        "endpoints": {
            "query": "POST /api/v1/query",
            "health": "GET /api/v1/health",
            "docs": "/docs"
        }
    }


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
        503: {"model": ErrorResponse, "description": "Service Unavailable"}
    }
)
async def query_rag_system(
    request: QueryRequest,
    orchestrator: RAGOrchestrator = Depends(get_orchestrator)
) -> QueryResponse:
    """
    Execute RAG pipeline end-to-end.

    Orchestrates the complete RAG workflow: Query Analysis, Document Retrieval,
    Document Ranking, LaTeX Parsing, LLM Generation, and Response Formatting.

    :param request: Query request with configuration.
    :param orchestrator: Injected orchestrator instance.
    :return: Complete response with formatted answer and sources.
    :raises HTTPException: If pipeline fails.
    """
    try:
        # Enhanced input validation
        is_valid, error_msg = InputValidator.validate_query(
            request.query,
            max_length=settings.MAX_QUERY_LENGTH
        )

        if not is_valid:
            logger.warning(f"Query validation failed: {error_msg}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg
            )

        # Sanitize query
        sanitized_query = InputValidator.sanitize_string(
            request.query,
            max_length=settings.MAX_QUERY_LENGTH,
            allow_html=False,
            remove_control_chars=True
        )

        logger.info(f"Received query: '{sanitized_query[:100]}...'")

        # Execute RAG pipeline with sanitized query
        result = await orchestrator.execute_rag_pipeline(
            query=sanitized_query,
            retrieval_top_k=request.retrieval_top_k or settings.RETRIEVAL_TOP_K,
            ranking_top_k=request.ranking_top_k or settings.RANKING_TOP_K,
            output_format=request.output_format or settings.DEFAULT_OUTPUT_FORMAT,
            enable_query_analysis=request.enable_query_analysis,
            enable_ranking=request.enable_ranking,
            enable_latex_parsing=request.enable_latex_parsing
        )

        # Check if pipeline succeeded
        if not result.get("success", False):
            logger.error(f"Pipeline failed: {result.get('error')}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("error", "Pipeline execution failed")
            )

        logger.info(f"Query processed successfully (correlation_id={result['correlation_id']})")

        return QueryResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}"
        )
