"""
API routes for LLM Generation Service.
"""
import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.responses import StreamingResponse

from src.models.schemas import (
    GenerationRequest,
    GenerationResponse,
    ErrorResponse
)
from src.services.provider_factory import LLMService
from src.api.dependencies import get_llm_service
from src.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
async def health_check(
    service: LLMService = Depends(get_llm_service)
) -> Dict[str, Any]:
    """
    Health check endpoint.

    :param service: Injected LLMService instance.
    :return: Health status dictionary.
    """
    try:
        is_healthy = await service.health_check()

        return {
            "status": "healthy" if is_healthy else "degraded",
            "provider": settings.LLM_PROVIDER,
            "model": settings.DEFAULT_MODEL
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@router.get("/models")
async def list_models(
    service: LLMService = Depends(get_llm_service)
) -> Dict[str, Any]:
    """
    List available models.

    :param service: Injected LLMService instance.
    :return: List of available models.
    :raises HTTPException: If failed to list models.
    """
    try:
        models = service.get_available_models()

        return {
            "provider": settings.LLM_PROVIDER,
            "default_model": settings.DEFAULT_MODEL,
            "available_models": models
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}"
        )


@router.post(
    "/generate",
    response_model=GenerationResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
        503: {"model": ErrorResponse, "description": "Service Unavailable"}
    }
)
async def generate_text(
    request: GenerationRequest,
    service: LLMService = Depends(get_llm_service)
) -> GenerationResponse:
    """
    Generate text response using LLM.

    :param request: Generation request with query and context.
    :param service: Injected LLMService instance.
    :return: Generated response.
    :raises HTTPException: If generation fails.
    """
    try:
        if len(request.context_documents) > settings.MAX_CONTEXT_DOCUMENTS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Maximum {settings.MAX_CONTEXT_DOCUMENTS} context documents allowed"
            )

        if len(request.query) > settings.MAX_QUERY_LENGTH:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Query exceeds maximum length of {settings.MAX_QUERY_LENGTH} characters"
            )

        logger.info(f"Generating response for query: {request.query[:100]}...")

        response = await service.generate(request)

        logger.info(f"Successfully generated response with {response.tokens_used} tokens")

        return response

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {str(e)}"
        )


@router.post("/generate/stream")
async def generate_text_stream(
    request: GenerationRequest,
    service: LLMService = Depends(get_llm_service)
) -> StreamingResponse:
    """
    Generate streaming text response using LLM.

    :param request: Generation request with query and context.
    :param service: Injected LLMService instance.
    :return: Server-sent events stream.
    :raises HTTPException: If generation fails.
    """
    try:
        if len(request.context_documents) > settings.MAX_CONTEXT_DOCUMENTS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Maximum {settings.MAX_CONTEXT_DOCUMENTS} context documents allowed"
            )

        if len(request.query) > settings.MAX_QUERY_LENGTH:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Query exceeds maximum length of {settings.MAX_QUERY_LENGTH} characters"
            )

        logger.info(f"Starting streaming generation for query: {request.query[:100]}...")

        async def event_generator():
            """Generate Server-Sent Events."""
            try:
                async for chunk in service.generate_stream(request):
                    yield f"data: {chunk.json()}\n\n"

                yield "data: [DONE]\n\n"

            except Exception as e:
                logger.error(f"Streaming generation error: {e}", exc_info=True)
                error_data = ErrorResponse(
                    error=str(e),
                    error_code="GENERATION_ERROR",
                    details={}
                )
                yield f"data: {error_data.json()}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start streaming: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start streaming: {str(e)}"
        )
