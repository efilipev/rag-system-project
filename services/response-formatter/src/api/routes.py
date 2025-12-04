"""
API routes for Response Formatter Service.
"""
import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, status, Depends

from src.models.schemas import (
    FormatRequest,
    FormatResponse,
    BatchFormatRequest,
    BatchFormatResponse,
    TemplateValidationRequest,
    TemplateValidationResponse,
    ErrorResponse
)
from src.services.formatter_factory import FormattingService
from src.api.dependencies import get_formatting_service
from src.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
async def health_check(
    service: FormattingService = Depends(get_formatting_service)
) -> Dict[str, Any]:
    """
    Health check endpoint.

    :param service: Injected FormattingService instance.
    :return: Health status dictionary.
    """
    try:
        is_healthy = await service.health_check()

        return {
            "status": "healthy" if is_healthy else "degraded",
            "formatter": service.get_formatter_name()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@router.get("/info")
async def get_info(
    service: FormattingService = Depends(get_formatting_service)
) -> Dict[str, Any]:
    """
    Get service information.

    :param service: Injected FormattingService instance.
    :return: Service configuration and capabilities dictionary.
    :raises HTTPException: If failed to get info.
    """
    try:
        return {
            "service": settings.SERVICE_NAME,
            "formatter": service.get_formatter_name(),
            "supported_formats": ["markdown", "html", "json", "plain_text"],
            "available_variables": service.get_available_variables(),
            "max_content_length": settings.MAX_CONTENT_LENGTH,
            "max_sources": settings.MAX_SOURCES,
            "max_batch_size": settings.MAX_BATCH_SIZE,
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
    "/format",
    response_model=FormatResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
        503: {"model": ErrorResponse, "description": "Service Unavailable"}
    }
)
async def format_response(
    request: FormatRequest,
    service: FormattingService = Depends(get_formatting_service)
) -> FormatResponse:
    """
    Format content with sources and citations.

    :param request: Format request with content and options.
    :param service: Injected FormattingService instance.
    :return: Formatted content.
    :raises HTTPException: If formatting fails.
    """
    try:
        if len(request.content) > settings.MAX_CONTENT_LENGTH:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Content exceeds maximum length of {settings.MAX_CONTENT_LENGTH} characters"
            )

        logger.info(f"Formatting response to {request.output_format} with {len(request.sources)} sources...")

        result = await service.format(
            content=request.content,
            sources=request.sources,
            output_format=request.output_format,
            query=request.query,
            include_citations=request.include_citations,
            include_metadata=request.include_metadata,
            custom_template=request.custom_template
        )

        logger.info(f"Successfully formatted response")

        return FormatResponse(**result)

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Formatting failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Formatting failed: {str(e)}"
        )


@router.post(
    "/format/batch",
    response_model=BatchFormatResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
        503: {"model": ErrorResponse, "description": "Service Unavailable"}
    }
)
async def format_response_batch(
    request: BatchFormatRequest,
    service: FormattingService = Depends(get_formatting_service)
) -> BatchFormatResponse:
    """
    Format multiple responses in batch.

    :param request: Batch format request.
    :param service: Injected FormattingService instance.
    :return: Batch formatted results.
    :raises HTTPException: If formatting fails.
    """
    try:
        logger.info(f"Batch formatting {len(request.items)} responses...")

        results = []
        total_success = 0
        total_failed = 0

        for item in request.items:
            try:
                result = await service.format(
                    content=item.content,
                    sources=item.sources,
                    output_format=item.output_format,
                    query=item.query,
                    include_citations=item.include_citations,
                    include_metadata=item.include_metadata,
                    custom_template=item.custom_template
                )
                results.append(FormatResponse(**result))
                total_success += 1

            except Exception as e:
                logger.warning(f"Failed to format item: {e}")
                results.append(FormatResponse(
                    formatted_content=f"Error: {str(e)}",
                    output_format=item.output_format,
                    original_query=item.query,
                    num_sources=0,
                    metadata={"error": str(e)}
                ))
                total_failed += 1

        logger.info(f"Batch formatting completed. Success: {total_success}, Failed: {total_failed}")

        return BatchFormatResponse(
            results=results,
            total_processed=len(results),
            total_success=total_success,
            total_failed=total_failed
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch formatting failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch formatting failed: {str(e)}"
        )


@router.post(
    "/template/validate",
    response_model=TemplateValidationResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"}
    }
)
async def validate_template(
    request: TemplateValidationRequest,
    service: FormattingService = Depends(get_formatting_service)
) -> TemplateValidationResponse:
    """
    Validate a Jinja2 template.

    :param request: Template validation request.
    :param service: Injected FormattingService instance.
    :return: Validation result.
    :raises HTTPException: If validation fails.
    """
    try:
        logger.info(f"Validating template...")

        result = await service.validate_template(request.template)

        logger.info(f"Template validation result: {'valid' if result['is_valid'] else 'invalid'}")

        return TemplateValidationResponse(
            is_valid=result["is_valid"],
            error_message=result.get("error_message"),
            available_variables=result.get("available_variables", [])
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Template validation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Template validation failed: {str(e)}"
        )
