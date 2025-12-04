"""
API routes for LaTeX Parser Service.
"""
import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, status, Depends

from src.models.schemas import (
    ParseRequest,
    ParseResponse,
    BatchParseRequest,
    BatchParseResponse,
    ValidationResult,
    ErrorResponse
)
from src.services.parser_factory import ParsingService
from src.api.dependencies import get_parsing_service
from src.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
async def health_check(
    service: ParsingService = Depends(get_parsing_service)
) -> Dict[str, Any]:
    """
    Health check endpoint.

    :param service: Injected ParsingService instance.
    :return: Health status dictionary.
    """
    try:
        is_healthy = await service.health_check()

        return {
            "status": "healthy" if is_healthy else "degraded",
            "parser": service.get_parser_name()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@router.get("/info")
async def get_info(
    service: ParsingService = Depends(get_parsing_service)
) -> Dict[str, Any]:
    """
    Get service information.

    :param service: Injected ParsingService instance.
    :return: Service configuration and capabilities dictionary.
    :raises HTTPException: If failed to get info.
    """
    try:
        return {
            "service": settings.SERVICE_NAME,
            "parser": service.get_parser_name(),
            "max_latex_length": settings.MAX_LATEX_LENGTH,
            "max_batch_size": settings.MAX_BATCH_SIZE,
            "supported_formats": ["mathml", "text", "unicode", "simplified", "latex"]
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
    "/parse",
    response_model=ParseResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
        503: {"model": ErrorResponse, "description": "Service Unavailable"}
    }
)
async def parse_latex(
    request: ParseRequest,
    service: ParsingService = Depends(get_parsing_service)
) -> ParseResponse:
    """
    Parse LaTeX formula to desired format.

    :param request: Parse request with LaTeX string and options.
    :param service: Injected ParsingService instance.
    :return: Parsed LaTeX in requested format.
    :raises HTTPException: If parsing fails.
    """
    try:
        logger.info(f"Parsing LaTeX: {request.latex_string[:100]}...")

        if request.validate_only:
            validation = await service.validate(request.latex_string)
            return ParseResponse(
                original_latex=request.latex_string,
                parsed_output="",
                output_format=request.output_format,
                is_valid=validation.is_valid,
                simplified_form=None,
                metadata={
                    "validation": validation.dict(),
                    "validate_only": True
                }
            )

        result = await service.parse(
            latex_string=request.latex_string,
            output_format=request.output_format,
            simplify=request.simplify
        )

        logger.info(f"Successfully parsed LaTeX to {request.output_format}")

        return ParseResponse(**result)

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Parsing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Parsing failed: {str(e)}"
        )


@router.post("/validate", response_model=ValidationResult)
async def validate_latex(
    latex_string: str,
    service: ParsingService = Depends(get_parsing_service)
) -> ValidationResult:
    """
    Validate LaTeX syntax.

    :param latex_string: LaTeX formula to validate.
    :param service: Injected ParsingService instance.
    :return: Validation result.
    :raises HTTPException: If validation fails.
    """
    try:
        if len(latex_string) > settings.MAX_LATEX_LENGTH:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"LaTeX string exceeds maximum length of {settings.MAX_LATEX_LENGTH} characters"
            )

        logger.info(f"Validating LaTeX: {latex_string[:100]}...")

        result = await service.validate(latex_string)

        logger.info(f"Validation result: {'valid' if result.is_valid else 'invalid'}")

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Validation failed: {str(e)}"
        )


@router.post(
    "/parse/batch",
    response_model=BatchParseResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
        503: {"model": ErrorResponse, "description": "Service Unavailable"}
    }
)
async def parse_latex_batch(
    request: BatchParseRequest,
    service: ParsingService = Depends(get_parsing_service)
) -> BatchParseResponse:
    """
    Parse multiple LaTeX formulas in batch.

    :param request: Batch parse request with LaTeX strings and options.
    :param service: Injected ParsingService instance.
    :return: Parsed results for each formula.
    :raises HTTPException: If parsing fails.
    """
    try:
        logger.info(f"Batch parsing {len(request.latex_strings)} LaTeX formulas...")

        results = await service.parse_batch(
            latex_strings=request.latex_strings,
            output_format=request.output_format,
            simplify=request.simplify
        )

        parse_responses = [ParseResponse(**result) for result in results]

        total_valid = sum(1 for r in parse_responses if r.is_valid)
        total_invalid = len(parse_responses) - total_valid

        logger.info(f"Batch parsing completed. Valid: {total_valid}, Invalid: {total_invalid}")

        return BatchParseResponse(
            results=parse_responses,
            total_processed=len(parse_responses),
            total_valid=total_valid,
            total_invalid=total_invalid
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
        logger.error(f"Batch parsing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch parsing failed: {str(e)}"
        )
