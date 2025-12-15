"""
FastAPI dependencies for Response Formatter Service.
"""
from fastapi import HTTPException, status
from src.services.formatter_factory import FormatterFactory, FormattingService


def get_formatting_service() -> FormattingService:
    """
    Dependency to get the formatting service instance.

    Returns:
        FormattingService instance with the configured formatter.

    Raises:
        HTTPException: If the formatter is not initialized.
    """
    formatter = FormatterFactory.get_formatter()
    if formatter is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Formatter not initialized"
        )
    return FormattingService(formatter)
