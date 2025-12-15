"""
FastAPI dependencies for LaTeX Parser Service.
"""
from fastapi import HTTPException, status
from src.services.parser_factory import ParserFactory, ParsingService


def get_parsing_service() -> ParsingService:
    """
    Dependency to get the parsing service instance.

    Returns:
        ParsingService instance with the configured parser.

    Raises:
        HTTPException: If the parser is not initialized.
    """
    parser = ParserFactory.get_parser()
    if parser is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Parser not initialized"
        )
    return ParsingService(parser)
