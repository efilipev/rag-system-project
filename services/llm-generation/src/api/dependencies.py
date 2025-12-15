"""
FastAPI dependencies for LLM Generation Service.
"""
from fastapi import HTTPException, status
from src.services.provider_factory import LLMProviderFactory, LLMService


def get_llm_service() -> LLMService:
    """
    Dependency to get the LLM service instance.

    Returns:
        LLMService instance with the configured provider.

    Raises:
        HTTPException: If the provider is not initialized.
    """
    provider = LLMProviderFactory.get_provider()
    if provider is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM provider not initialized"
        )
    return LLMService(provider)
