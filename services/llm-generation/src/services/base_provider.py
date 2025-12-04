"""
Abstract base class for LLM providers following SOLID principles
"""
from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Dict, Any
from src.models.schemas import GenerationRequest, GenerationResponse, StreamChunk


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers (Dependency Inversion Principle)

    This interface defines the contract that all LLM providers must implement,
    allowing the system to work with any LLM backend without knowing the specifics.
    """

    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """
        Generate a response for the given request

        Args:
            request: Generation request with query and context

        Returns:
            Generated response

        Raises:
            Exception: If generation fails
        """
        pass

    @abstractmethod
    async def generate_stream(
        self,
        request: GenerationRequest
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Generate a streaming response for the given request

        Args:
            request: Generation request with query and context

        Yields:
            Stream chunks

        Raises:
            Exception: If generation fails
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str, model: str) -> int:
        """
        Count tokens in text for a specific model

        Args:
            text: Text to count tokens for
            model: Model name

        Returns:
            Token count
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the provider is healthy and ready

        Returns:
            True if healthy, False otherwise
        """
        pass

    @abstractmethod
    async def close(self):
        """Clean up resources"""
        pass

    @abstractmethod
    def get_available_models(self) -> List[str]:
        """
        Get list of available models for this provider

        Returns:
            List of model names
        """
        pass
