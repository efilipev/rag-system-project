"""
LLM Provider Factory - Factory Pattern with Dependency Injection
"""
import logging
from typing import Optional
from src.services.base_provider import BaseLLMProvider
from src.services.openai_provider import OpenAIProvider
from src.services.local_provider import LocalLLMProvider

logger = logging.getLogger(__name__)


class LLMProviderFactory:
    """
    Factory for creating LLM providers (Factory Pattern)

    This class follows the Open/Closed Principle - open for extension
    (can add new providers) but closed for modification (existing code
    doesn't need to change).
    """

    _instance: Optional['LLMProviderFactory'] = None
    _provider: Optional[BaseLLMProvider] = None

    def __new__(cls):
        """Singleton pattern to ensure single factory instance"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def create_provider(
        cls,
        provider_type: str,
        **kwargs
    ) -> BaseLLMProvider:
        """
        Create an LLM provider based on configuration

        Args:
            provider_type: Type of provider ('openai', 'local', 'ollama', 'vllm')
            **kwargs: Provider-specific configuration

        Returns:
            LLM provider instance

        Raises:
            ValueError: If provider type is not supported
        """
        provider_type = provider_type.lower()

        try:
            if provider_type == "openai":
                api_key = kwargs.get("api_key")
                if not api_key:
                    raise ValueError("OpenAI API key is required")

                default_model = kwargs.get("default_model", "gpt-3.5-turbo")
                logger.info(f"Creating OpenAI provider with model: {default_model}")

                return OpenAIProvider(
                    api_key=api_key,
                    default_model=default_model
                )

            elif provider_type in ["local", "ollama", "vllm", "llamacpp"]:
                base_url = kwargs.get("base_url", "http://localhost:11434")
                default_model = kwargs.get("default_model", "llama3")
                timeout = kwargs.get("timeout", 300)

                # Determine API type
                if provider_type == "local":
                    api_type = kwargs.get("api_type", "ollama")
                else:
                    api_type = provider_type

                logger.info(f"Creating Local LLM provider ({api_type}) at {base_url} with model: {default_model}")

                return LocalLLMProvider(
                    base_url=base_url,
                    default_model=default_model,
                    api_type=api_type,
                    timeout=timeout
                )

            else:
                raise ValueError(
                    f"Unsupported provider type: {provider_type}. "
                    f"Supported types: openai, local, ollama, vllm, llamacpp"
                )

        except Exception as e:
            logger.error(f"Failed to create provider {provider_type}: {e}")
            raise

    @classmethod
    def get_provider(cls) -> Optional[BaseLLMProvider]:
        """
        Get the current provider instance

        Returns:
            Current provider or None if not initialized
        """
        return cls._provider

    @classmethod
    def set_provider(cls, provider: BaseLLMProvider):
        """
        Set the provider instance (Dependency Injection)

        Args:
            provider: Provider instance to set
        """
        cls._provider = provider
        logger.info(f"Provider set to: {provider.__class__.__name__}")

    @classmethod
    async def close_provider(cls):
        """Close the current provider and cleanup resources"""
        if cls._provider:
            await cls._provider.close()
            cls._provider = None
            logger.info("Provider closed and cleaned up")


class LLMService:
    """
    High-level service class that uses dependency injection
    to work with any LLM provider (Dependency Inversion Principle)

    This class depends on the BaseLLMProvider abstraction, not on
    concrete implementations, allowing easy swapping of providers.
    """

    def __init__(self, provider: BaseLLMProvider):
        """
        Initialize service with a provider

        Args:
            provider: LLM provider instance (injected dependency)
        """
        self.provider = provider
        logger.info(f"LLMService initialized with {provider.__class__.__name__}")

    async def generate(self, request):
        """Delegate to provider"""
        return await self.provider.generate(request)

    async def generate_stream(self, request):
        """Delegate to provider"""
        async for chunk in self.provider.generate_stream(request):
            yield chunk

    def count_tokens(self, text: str, model: str) -> int:
        """Delegate to provider"""
        return self.provider.count_tokens(text, model)

    async def health_check(self) -> bool:
        """Delegate to provider"""
        return await self.provider.health_check()

    def get_available_models(self) -> list:
        """Delegate to provider"""
        return self.provider.get_available_models()
