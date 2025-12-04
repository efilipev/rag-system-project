"""
LLM Generation Service using OpenAI or local models
"""
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from src.api.routes import router
from src.core.config import settings
from src.core.logging import setup_logging
from src.services.provider_factory import LLMProviderFactory

logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan context manager for FastAPI application
    Handles initialization and cleanup of LLM provider
    """
    logger.info(f"Starting {settings.SERVICE_NAME} service")
    logger.info(f"Initializing LLM provider: {settings.LLM_PROVIDER}")

    try:
        # Create provider based on configuration (Factory Pattern)
        if settings.LLM_PROVIDER == "openai":
            provider = LLMProviderFactory.create_provider(
                provider_type="openai",
                api_key=settings.OPENAI_API_KEY,
                default_model=settings.DEFAULT_MODEL
            )
        else:
            # Local provider (ollama, vllm, llamacpp)
            provider = LLMProviderFactory.create_provider(
                provider_type=settings.LLM_PROVIDER,
                base_url=settings.LOCAL_LLM_BASE_URL,
                default_model=settings.DEFAULT_MODEL,
                api_type=settings.LOCAL_LLM_API_TYPE,
                timeout=settings.LOCAL_LLM_TIMEOUT
            )

        # Set provider in factory (Dependency Injection)
        LLMProviderFactory.set_provider(provider)

        # Health check
        is_healthy = await provider.health_check()
        if is_healthy:
            logger.info(f"LLM provider health check passed")
        else:
            logger.warning(f"LLM provider health check failed - service may not work correctly")

        logger.info(f"Available models: {provider.get_available_models()}")

    except Exception as e:
        logger.error(f"Failed to initialize LLM provider: {e}", exc_info=True)
        raise

    yield

    # Cleanup
    logger.info("Shutting down service")
    await LLMProviderFactory.close_provider()


app = FastAPI(
    title="Llm Generation",
    description="LLM Generation Service using OpenAI or local models",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

# Add Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": settings.SERVICE_NAME, "version": "0.1.0"}


@app.get("/")
async def root():
    return {"service": settings.SERVICE_NAME, "version": "0.1.0", "status": "running"}
