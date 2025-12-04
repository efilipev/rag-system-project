"""
Response Formatter Service with Jinja2 templating
"""
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from src.api.routes import router
from src.core.config import settings
from src.core.logging import setup_logging
from src.services.formatter_factory import FormatterFactory

logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan context manager for FastAPI application
    Handles initialization and cleanup of formatter
    """
    logger.info(f"Starting {settings.SERVICE_NAME} service")
    logger.info(f"Initializing formatter: {settings.FORMATTER_TYPE}")

    try:
        # Create formatter based on configuration (Factory Pattern)
        formatter = FormatterFactory.create_formatter(
            formatter_type=settings.FORMATTER_TYPE
        )

        # Set formatter in factory (Dependency Injection)
        FormatterFactory.set_formatter(formatter)

        # Health check
        is_healthy = await formatter.health_check()
        if is_healthy:
            logger.info(f"Formatter health check passed")
        else:
            logger.warning(f"Formatter health check failed - service may not work correctly")

        logger.info(f"Using formatter: {formatter.get_formatter_name()}")
        logger.info(f"Available template variables: {formatter.get_available_variables()}")

    except Exception as e:
        logger.error(f"Failed to initialize formatter: {e}", exc_info=True)
        raise

    yield

    # Cleanup
    logger.info("Shutting down service")
    await FormatterFactory.close_formatter()


app = FastAPI(
    title="Response Formatter",
    description="Response Formatter Service with Jinja2 templating",
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
