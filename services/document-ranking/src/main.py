"""
Document Ranking Service using cross-encoder models.
"""
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from src.api.routes import router
from src.core.config import settings
from src.core.logging import setup_logging
from src.services.ranker_factory import RankerFactory

logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan context manager for FastAPI application.

    Handles initialization and cleanup of ranker service.

    :param app: FastAPI application instance.
    :return: Async generator for lifespan management.
    :raises ValueError: If ranker type is unknown.
    """
    logger.info(f"Starting {settings.SERVICE_NAME} service")
    logger.info(f"Initializing ranker: {settings.RANKER_TYPE}")

    try:
        # Create ranker based on configuration (Factory Pattern)
        if settings.RANKER_TYPE == "cross-encoder":
            # Handle 'auto' device setting
            device = settings.CROSS_ENCODER_DEVICE
            if device.lower() == "auto":
                device = None

            ranker = RankerFactory.create_ranker(
                ranker_type="cross-encoder",
                model_name=settings.CROSS_ENCODER_MODEL,
                max_length=settings.CROSS_ENCODER_MAX_LENGTH,
                batch_size=settings.CROSS_ENCODER_BATCH_SIZE,
                device=device
            )
        elif settings.RANKER_TYPE == "bm25":
            ranker = RankerFactory.create_ranker(
                ranker_type="bm25",
                k1=settings.BM25_K1,
                b=settings.BM25_B
            )
        else:
            raise ValueError(f"Unknown ranker type: {settings.RANKER_TYPE}")

        # Store ranker in app state for dependency injection
        app.state.ranker = ranker

        # Health check
        is_healthy = await ranker.health_check()
        if is_healthy:
            logger.info(f"Ranker health check passed")
        else:
            logger.warning(f"Ranker health check failed - service may not work correctly")

        logger.info(f"Using ranker model: {ranker.get_model_name()}")

    except Exception as e:
        logger.error(f"Failed to initialize ranker: {e}", exc_info=True)
        raise

    yield

    # Cleanup
    logger.info("Shutting down service")
    if hasattr(app.state, "ranker") and app.state.ranker:
        await app.state.ranker.close()
        logger.info("Ranker closed and cleaned up")


app = FastAPI(
    title="Document Ranking",
    description="Document Ranking Service using cross-encoder models",
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
    """
    Root health check endpoint.

    :return: Health status dictionary with service name and version.
    """
    return {"status": "healthy", "service": settings.SERVICE_NAME, "version": "0.1.0"}


@app.get("/")
async def root():
    """
    Root endpoint returning service information.

    :return: Service info dictionary with name, version, and status.
    """
    return {"service": settings.SERVICE_NAME, "version": "0.1.0", "status": "running"}
