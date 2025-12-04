"""
LaTeX Parser Service for mathematical formulas
"""
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from src.api.routes import router
from src.core.config import settings
from src.core.logging import setup_logging
from src.services.parser_factory import ParserFactory

logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan context manager for FastAPI application
    Handles initialization and cleanup of parser
    """
    logger.info(f"Starting {settings.SERVICE_NAME} service")
    logger.info(f"Initializing parser: {settings.PARSER_TYPE}")

    try:
        # Create parser based on configuration (Factory Pattern)
        parser = ParserFactory.create_parser(
            parser_type=settings.PARSER_TYPE
        )

        # Set parser in factory (Dependency Injection)
        ParserFactory.set_parser(parser)

        # Health check
        is_healthy = await parser.health_check()
        if is_healthy:
            logger.info(f"Parser health check passed")
        else:
            logger.warning(f"Parser health check failed - service may not work correctly")

        logger.info(f"Using parser: {parser.get_parser_name()}")

    except Exception as e:
        logger.error(f"Failed to initialize parser: {e}", exc_info=True)
        raise

    yield

    # Cleanup
    logger.info("Shutting down service")
    await ParserFactory.close_parser()


app = FastAPI(
    title="Latex Parser",
    description="LaTeX Parser Service for mathematical formulas",
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
