"""
Document Retrieval Service - Main Application.

Retrieves relevant documents using Qdrant and Chroma vector databases.
"""
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from src.api.routes import router
from src.core.config import settings
from src.core.logging import setup_logging
from src.services.message_queue import MessageQueueService
from src.services.vector_store import VectorStoreService

logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan context manager for FastAPI application.

    Handles initialization and cleanup of document retrieval services.

    :param app: FastAPI application instance.
    :return: Async generator for lifespan management.
    """
    # Startup
    logger.info(f"Starting {settings.SERVICE_NAME} service")

    # Initialize services
    app.state.mq_service = MessageQueueService()
    await app.state.mq_service.connect()

    app.state.vector_store = VectorStoreService()
    await app.state.vector_store.initialize()

    logger.info("Service started successfully")

    yield

    # Shutdown
    logger.info("Shutting down service")
    await app.state.mq_service.disconnect()
    await app.state.vector_store.close()
    logger.info("Service stopped")


# Create FastAPI app
app = FastAPI(
    title="Document Retrieval Service",
    description="Retrieves relevant documents using vector similarity search",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
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
    return {
        "status": "healthy",
        "service": settings.SERVICE_NAME,
        "version": "0.1.0"
    }


@app.get("/")
async def root():
    """
    Root endpoint returning service information.

    :return: Service info dictionary with name, version, and status.
    """
    return {
        "service": settings.SERVICE_NAME,
        "version": "0.1.0",
        "status": "running"
    }
