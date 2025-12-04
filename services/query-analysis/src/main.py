"""
Query Analysis Service - Main Application
Analyzes user queries using NLP and LangChain for RAG system
"""
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app
from redis.asyncio import Redis

from src.api.routes import router
from src.core.config import settings
from src.core.logging import setup_logging
from src.services.message_queue import MessageQueueService
from src.services.query_analyzer import QueryAnalyzerService
from src.services.consumers import ConsumerManager
from src.services.cache_service import get_cache_service

# Setup logging
logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Startup and shutdown events
    """
    # Startup
    logger.info(f"Starting {settings.SERVICE_NAME} service")

    # Initialize Redis client
    try:
        app.state.redis_client = Redis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=False
        )
        await app.state.redis_client.ping()
        logger.info("Connected to Redis successfully")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}. Caching disabled.")
        app.state.redis_client = None

    # Initialize cache service
    cache_service = None
    if app.state.redis_client:
        cache_service = get_cache_service(app.state.redis_client)
        app.state.cache_service = cache_service
        logger.info("Cache service initialized")
    else:
        app.state.cache_service = None

    # Initialize message queue
    app.state.mq_service = MessageQueueService()
    await app.state.mq_service.connect()

    # Initialize query analyzer (with cache service and LaTeX parser URL)
    app.state.analyzer_service = QueryAnalyzerService(
        latex_parser_url=settings.LATEX_PARSER_URL,
        cache_service=cache_service
    )
    await app.state.analyzer_service.initialize()

    # Initialize and start consumer services
    if app.state.redis_client:
        try:
            app.state.consumer_manager = ConsumerManager(
                app.state.mq_service,
                app.state.redis_client
            )
            await app.state.consumer_manager.start_consumers()
            logger.info("Consumer services started successfully")
        except Exception as e:
            logger.error(f"Failed to start consumer services: {e}", exc_info=True)
            app.state.consumer_manager = None
    else:
        logger.warning("Consumer services not started (Redis unavailable)")
        app.state.consumer_manager = None

    logger.info("Service started successfully")

    yield

    # Shutdown
    logger.info("Shutting down service")

    # Close Redis connection
    if app.state.redis_client:
        await app.state.redis_client.close()

    # Close RabbitMQ connection
    await app.state.mq_service.disconnect()

    logger.info("Service stopped")


# Create FastAPI app
app = FastAPI(
    title="Query Analysis Service",
    description="Analyzes user queries using NLP and LangChain",
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
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": settings.SERVICE_NAME,
        "version": "0.1.0"
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": settings.SERVICE_NAME,
        "version": "0.1.0",
        "status": "running"
    }
