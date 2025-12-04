"""
API Gateway - Main FastAPI Application.

Orchestrates the complete RAG pipeline across all microservices.
"""
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from src.api import register_routes
from src.core.config import settings
from src.core.logging import setup_logging
from src.services.orchestrator import RAGOrchestrator
from src.middleware.auth_middleware import AuthenticationMiddleware, set_api_key_manager
from src.middleware.rate_limit_middleware import RateLimitMiddleware
from src.middleware.security_headers_middleware import SecurityHeadersMiddleware
from shared.auth.jwt_handler import JWTHandler
from shared.auth.api_key_manager import APIKeyManager
from shared.rate_limiting.redis_rate_limiter import RedisRateLimiter
from shared.audit.audit_logger import init_audit_logger, AuditEventType

logger = setup_logging(settings.LOG_LEVEL)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan context manager for FastAPI application.

    Handles initialization and cleanup of orchestrator and related services.

    :param app: FastAPI application instance.
    :return: Async generator for lifespan management.
    """
    logger.info(f"Starting {settings.SERVICE_NAME} service")

    # Initialize audit logger
    audit_logger = init_audit_logger(
        service_name=settings.SERVICE_NAME,
        log_to_file=True,
        audit_log_path=f"logs/audit_{settings.SERVICE_NAME}.log"
    )
    audit_logger.log_event(AuditEventType.SERVICE_START, details={"version": "0.1.0"})
    logger.info("Audit Logger initialized")

    # Initialize API key manager for service-to-service authentication
    api_key_manager = APIKeyManager(secret_key=settings.JWT_SECRET_KEY)
    set_api_key_manager(api_key_manager)
    logger.info("API Key Manager initialized")

    # Initialize rate limiter
    rate_limiter = RedisRateLimiter(
        redis_url=settings.REDIS_URL,
        requests_per_minute=settings.RATE_LIMIT_REQUESTS_PER_MINUTE,
        burst_size=settings.RATE_LIMIT_BURST,
        enabled=settings.ENABLE_RATE_LIMITING
    )

    if settings.ENABLE_RATE_LIMITING:
        await rate_limiter.connect()
        logger.info("Rate Limiter initialized and connected to Redis")
    else:
        logger.info("Rate Limiting disabled")

    logger.info("Initializing RAG Orchestrator...")

    try:
        # Create orchestrator with service URLs
        orchestrator = RAGOrchestrator(
            query_analysis_url=settings.QUERY_ANALYSIS_URL,
            document_retrieval_url=settings.DOCUMENT_RETRIEVAL_URL,
            document_ranking_url=settings.DOCUMENT_RANKING_URL,
            latex_parser_url=settings.LATEX_PARSER_URL,
            llm_generation_url=settings.LLM_GENERATION_URL,
            response_formatter_url=settings.RESPONSE_FORMATTER_URL
        )

        # Store orchestrator in app state for dependency injection
        app.state.orchestrator = orchestrator

        # Health check all services
        logger.info("Checking health of all services...")
        service_health = await orchestrator.health_check_all_services()

        for service_name, is_healthy in service_health.items():
            status = "[OK]" if is_healthy else "[FAIL]"
            logger.info(f"  {status} {service_name}: {'healthy' if is_healthy else 'unhealthy'}")

        healthy_services = sum(1 for h in service_health.values() if h)
        total_services = len(service_health)

        if healthy_services == total_services:
            logger.info(f"All {total_services} services are healthy - Ready to process queries!")
        elif healthy_services > 0:
            logger.warning(
                f"[WARN] {healthy_services}/{total_services} services are healthy - "
                f"System may have degraded performance"
            )
        else:
            logger.error(
                f"[FAIL] No services are healthy - System cannot process queries. "
                f"Please check service configurations."
            )

        logger.info(f"API Gateway initialized successfully")
        logger.info(f"API documentation available at http://localhost:8000/docs")

    except Exception as e:
        logger.error(f"Failed to initialize API Gateway: {e}", exc_info=True)
        raise

    yield

    # Cleanup
    logger.info("Shutting down API Gateway")

    # Log service stop
    if audit_logger:
        audit_logger.log_event(AuditEventType.SERVICE_STOP, details={"reason": "shutdown"})

    try:
        if hasattr(app.state, "orchestrator") and app.state.orchestrator:
            await app.state.orchestrator.close()
            logger.info("Orchestrator closed successfully")

        if settings.ENABLE_RATE_LIMITING and rate_limiter:
            await rate_limiter.close()
            logger.info("Rate limiter closed successfully")

    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


app = FastAPI(
    title="RAG System API Gateway",
    description="""
    API Gateway and Orchestrator for Retrieval-Augmented Generation (RAG) System

    This gateway coordinates multiple microservices to provide intelligent question-answering:
    - Query Analysis: Understands user intent and extracts keywords
    - Document Retrieval: Finds relevant documents using vector search
    - Document Ranking: Reranks documents for better relevance
    - LaTeX Parsing: Handles mathematical formulas
    - LLM Generation: Generates answers using retrieved context
    - Response Formatting: Formats responses with citations

    ## Usage

    Send a POST request to `/api/v1/query` with your question:

    ```json
    {
        "query": "What is the Pythagorean theorem?",
        "output_format": "markdown"
    }
    ```

    The system will return a formatted response with citations.
    """,
    version="0.1.0",
    lifespan=lifespan,
)

# Security headers middleware (add first to ensure headers are set)
if settings.ENABLE_SECURITY_HEADERS:
    app.add_middleware(
        SecurityHeadersMiddleware,
        enable_hsts=settings.ENABLE_HSTS
    )
    logger.info("Security headers middleware enabled")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ALLOWED_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    max_age=settings.CORS_MAX_AGE
)

# Rate limiting middleware (optional, based on settings)
if settings.ENABLE_RATE_LIMITING:
    logger.info("Rate limiting enabled")
    # Rate limiter will be initialized in lifespan
    # We need to create it here for middleware, but it will be connected in lifespan
    rate_limiter_for_middleware = RedisRateLimiter(
        redis_url=settings.REDIS_URL,
        requests_per_minute=settings.RATE_LIMIT_REQUESTS_PER_MINUTE,
        burst_size=settings.RATE_LIMIT_BURST,
        enabled=settings.ENABLE_RATE_LIMITING
    )
    app.add_middleware(RateLimitMiddleware, rate_limiter=rate_limiter_for_middleware)
else:
    logger.info("Rate limiting disabled")

# Authentication middleware (optional, based on settings)
if settings.ENABLE_AUTHENTICATION:
    logger.info("Authentication enabled - JWT middleware active")
    jwt_handler = JWTHandler(
        secret_key=settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
        access_token_expire_minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES,
        refresh_token_expire_days=settings.REFRESH_TOKEN_EXPIRE_DAYS
    )
    app.add_middleware(AuthenticationMiddleware, jwt_handler=jwt_handler)
else:
    logger.info("Authentication disabled - All endpoints are public")

# Include API routes
register_routes(app)

# Add Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.get("/")
async def root():
    """
    Root endpoint returning service information.

    :return: Service info dictionary with name, version, status, and endpoints.
    """
    return {
        "service": "RAG System API Gateway",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs",
        "api_base": "/api/v1"
    }


@app.get("/health")
async def health():
    """
    Root health check endpoint for Docker/Kubernetes.

    :return: Health status dictionary.
    """
    return {"status": "healthy", "service": "api-gateway"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.LOG_LEVEL.lower()
    )
