"""
API module - Route registration for API Gateway.
"""
from fastapi import FastAPI

from src.api import routes, proxy_routes, auth_routes, api_key_routes


def register_routes(app: FastAPI, prefix: str = "/api/v1") -> None:
    """
    Register all API routes with the FastAPI application.

    :param app: FastAPI application instance.
    :param prefix: URL prefix for all routes (default: /api/v1).
    """
    # Main RAG pipeline routes (query, health)
    app.include_router(routes.router, prefix=prefix)

    # Proxy routes to downstream services (collections, documents, etc.)
    app.include_router(proxy_routes.router, prefix=prefix)

    # Authentication routes (/api/v1/auth/*)
    app.include_router(auth_routes.router, prefix=prefix)

    # Admin API key management routes (/api/v1/admin/api-keys/*)
    app.include_router(api_key_routes.router, prefix=prefix)
