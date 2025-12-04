"""
Authentication Middleware
Provides optional JWT authentication for API Gateway
"""
import logging
from typing import Optional
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware

from src.core.config import settings
from shared.auth.jwt_handler import JWTHandler
from shared.auth.api_key_manager import APIKeyManager

logger = logging.getLogger(__name__)

# Global API key manager instance
_api_key_manager: Optional[APIKeyManager] = None


def get_api_key_manager() -> Optional[APIKeyManager]:
    """Get API key manager instance"""
    return _api_key_manager


def set_api_key_manager(manager: APIKeyManager):
    """Set global API key manager instance"""
    global _api_key_manager
    _api_key_manager = manager


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Middleware for JWT token authentication
    Only enforces auth when ENABLE_AUTHENTICATION is True
    """

    def __init__(self, app, jwt_handler: JWTHandler):
        super().__init__(app)
        self.jwt_handler = jwt_handler
        self.enabled = settings.ENABLE_AUTHENTICATION

        # Public endpoints that don't require authentication
        self.public_endpoints = [
            "/",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/api/v1/health",
            "/api/v1/auth/login",
            "/api/v1/auth/register",
            "/api/v1/auth/refresh"
        ]

    async def dispatch(self, request: Request, call_next):
        """
        Process request and verify authentication if required

        Args:
            request: Incoming HTTP request
            call_next: Next middleware/route handler

        Returns:
            Response from next handler

        Raises:
            HTTPException: If authentication is required but invalid
        """
        # Skip authentication if disabled
        if not self.enabled:
            return await call_next(request)

        # Skip authentication for public endpoints
        if request.url.path in self.public_endpoints:
            return await call_next(request)

        # Extract token from Authorization header
        auth_header = request.headers.get("Authorization")
        api_key_header = request.headers.get("X-API-Key")

        # Try JWT authentication first
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]

            try:
                payload = self.jwt_handler.verify_access_token(token)

                if payload:
                    # Add user info to request state
                    request.state.user = payload
                    request.state.auth_type = "jwt"
                    logger.debug(f"Authenticated user: {payload.get('user_id')}")
                    return await call_next(request)

            except Exception as e:
                logger.warning(f"JWT verification failed: {e}")

        # Try API key authentication
        if api_key_header:
            api_key_mgr = get_api_key_manager()

            if api_key_mgr:
                api_key_info = api_key_mgr.verify_api_key(api_key_header)

                if api_key_info:
                    request.state.user = {
                        "service_name": api_key_info.service_name,
                        "auth_type": "api_key"
                    }
                    request.state.auth_type = "api_key"
                    logger.debug(f"Authenticated service via API key: {api_key_info.service_name}")
                    return await call_next(request)

        # No valid authentication provided
        logger.warning(f"Unauthorized access attempt to {request.url.path}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_current_user(request: Request) -> Optional[dict]:
    """
    Get current authenticated user from request state

    Args:
        request: FastAPI request object

    Returns:
        User payload if authenticated, None otherwise
    """
    return getattr(request.state, "user", None)


def require_authentication(request: Request) -> dict:
    """
    Require authentication for endpoint

    Args:
        request: FastAPI request object

    Returns:
        User payload

    Raises:
        HTTPException: If not authenticated
    """
    user = get_current_user(request)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )

    return user
