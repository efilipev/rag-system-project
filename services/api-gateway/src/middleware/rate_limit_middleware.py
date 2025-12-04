"""
Rate Limiting Middleware
Implements per-user and per-IP rate limiting using Redis
"""
import logging
from typing import Optional

from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware

from src.core.config import settings
from shared.rate_limiting.redis_rate_limiter import RedisRateLimiter

logger = logging.getLogger(__name__)

# Global rate limiter instance
_rate_limiter: Optional[RedisRateLimiter] = None


def get_rate_limiter() -> Optional[RedisRateLimiter]:
    """Get rate limiter instance"""
    return _rate_limiter


def set_rate_limiter(limiter: RedisRateLimiter):
    """Set global rate limiter instance"""
    global _rate_limiter
    _rate_limiter = limiter


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware
    Limits requests per user/IP address
    """

    def __init__(self, app, rate_limiter: RedisRateLimiter):
        super().__init__(app)
        self.rate_limiter = rate_limiter
        self.enabled = settings.ENABLE_RATE_LIMITING
        self._connected = False

        # Endpoints exempt from rate limiting
        self.exempt_endpoints = [
            "/",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/api/v1/health"
        ]

    async def _ensure_connected(self):
        """Ensure rate limiter is connected (lazy connection)"""
        if not self._connected and self.enabled and self.rate_limiter:
            try:
                await self.rate_limiter.connect()
                self._connected = True
                logger.info("Rate limiter connected (lazy initialization)")
            except Exception as e:
                logger.error(f"Failed to connect rate limiter: {e}")
                self.enabled = False

    def _get_rate_limit_key(self, request: Request) -> str:
        """
        Get unique key for rate limiting

        Uses user ID if authenticated, otherwise uses IP address

        Args:
            request: Incoming request

        Returns:
            Unique key for rate limiting
        """
        # Try to get user from request state (if authenticated)
        user = getattr(request.state, "user", None)

        if user and isinstance(user, dict):
            user_id = user.get("user_id") or user.get("service_name")
            if user_id:
                return f"user:{user_id}"

        # Fall back to IP address
        client_ip = request.client.host if request.client else "unknown"

        # Check for forwarded IP (when behind proxy)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()

        return f"ip:{client_ip}"

    async def dispatch(self, request: Request, call_next):
        """
        Process request and check rate limits

        Args:
            request: Incoming HTTP request
            call_next: Next middleware/route handler

        Returns:
            Response from next handler

        Raises:
            HTTPException: If rate limit exceeded
        """
        # Skip if disabled
        if not self.enabled or self.rate_limiter is None:
            return await call_next(request)

        # Skip exempt endpoints
        if request.url.path in self.exempt_endpoints:
            return await call_next(request)

        # Ensure connection (lazy initialization)
        await self._ensure_connected()

        # Get rate limit key
        rate_limit_key = self._get_rate_limit_key(request)

        # Check rate limit
        is_allowed, metadata = await self.rate_limiter.is_allowed(rate_limit_key)

        # Add rate limit headers to response
        async def add_rate_limit_headers(response):
            """Add rate limit headers to response"""
            response.headers["X-RateLimit-Limit"] = str(metadata.get("limit", "N/A"))
            response.headers["X-RateLimit-Remaining"] = str(metadata.get("remaining", "N/A"))
            response.headers["X-RateLimit-Reset"] = str(metadata.get("reset_time", "N/A"))

            if not is_allowed:
                response.headers["Retry-After"] = str(metadata.get("retry_after", 60))

            return response

        # Check if allowed
        if not is_allowed:
            logger.warning(
                f"Rate limit exceeded for {rate_limit_key} - "
                f"Retry after {metadata.get('retry_after', 60)}s"
            )

            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "Rate limit exceeded",
                    "retry_after": metadata.get("retry_after", 60),
                    "limit": metadata.get("limit"),
                    "reset_time": metadata.get("reset_time")
                },
                headers={
                    "Retry-After": str(metadata.get("retry_after", 60)),
                    "X-RateLimit-Limit": str(metadata.get("limit", "N/A")),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(metadata.get("reset_time", "N/A"))
                }
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        response = await add_rate_limit_headers(response)

        return response
