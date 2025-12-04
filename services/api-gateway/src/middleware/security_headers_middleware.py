"""
Security Headers Middleware
Adds security-related HTTP headers to all responses
"""
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add security headers to all responses

    Headers added:
    - X-Content-Type-Options: Prevent MIME type sniffing
    - X-Frame-Options: Prevent clickjacking
    - X-XSS-Protection: Enable XSS filter
    - Strict-Transport-Security: Enforce HTTPS
    - Content-Security-Policy: Control resource loading
    - Referrer-Policy: Control referrer information
    - Permissions-Policy: Control browser features
    """

    def __init__(self, app, enable_hsts: bool = False):
        """
        Initialize security headers middleware

        Args:
            app: FastAPI application
            enable_hsts: Enable HTTP Strict Transport Security (only for HTTPS)
        """
        super().__init__(app)
        self.enable_hsts = enable_hsts

    async def dispatch(self, request: Request, call_next):
        """
        Add security headers to response

        Args:
            request: Incoming request
            call_next: Next middleware/handler

        Returns:
            Response with security headers
        """
        response = await call_next(request)

        # Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"

        # Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"

        # Enable XSS filter (legacy browsers)
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # HTTP Strict Transport Security (only if HTTPS)
        if self.enable_hsts:
            # max-age=31536000 = 1 year
            # includeSubDomains = apply to all subdomains
            # preload = allow browser preload lists
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains; preload"
            )

        # Content Security Policy
        # Restrict resource loading to same origin
        csp_directives = [
            "default-src 'self'",
            "script-src 'self' 'unsafe-inline'",  # Allow inline scripts for Swagger UI
            "style-src 'self' 'unsafe-inline'",   # Allow inline styles for Swagger UI
            "img-src 'self' data: https:",         # Allow data URIs and HTTPS images
            "font-src 'self' data:",               # Allow data URIs for fonts
            "connect-src 'self'",                  # API calls to same origin
            "frame-ancestors 'none'",              # Prevent framing (similar to X-Frame-Options)
            "base-uri 'self'",                     # Restrict base tag
            "form-action 'self'"                   # Restrict form submissions
        ]
        response.headers["Content-Security-Policy"] = "; ".join(csp_directives)

        # Referrer Policy
        # Don't send referrer information to external sites
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Permissions Policy (formerly Feature-Policy)
        # Disable unnecessary browser features
        permissions_directives = [
            "geolocation=()",        # Disable geolocation
            "microphone=()",         # Disable microphone
            "camera=()",             # Disable camera
            "payment=()",            # Disable payment API
            "usb=()",                # Disable USB
            "magnetometer=()",       # Disable magnetometer
            "gyroscope=()",          # Disable gyroscope
            "accelerometer=()"       # Disable accelerometer
        ]
        response.headers["Permissions-Policy"] = ", ".join(permissions_directives)

        # Remove server header (don't leak server information)
        if "Server" in response.headers:
            del response.headers["Server"]

        # Remove X-Powered-By header if present
        if "X-Powered-By" in response.headers:
            del response.headers["X-Powered-By"]

        return response
