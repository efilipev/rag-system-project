"""
Base HTTP client with retry logic, circuit breaker, and timeout handling
"""
import logging
from typing import Optional, Dict, Any
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
from pybreaker import CircuitBreaker, CircuitBreakerError

logger = logging.getLogger(__name__)


class BaseHTTPClient:
    """
    Base HTTP client for service-to-service communication

    Features:
    - Connection pooling
    - Automatic retries with exponential backoff
    - Circuit breaker pattern
    - Timeout handling
    - Request/response logging
    - Correlation ID propagation
    """

    def __init__(
        self,
        base_url: str,
        service_name: str,
        timeout: float = 30.0,
        max_retries: int = 3,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: int = 60
    ):
        """
        Initialize HTTP client

        Args:
            base_url: Base URL of the service
            service_name: Name of the service (for logging)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            circuit_breaker_threshold: Number of failures before opening circuit
            circuit_breaker_timeout: Seconds to wait before attempting to close circuit
        """
        self.base_url = base_url.rstrip('/')
        self.service_name = service_name
        self.timeout = timeout
        self.max_retries = max_retries

        # HTTP client with connection pooling
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            limits=httpx.Limits(
                max_keepalive_connections=10,
                max_connections=20
            ),
            follow_redirects=True
        )

        # Circuit breaker configuration
        self.circuit_breaker = CircuitBreaker(
            fail_max=circuit_breaker_threshold,
            reset_timeout=circuit_breaker_timeout,
            name=f"{service_name}_circuit_breaker"
        )

        logger.info(
            f"Initialized HTTP client for {service_name} at {base_url} "
            f"(timeout={timeout}s, retries={max_retries})"
        )

    def _prepare_headers(self, correlation_id: Optional[str] = None) -> Dict[str, str]:
        """
        Prepare request headers with correlation ID

        Args:
            correlation_id: Request correlation ID for distributed tracing

        Returns:
            Headers dictionary
        """
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        if correlation_id:
            headers["X-Correlation-ID"] = correlation_id

        return headers

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        correlation_id: Optional[str] = None,
        **kwargs
    ) -> httpx.Response:
        """
        Make HTTP request with retry logic (internal method)

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            correlation_id: Request correlation ID
            **kwargs: Additional arguments for httpx request

        Returns:
            HTTP response

        Raises:
            httpx.HTTPError: If request fails after retries
            CircuitBreakerError: If circuit is open
        """
        headers = self._prepare_headers(correlation_id)

        # Merge with any additional headers
        if "headers" in kwargs:
            headers.update(kwargs["headers"])
        kwargs["headers"] = headers

        logger.debug(
            f"{self.service_name}: {method} {endpoint} "
            f"(correlation_id={correlation_id})"
        )

        # Make request through circuit breaker
        try:
            response = await self.circuit_breaker.call_async(
                self.client.request,
                method,
                endpoint,
                **kwargs
            )

            response.raise_for_status()

            logger.debug(
                f"{self.service_name}: {method} {endpoint} -> {response.status_code}"
            )

            return response

        except CircuitBreakerError as e:
            logger.error(
                f"{self.service_name}: Circuit breaker is OPEN. "
                f"Service appears to be down."
            )
            raise

        except httpx.HTTPStatusError as e:
            logger.error(
                f"{self.service_name}: HTTP {e.response.status_code} error "
                f"for {method} {endpoint}: {e.response.text}"
            )
            raise

        except httpx.TimeoutException as e:
            logger.error(
                f"{self.service_name}: Timeout for {method} {endpoint} "
                f"(timeout={self.timeout}s)"
            )
            raise

        except Exception as e:
            logger.error(
                f"{self.service_name}: Unexpected error for {method} {endpoint}: {e}",
                exc_info=True
            )
            raise

    async def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make GET request

        Args:
            endpoint: API endpoint
            params: Query parameters
            correlation_id: Request correlation ID
            **kwargs: Additional arguments

        Returns:
            Response JSON as dictionary
        """
        response = await self._make_request(
            "GET",
            endpoint,
            correlation_id=correlation_id,
            params=params,
            **kwargs
        )
        return response.json()

    async def post(
        self,
        endpoint: str,
        json: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make POST request

        Args:
            endpoint: API endpoint
            json: JSON body
            correlation_id: Request correlation ID
            **kwargs: Additional arguments

        Returns:
            Response JSON as dictionary
        """
        response = await self._make_request(
            "POST",
            endpoint,
            correlation_id=correlation_id,
            json=json,
            **kwargs
        )
        return response.json()

    async def put(
        self,
        endpoint: str,
        json: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make PUT request

        Args:
            endpoint: API endpoint
            json: JSON body
            correlation_id: Request correlation ID
            **kwargs: Additional arguments

        Returns:
            Response JSON as dictionary
        """
        response = await self._make_request(
            "PUT",
            endpoint,
            correlation_id=correlation_id,
            json=json,
            **kwargs
        )
        return response.json()

    async def delete(
        self,
        endpoint: str,
        correlation_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make DELETE request

        Args:
            endpoint: API endpoint
            correlation_id: Request correlation ID
            **kwargs: Additional arguments

        Returns:
            Response JSON as dictionary
        """
        response = await self._make_request(
            "DELETE",
            endpoint,
            correlation_id=correlation_id,
            **kwargs
        )
        return response.json()

    async def health_check(self) -> bool:
        """
        Check if service is healthy

        Returns:
            True if service is healthy, False otherwise
        """
        try:
            response = await self.get("/health")
            return response.get("status") in ["healthy", "ok"]
        except Exception as e:
            logger.warning(f"{self.service_name}: Health check failed: {e}")
            return False

    async def close(self):
        """Close HTTP client and cleanup resources"""
        await self.client.aclose()
        logger.info(f"{self.service_name}: HTTP client closed")

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
