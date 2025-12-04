"""
Redis-based Rate Limiter
Implements token bucket and fixed window rate limiting
"""
import logging
import time
from typing import Optional, Tuple
from datetime import timedelta

try:
    import redis.asyncio as aioredis
except ImportError:
    aioredis = None

logger = logging.getLogger(__name__)


class RedisRateLimiter:
    """
    Redis-based rate limiter using token bucket algorithm

    Features:
    - Per-user rate limiting
    - Per-endpoint rate limiting
    - Burst handling
    - Distributed rate limiting across multiple instances
    """

    def __init__(
        self,
        redis_url: str,
        requests_per_minute: int = 60,
        burst_size: int = 10,
        enabled: bool = True
    ):
        """
        Initialize rate limiter

        Args:
            redis_url: Redis connection URL
            requests_per_minute: Number of requests allowed per minute
            burst_size: Number of burst requests allowed
            enabled: Whether rate limiting is enabled
        """
        self.redis_url = redis_url
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.enabled = enabled
        self.redis_client: Optional[aioredis.Redis] = None

        if not enabled:
            logger.info("Rate limiting is disabled")
            return

        if aioredis is None:
            logger.warning(
                "redis package not installed - rate limiting will be disabled. "
                "Install with: pip install redis"
            )
            self.enabled = False

    async def connect(self):
        """Connect to Redis"""
        if not self.enabled or aioredis is None:
            return

        try:
            self.redis_client = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info(f"Connected to Redis at {self.redis_url}")

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            logger.warning("Rate limiting will be disabled")
            self.enabled = False

    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Redis connection closed")

    async def is_allowed(
        self,
        key: str,
        requests_per_minute: Optional[int] = None,
        burst_size: Optional[int] = None
    ) -> Tuple[bool, dict]:
        """
        Check if request is allowed under rate limit

        Args:
            key: Unique key for rate limiting (e.g., user_id, ip_address)
            requests_per_minute: Override default requests per minute
            burst_size: Override default burst size

        Returns:
            Tuple of (is_allowed, metadata)
            metadata contains: remaining, reset_time, retry_after
        """
        # If disabled, always allow
        if not self.enabled or self.redis_client is None:
            return True, {
                "remaining": 9999,
                "reset_time": int(time.time() + 60),
                "retry_after": 0
            }

        rpm = requests_per_minute or self.requests_per_minute
        burst = burst_size or self.burst_size

        try:
            # Use token bucket algorithm with Redis
            redis_key = f"rate_limit:{key}"
            current_time = time.time()
            window_start = int(current_time / 60) * 60  # Start of current minute

            # Get current count and timestamp
            pipe = self.redis_client.pipeline()
            pipe.get(f"{redis_key}:count")
            pipe.get(f"{redis_key}:window")
            results = await pipe.execute()

            current_count = int(results[0]) if results[0] else 0
            stored_window = int(results[1]) if results[1] else 0

            # Reset if new window
            if stored_window != window_start:
                current_count = 0

            # Check if allowed
            max_requests = rpm + burst
            is_allowed = current_count < max_requests

            if is_allowed:
                # Increment counter
                pipe = self.redis_client.pipeline()
                pipe.incr(f"{redis_key}:count")
                pipe.set(f"{redis_key}:window", window_start)
                pipe.expire(f"{redis_key}:count", 120)  # Expire after 2 minutes
                pipe.expire(f"{redis_key}:window", 120)
                await pipe.execute()

                current_count += 1

            # Calculate metadata
            remaining = max(0, max_requests - current_count)
            reset_time = window_start + 60  # End of current window
            retry_after = max(0, int(reset_time - current_time)) if not is_allowed else 0

            metadata = {
                "remaining": remaining,
                "reset_time": reset_time,
                "retry_after": retry_after,
                "limit": max_requests
            }

            if not is_allowed:
                logger.warning(f"Rate limit exceeded for key: {key}")

            return is_allowed, metadata

        except Exception as e:
            logger.error(f"Rate limiting check failed: {e}")
            # On error, allow request (fail open)
            return True, {
                "remaining": 9999,
                "reset_time": int(time.time() + 60),
                "retry_after": 0,
                "error": str(e)
            }

    async def reset_limit(self, key: str):
        """
        Reset rate limit for a key

        Args:
            key: Key to reset
        """
        if not self.enabled or self.redis_client is None:
            return

        try:
            redis_key = f"rate_limit:{key}"
            await self.redis_client.delete(f"{redis_key}:count", f"{redis_key}:window")
            logger.info(f"Reset rate limit for key: {key}")

        except Exception as e:
            logger.error(f"Failed to reset rate limit: {e}")

    async def get_remaining(self, key: str) -> int:
        """
        Get remaining requests for a key

        Args:
            key: Key to check

        Returns:
            Number of remaining requests
        """
        if not self.enabled or self.redis_client is None:
            return 9999

        try:
            redis_key = f"rate_limit:{key}"
            current_time = time.time()
            window_start = int(current_time / 60) * 60

            pipe = self.redis_client.pipeline()
            pipe.get(f"{redis_key}:count")
            pipe.get(f"{redis_key}:window")
            results = await pipe.execute()

            current_count = int(results[0]) if results[0] else 0
            stored_window = int(results[1]) if results[1] else 0

            # Reset if new window
            if stored_window != window_start:
                current_count = 0

            max_requests = self.requests_per_minute + self.burst_size
            remaining = max(0, max_requests - current_count)

            return remaining

        except Exception as e:
            logger.error(f"Failed to get remaining requests: {e}")
            return 9999


class FixedWindowRateLimiter:
    """
    Simple fixed window rate limiter
    Less accurate but simpler than token bucket
    """

    def __init__(
        self,
        redis_url: str,
        requests_per_window: int = 100,
        window_seconds: int = 60,
        enabled: bool = True
    ):
        """
        Initialize fixed window rate limiter

        Args:
            redis_url: Redis connection URL
            requests_per_window: Requests allowed per window
            window_seconds: Window size in seconds
            enabled: Whether rate limiting is enabled
        """
        self.redis_url = redis_url
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self.enabled = enabled
        self.redis_client: Optional[aioredis.Redis] = None

        if not enabled:
            logger.info("Fixed window rate limiting is disabled")
            return

        if aioredis is None:
            logger.warning("redis package not installed")
            self.enabled = False

    async def connect(self):
        """Connect to Redis"""
        if not self.enabled or aioredis is None:
            return

        try:
            self.redis_client = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info(f"Fixed window limiter connected to Redis")

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.enabled = False

    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()

    async def is_allowed(self, key: str) -> Tuple[bool, dict]:
        """
        Check if request is allowed

        Args:
            key: Unique key for rate limiting

        Returns:
            Tuple of (is_allowed, metadata)
        """
        if not self.enabled or self.redis_client is None:
            return True, {"remaining": 9999}

        try:
            redis_key = f"rate_limit_fw:{key}"
            current_window = int(time.time() / self.window_seconds)

            # Use Lua script for atomic increment and expiry
            lua_script = """
            local key = KEYS[1]
            local window = ARGV[1]
            local limit = tonumber(ARGV[2])
            local ttl = tonumber(ARGV[3])

            local current = redis.call('GET', key .. ':' .. window)
            current = current and tonumber(current) or 0

            if current < limit then
                redis.call('INCR', key .. ':' .. window)
                redis.call('EXPIRE', key .. ':' .. window, ttl)
                return {1, limit - current - 1}
            else
                return {0, 0}
            end
            """

            result = await self.redis_client.eval(
                lua_script,
                1,
                redis_key,
                current_window,
                self.requests_per_window,
                self.window_seconds * 2  # TTL = 2x window size
            )

            is_allowed = bool(result[0])
            remaining = int(result[1])

            return is_allowed, {"remaining": remaining}

        except Exception as e:
            logger.error(f"Rate limiting check failed: {e}")
            return True, {"remaining": 9999, "error": str(e)}
