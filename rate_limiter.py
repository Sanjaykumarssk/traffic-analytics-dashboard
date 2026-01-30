"""
Rate limiting middleware for FastAPI.
Implements token bucket algorithm for API rate limiting.
"""

import os
import time
from collections import defaultdict
from typing import Optional
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import logging

logger = logging.getLogger(__name__)


class TokenBucket:
    """Token bucket implementation for rate limiting."""
    
    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize token bucket.
        
        Args:
            capacity: Maximum number of tokens
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
    
    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from the bucket.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were consumed, False otherwise
        """
        now = time.time()
        # Refill tokens based on time passed
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False


class RateLimiter:
    """Rate limiter with per-IP token buckets."""
    
    def __init__(self):
        self.buckets = defaultdict(lambda: TokenBucket(
            capacity=int(os.getenv("RATE_LIMIT_CAPACITY", "100")),
            refill_rate=float(os.getenv("RATE_LIMIT_REFILL", "10"))
        ))
        self.cleanup_interval = 300  # Clean up old buckets every 5 minutes
        self.last_cleanup = time.time()
    
    def get_client_identifier(self, request: Request) -> str:
        """
        Get client identifier for rate limiting.
        Uses IP address or user ID if authenticated.
        
        Args:
            request: FastAPI request object
            
        Returns:
            Client identifier string
        """
        # Try to get user ID from request state (if authenticated)
        if hasattr(request.state, "user_id"):
            return f"user:{request.state.user_id}"
        
        # Fall back to IP address
        client_ip = request.client.host if request.client else "unknown"
        # Handle forwarded IPs (from proxies/load balancers)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        
        return f"ip:{client_ip}"
    
    def is_allowed(self, request: Request, endpoint: str = None) -> tuple[bool, dict]:
        """
        Check if request is allowed based on rate limits.
        
        Args:
            request: FastAPI request object
            endpoint: Optional endpoint name for per-endpoint limits
            
        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        client_id = self.get_client_identifier(request)
        
        # Different limits for different endpoints
        endpoint_limits = {
            "/autocomplete": {"capacity": 50, "refill": 5},
            "/analyze-route": {"capacity": 20, "refill": 2},
            "/api/analytics": {"capacity": 30, "refill": 3},
            "default": {"capacity": 100, "refill": 10}
        }
        
        # Get limits for this endpoint
        limits = endpoint_limits.get(endpoint, endpoint_limits["default"])
        bucket_key = f"{client_id}:{endpoint or 'default'}"
        
        # Get or create bucket
        if bucket_key not in self.buckets:
            self.buckets[bucket_key] = TokenBucket(
                capacity=limits["capacity"],
                refill_rate=limits["refill"]
            )
        
        bucket = self.buckets[bucket_key]
        tokens_needed = 1
        
        # Check if request is allowed
        if bucket.consume(tokens_needed):
            remaining = int(bucket.tokens)
            return True, {
                "remaining": remaining,
                "limit": bucket.capacity,
                "reset_in": int((bucket.capacity - remaining) / bucket.refill_rate) if bucket.refill_rate > 0 else 0
            }
        else:
            remaining = int(bucket.tokens)
            return False, {
                "remaining": 0,
                "limit": bucket.capacity,
                "reset_in": int((1 - remaining) / bucket.refill_rate) if bucket.refill_rate > 0 else 60
            }
    
    def cleanup_old_buckets(self):
        """Remove old buckets to prevent memory leaks."""
        now = time.time()
        if now - self.last_cleanup > self.cleanup_interval:
            # Remove buckets that haven't been used in cleanup_interval
            # (In a production system, you'd track last access time)
            self.last_cleanup = now
            # Simple cleanup: keep only recent buckets
            if len(self.buckets) > 1000:
                # Keep only the most recent 500 buckets
                keys_to_remove = list(self.buckets.keys())[:-500]
                for key in keys_to_remove:
                    del self.buckets[key]


# Global rate limiter instance
rate_limiter = RateLimiter()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting."""
    
    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting."""
        # Skip rate limiting for health checks and static files
        if request.url.path in ["/health", "/docs", "/openapi.json", "/favicon.ico"]:
            return await call_next(request)
        
        # Clean up old buckets periodically
        rate_limiter.cleanup_old_buckets()
        
        # Get endpoint name
        endpoint = request.url.path
        
        # Check rate limit
        is_allowed, rate_info = rate_limiter.is_allowed(request, endpoint)
        
        # Add rate limit headers
        response_headers = {
            "X-RateLimit-Limit": str(rate_info["limit"]),
            "X-RateLimit-Remaining": str(rate_info["remaining"]),
            "X-RateLimit-Reset": str(rate_info["reset_in"])
        }
        
        if not is_allowed:
            logger.warning(
                f"Rate limit exceeded for {rate_limiter.get_client_identifier(request)} "
                f"on endpoint {endpoint}"
            )
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests. Please try again in {rate_info['reset_in']} seconds.",
                    "retry_after": rate_info["reset_in"]
                },
                headers=response_headers
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to response
        for key, value in response_headers.items():
            response.headers[key] = value
        
        return response

