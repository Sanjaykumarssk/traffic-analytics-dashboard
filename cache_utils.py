"""
Caching utilities for performance optimization.
"""

from functools import wraps
from typing import Callable, Any
import hashlib
import json
from datetime import datetime, timedelta
from cachetools import TTLCache
import pickle

# Route cache (TTL: 5 minutes)
route_cache = TTLCache(maxsize=1000, ttl=300)

# Analysis cache (TTL: 10 minutes)
analysis_cache = TTLCache(maxsize=500, ttl=600)


def cache_key(*args, **kwargs) -> str:
    """Generate cache key from function arguments."""
    key_data = {
        'args': args,
        'kwargs': sorted(kwargs.items())
    }
    key_str = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.md5(key_str.encode()).hexdigest()


def cached(cache_dict: dict = route_cache):
    """Decorator for caching function results."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = f"{func.__name__}:{cache_key(*args, **kwargs)}"
            
            # Check cache
            if key in cache_dict:
                return cache_dict[key]
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Store in cache
            cache_dict[key] = result
            
            return result
        return wrapper
    return decorator


def clear_cache(cache_dict: dict = route_cache, pattern: str = None):
    """Clear cache entries."""
    if pattern:
        keys_to_remove = [k for k in cache_dict.keys() if pattern in k]
        for key in keys_to_remove:
            cache_dict.pop(key, None)
    else:
        cache_dict.clear()


def get_cache_stats(cache_dict: dict = route_cache) -> dict:
    """Get cache statistics."""
    return {
        "size": len(cache_dict),
        "maxsize": cache_dict.maxsize,
        "ttl": cache_dict.ttl
    }
