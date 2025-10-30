"""Redis-based caching manager for API responses."""
import json
import hashlib
from typing import Any, Optional, Callable
import redis
from functools import wraps
from config import config

class CacheManager:
    """Manages Redis caching for API calls and expensive operations."""
    
    def __init__(self):
        """Initialize Redis connection."""
        try:
            self.redis_client = redis.Redis(
                host=config.REDIS_HOST,
                port=config.REDIS_PORT,
                db=config.REDIS_DB,
                decode_responses=True
            )
            self.redis_client.ping()
            self.enabled = True
        except (redis.ConnectionError, redis.TimeoutError):
            print("Warning: Redis not available. Caching disabled.")
            self.redis_client = None
            self.enabled = False
    
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from function arguments."""
        key_data = f"{prefix}:{str(args)}:{str(sorted(kwargs.items()))}"
        return f"movie_rec:{hashlib.md5(key_data.encode()).hexdigest()}"
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve value from cache."""
        if not self.enabled:
            return None
        
        try:
            value = self.redis_client.get(key)
            return json.loads(value) if value else None
        except Exception as e:
            print(f"Cache get error: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Store value in cache with TTL."""
        if not self.enabled:
            return False
        
        try:
            ttl = ttl or config.CACHE_TTL
            self.redis_client.setex(key, ttl, json.dumps(value))
            return True
        except Exception as e:
            print(f"Cache set error: {e}")
            return False
    
    def delete(self, pattern: str) -> int:
        """Delete keys matching pattern."""
        if not self.enabled:
            return 0
        
        try:
            keys = self.redis_client.keys(pattern)
            return self.redis_client.delete(*keys) if keys else 0
        except Exception as e:
            print(f"Cache delete error: {e}")
            return 0
    
    def cached(self, prefix: str, ttl: int = None):
        """Decorator for caching function results."""
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                cache_key = self._generate_key(prefix, *args, **kwargs)
                
                # Try to get from cache
                cached_value = self.get(cache_key)
                if cached_value is not None:
                    return cached_value
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl)
                return result
            
            return wrapper
        return decorator

# Global cache manager instance
cache_manager = CacheManager()
