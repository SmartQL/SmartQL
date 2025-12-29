"""
Cache backends for QueryWise.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional
import hashlib
import json
import time


class CacheBackend(ABC):
    """Abstract base class for cache backends."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[dict[str, Any]]:
        """Get a value from the cache."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: dict[str, Any], ttl: Optional[int] = None) -> None:
        """Set a value in the cache."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete a value from the cache."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all cached values."""
        pass


class MemoryCache(CacheBackend):
    """In-memory cache backend."""
    
    def __init__(self, config: dict[str, Any]):
        self.ttl = config.get("ttl_seconds", 3600)
        self._cache: dict[str, tuple[dict, float]] = {}  # (value, expiry_time)
    
    def get(self, key: str) -> Optional[dict[str, Any]]:
        if key not in self._cache:
            return None
        
        value, expiry = self._cache[key]
        if time.time() > expiry:
            del self._cache[key]
            return None
        
        return value
    
    def set(self, key: str, value: dict[str, Any], ttl: Optional[int] = None) -> None:
        expiry = time.time() + (ttl or self.ttl)
        self._cache[key] = (value, expiry)
    
    def delete(self, key: str) -> None:
        self._cache.pop(key, None)
    
    def clear(self) -> None:
        self._cache.clear()
    
    def cleanup(self) -> int:
        """Remove expired entries. Returns number of entries removed."""
        now = time.time()
        expired = [k for k, (_, expiry) in self._cache.items() if now > expiry]
        for key in expired:
            del self._cache[key]
        return len(expired)


class FileCache(CacheBackend):
    """File-based cache backend."""
    
    def __init__(self, config: dict[str, Any]):
        import os
        
        self.ttl = config.get("ttl_seconds", 3600)
        self.directory = config.get("directory", "/tmp/smartql_cache")
        
        os.makedirs(self.directory, exist_ok=True)
    
    def _key_to_path(self, key: str) -> str:
        import os
        safe_key = hashlib.sha256(key.encode()).hexdigest()
        return os.path.join(self.directory, f"{safe_key}.json")
    
    def get(self, key: str) -> Optional[dict[str, Any]]:
        import os
        
        path = self._key_to_path(key)
        if not os.path.exists(path):
            return None
        
        try:
            with open(path, "r") as f:
                data = json.load(f)
            
            if time.time() > data.get("expiry", 0):
                os.remove(path)
                return None
            
            return data.get("value")
        except (json.JSONDecodeError, IOError):
            return None
    
    def set(self, key: str, value: dict[str, Any], ttl: Optional[int] = None) -> None:
        path = self._key_to_path(key)
        data = {
            "value": value,
            "expiry": time.time() + (ttl or self.ttl),
        }
        
        with open(path, "w") as f:
            json.dump(data, f)
    
    def delete(self, key: str) -> None:
        import os
        
        path = self._key_to_path(key)
        if os.path.exists(path):
            os.remove(path)
    
    def clear(self) -> None:
        import os
        import glob
        
        for path in glob.glob(os.path.join(self.directory, "*.json")):
            os.remove(path)


class RedisCache(CacheBackend):
    """Redis cache backend."""
    
    def __init__(self, config: dict[str, Any]):
        try:
            import redis
        except ImportError:
            raise ImportError("redis package is required. Install with: pip install redis")
        
        redis_config = config.get("redis", {})
        
        if "url" in redis_config:
            self.client = redis.from_url(redis_config["url"])
        else:
            self.client = redis.Redis(
                host=redis_config.get("host", "localhost"),
                port=redis_config.get("port", 6379),
                db=redis_config.get("db", 0),
                password=redis_config.get("password"),
            )
        
        self.ttl = config.get("ttl_seconds", 3600)
        self.prefix = config.get("key_prefix", "sql:")
    
    def _prefixed_key(self, key: str) -> str:
        return f"{self.prefix}{key}"
    
    def get(self, key: str) -> Optional[dict[str, Any]]:
        data = self.client.get(self._prefixed_key(key))
        if data is None:
            return None
        return json.loads(data)
    
    def set(self, key: str, value: dict[str, Any], ttl: Optional[int] = None) -> None:
        self.client.setex(
            self._prefixed_key(key),
            ttl or self.ttl,
            json.dumps(value, default=str),
        )
    
    def delete(self, key: str) -> None:
        self.client.delete(self._prefixed_key(key))
    
    def clear(self) -> None:
        pattern = f"{self.prefix}*"
        cursor = 0
        while True:
            cursor, keys = self.client.scan(cursor, match=pattern, count=100)
            if keys:
                self.client.delete(*keys)
            if cursor == 0:
                break


def create_cache(config: dict[str, Any]) -> CacheBackend:
    """
    Factory function to create the appropriate cache backend.
    
    Args:
        config: Cache configuration dictionary
        
    Returns:
        CacheBackend instance
    """
    backend = config.get("backend", "memory").lower()
    
    backends = {
        "memory": MemoryCache,
        "file": FileCache,
        "redis": RedisCache,
    }
    
    if backend not in backends:
        raise ValueError(f"Unknown cache backend: {backend}. Supported: {list(backends.keys())}")
    
    return backends[backend](config)
