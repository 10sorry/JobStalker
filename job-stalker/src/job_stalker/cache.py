"""
Caching Module for JobStalker

Provides:
- LRU cache for AI responses
- TTL-based cache expiration
- Async-safe caching
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, Optional, TypeVar

log = logging.getLogger("cache")

T = TypeVar('T')


@dataclass
class CacheEntry(Generic[T]):
    """Cache entry with TTL support"""
    value: T
    created_at: float = field(default_factory=time.time)
    ttl: float = 300.0  # 5 minutes default
    hits: int = 0

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired"""
        return time.time() - self.created_at > self.ttl

    def touch(self) -> None:
        """Update hit count"""
        self.hits += 1


class LRUCache(Generic[T]):
    """
    Thread-safe LRU cache with TTL support.

    Features:
    - Maximum size limit
    - TTL-based expiration
    - Hit/miss statistics
    - Async-safe operations
    """

    def __init__(
        self,
        max_size: int = 100,
        default_ttl: float = 300.0,
        name: str = "cache"
    ):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.name = name
        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0

    async def get(self, key: str) -> Optional[T]:
        """Get value from cache"""
        async with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._misses += 1
                return None

            if entry.is_expired:
                del self._cache[key]
                self._misses += 1
                log.debug(f"{self.name}: expired key={key[:20]}")
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()
            self._hits += 1

            log.debug(f"{self.name}: hit key={key[:20]}, hits={entry.hits}")
            return entry.value

    async def set(
        self,
        key: str,
        value: T,
        ttl: Optional[float] = None
    ) -> None:
        """Set value in cache"""
        async with self._lock:
            # Remove oldest if at capacity
            while len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                log.debug(f"{self.name}: evicted key={oldest_key[:20]}")

            self._cache[key] = CacheEntry(
                value=value,
                ttl=ttl or self.default_ttl
            )
            log.debug(f"{self.name}: set key={key[:20]}")

    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def clear(self) -> int:
        """Clear all entries, returns count of cleared"""
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            log.info(f"{self.name}: cleared {count} entries")
            return count

    async def cleanup_expired(self) -> int:
        """Remove expired entries, returns count of removed"""
        async with self._lock:
            expired_keys = [
                k for k, v in self._cache.items()
                if v.is_expired
            ]
            for key in expired_keys:
                del self._cache[key]

            if expired_keys:
                log.debug(f"{self.name}: cleaned up {len(expired_keys)} expired entries")
            return len(expired_keys)

    @property
    def size(self) -> int:
        """Current cache size"""
        return len(self._cache)

    @property
    def stats(self) -> Dict[str, Any]:
        """Cache statistics"""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        return {
            "name": self.name,
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{hit_rate:.1f}%"
        }


def make_cache_key(*args: Any, **kwargs: Any) -> str:
    """Generate cache key from arguments"""
    key_parts = [str(a) for a in args]
    key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
    key_str = "|".join(key_parts)
    return hashlib.md5(key_str.encode()).hexdigest()


class AIResponseCache:
    """
    Specialized cache for AI responses.

    Features:
    - Prompt-based caching
    - Model-specific caching
    - Configurable TTL per request type
    """

    # TTL settings for different request types (seconds)
    TTL_VACANCY_ANALYSIS = 3600  # 1 hour
    TTL_RECRUITER_ANALYSIS = 1800  # 30 minutes
    TTL_RESUME_IMPROVEMENT = 900  # 15 minutes
    TTL_RESUME_PARSE = 7200  # 2 hours

    def __init__(self, max_size: int = 200):
        self._cache = LRUCache[Dict[str, Any]](
            max_size=max_size,
            default_ttl=self.TTL_VACANCY_ANALYSIS,
            name="ai_cache"
        )

    def _make_key(self, prompt: str, model: str) -> str:
        """Create cache key from prompt and model"""
        # Use first 500 chars of prompt for key
        prompt_hash = hashlib.md5(prompt[:500].encode()).hexdigest()
        return f"{model}:{prompt_hash}"

    async def get_vacancy_analysis(
        self,
        vacancy_text: str,
        model: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached vacancy analysis"""
        key = self._make_key(vacancy_text, model)
        return await self._cache.get(key)

    async def set_vacancy_analysis(
        self,
        vacancy_text: str,
        model: str,
        result: Dict[str, Any]
    ) -> None:
        """Cache vacancy analysis"""
        key = self._make_key(vacancy_text, model)
        await self._cache.set(key, result, self.TTL_VACANCY_ANALYSIS)

    async def get_recruiter_analysis(
        self,
        vacancy_text: str,
        resume_text: str,
        model: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached recruiter analysis"""
        combined = f"{vacancy_text[:500]}|{resume_text[:500]}"
        key = self._make_key(combined, model)
        return await self._cache.get(key)

    async def set_recruiter_analysis(
        self,
        vacancy_text: str,
        resume_text: str,
        model: str,
        result: Dict[str, Any]
    ) -> None:
        """Cache recruiter analysis"""
        combined = f"{vacancy_text[:500]}|{resume_text[:500]}"
        key = self._make_key(combined, model)
        await self._cache.set(key, result, self.TTL_RECRUITER_ANALYSIS)

    async def clear(self) -> int:
        """Clear all cached responses"""
        return await self._cache.clear()

    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self._cache.stats


# Global cache instance
_ai_cache: Optional[AIResponseCache] = None


def get_ai_cache() -> AIResponseCache:
    """Get or create global AI cache"""
    global _ai_cache
    if _ai_cache is None:
        _ai_cache = AIResponseCache()
    return _ai_cache


async def cached_ai_call(
    cache_key: str,
    ttl: float,
    func: Callable,
    *args: Any,
    **kwargs: Any
) -> Any:
    """
    Decorator-like helper for caching async AI calls.

    Usage:
        result = await cached_ai_call(
            "vacancy_analysis",
            3600,
            analyze_vacancy,
            text, model
        )
    """
    cache = get_ai_cache()
    cached = await cache._cache.get(cache_key)

    if cached is not None:
        log.debug(f"Cache hit for {cache_key[:30]}")
        return cached

    log.debug(f"Cache miss for {cache_key[:30]}, calling function")
    result = await func(*args, **kwargs)

    if result is not None:
        await cache._cache.set(cache_key, result, ttl)

    return result
