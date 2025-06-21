"""
In-memory cache for storing surface geometry data.
"""
from typing import Dict, Any, Optional
import threading

class SurfaceCache:
    """
    A simple thread-safe, in-memory cache for surface data.
    In a production system, this would be replaced by a more robust solution
    like Redis or a database.
    """
    _cache: Dict[str, Dict[str, Any]] = {}
    _lock = threading.Lock()

    def set(self, key: str, value: Dict[str, Any]):
        """Adds a value to the cache."""
        with self._lock:
            self._cache[key] = value

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieves a value from the cache."""
        with self._lock:
            return self._cache.get(key)

    def delete(self, key: str):
        """Deletes a value from the cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]

    def clear(self):
        """Clears the entire cache."""
        with self._lock:
            self._cache.clear()

# Create a singleton instance of the cache
surface_cache = SurfaceCache() 