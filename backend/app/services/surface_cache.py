"""
In-memory cache for storing surface geometry data.
"""
from typing import Dict, Any, Optional
from app.utils.serialization import make_json_serializable

class SurfaceCache:
    """
    A simple in-memory cache for surface data.
    In a production system, this would be replaced by a more robust solution
    like Redis or a database.
    """
    _cache: Dict[str, Dict[str, Any]] = {}

    def set(self, key: str, value: Dict[str, Any]):
        """Adds a value to the cache."""
        # Ensure the value is JSON serializable before storing
        serializable_value = make_json_serializable(value)
        self._cache[key] = serializable_value

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieves a value from the cache."""
        return self._cache.get(key)

    def delete(self, key: str):
        """Deletes a value from the cache."""
        if key in self._cache:
            del self._cache[key]

    def clear(self):
        """Clears the entire cache."""
        self._cache.clear()

    def keys(self):
        """Returns all cache keys."""
        return list(self._cache.keys())

    def size(self) -> int:
        """Returns the number of items in the cache."""
        return len(self._cache)

    def contains(self, key: str) -> bool:
        """Checks if a key exists in the cache."""
        return key in self._cache

# Global instance
surface_cache = SurfaceCache() 