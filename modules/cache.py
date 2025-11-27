"""
Cache module for AI-KnowMap system.
Handles caching, logging, and feedback storage.
"""

import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
import json


class Cache:
    """Simple in-memory cache for logs and feedback."""
    
    def __init__(self):
        self._logs = []
        self._feedback = []
        self._cache_data = {}
    
    def add_log(self, level: str, message: str):
        """Add a log entry."""
        self._logs.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'level': level,
            'message': message
        })
    
    def get_logs(self) -> List[Dict]:
        """Get all logs."""
        return self._logs
    
    def clear_logs(self):
        """Clear all logs."""
        self._logs = []
    
    def add_feedback(self, feedback: Dict):
        """Add user feedback."""
        self._feedback.append(feedback)
    
    def get_feedback(self) -> List[Dict]:
        """Get all feedback."""
        return self._feedback
    
    def set(self, key: str, value: Any):
        """Set a cache value."""
        self._cache_data[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a cache value."""
        return self._cache_data.get(key, default)
    
    def clear(self):
        """Clear all cache data."""
        self._cache_data = {}
    
    def has(self, key: str) -> bool:
        """Check if key exists in cache."""
        return key in self._cache_data


# Global cache instance
cache = Cache()