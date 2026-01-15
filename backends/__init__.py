"""
Backend implementations for the Knowledge Base.

This module provides a pluggable backend architecture supporting multiple
database backends (SQLite, PostgreSQL, etc.) with a common interface.
"""

from .base import BaseBackend, BackendConfig
from .sqlite_backend import SQLiteBackend
from .factory import create_backend, create_backend_from_url, get_available_backends

__all__ = [
    "BaseBackend",
    "BackendConfig",
    "SQLiteBackend",
    "create_backend",
    "create_backend_from_url",
    "get_available_backends",
]

# Conditionally export PostgreSQL backend if psycopg2 is available
try:
    from .postgres_backend import PostgresBackend
    __all__.append("PostgresBackend")
except ImportError:
    pass
