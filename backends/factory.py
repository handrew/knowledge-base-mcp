"""
Factory for creating backend instances.

Provides a simple interface for instantiating backends based on configuration.
"""

from typing import Type

from .base import BaseBackend, BackendConfig
from .sqlite_backend import SQLiteBackend


# Registry of available backends
_BACKENDS: dict[str, Type[BaseBackend]] = {
    "sqlite": SQLiteBackend,
}

# Try to register PostgreSQL backend if psycopg2 is available
try:
    from .postgres_backend import PostgresBackend, PSYCOPG2_AVAILABLE
    if PSYCOPG2_AVAILABLE:
        _BACKENDS["postgres"] = PostgresBackend
        _BACKENDS["postgresql"] = PostgresBackend  # Alias
except ImportError:
    pass


def create_backend(config: BackendConfig) -> BaseBackend:
    """Create a backend instance from configuration.

    Args:
        config: BackendConfig specifying the backend type and connection details

    Returns:
        An initialized backend instance

    Raises:
        ValueError: If the backend type is not recognized
        ImportError: If required dependencies for the backend are not installed

    Example:
        >>> config = BackendConfig(
        ...     backend_type="sqlite",
        ...     connection_string="/path/to/db.sqlite"
        ... )
        >>> backend = create_backend(config)

        >>> config = BackendConfig(
        ...     backend_type="postgres",
        ...     connection_string="postgresql://user:pass@localhost/kb",
        ...     options={"schema": "knowledge_base"}
        ... )
        >>> backend = create_backend(config)
    """
    backend_type = config.backend_type.lower()

    if backend_type not in _BACKENDS:
        available = list(_BACKENDS.keys())
        raise ValueError(
            f"Unknown backend type: '{backend_type}'. "
            f"Available backends: {available}"
        )

    backend_class = _BACKENDS[backend_type]
    return backend_class(config)


def create_backend_from_url(url: str, **options) -> BaseBackend:
    """Create a backend from a URL string.

    Convenience function that parses a URL and creates the appropriate backend.

    Args:
        url: Database URL or path. Supported formats:
            - sqlite:///path/to/db.sqlite or just /path/to/db.sqlite
            - postgresql://user:pass@host:port/database
            - postgres://user:pass@host:port/database
        **options: Additional backend-specific options

    Returns:
        An initialized backend instance

    Example:
        >>> backend = create_backend_from_url("sqlite:///tmp/test.db")
        >>> backend = create_backend_from_url("/path/to/kb.db")  # Assumes SQLite
        >>> backend = create_backend_from_url("postgresql://localhost/kb")
    """
    url_lower = url.lower()

    if url_lower.startswith("sqlite:///"):
        # sqlite:///path/to/db -> /path/to/db
        path = url[10:]  # Remove "sqlite:///"
        config = BackendConfig(
            backend_type="sqlite",
            connection_string=path,
            options=options
        )
    elif url_lower.startswith("postgresql://") or url_lower.startswith("postgres://"):
        config = BackendConfig(
            backend_type="postgres",
            connection_string=url,
            options=options
        )
    elif url_lower.endswith(".db") or url_lower.endswith(".sqlite") or url_lower.endswith(".sqlite3"):
        # Assume SQLite for common file extensions
        config = BackendConfig(
            backend_type="sqlite",
            connection_string=url,
            options=options
        )
    elif "/" in url and not "://" in url:
        # Looks like a file path - assume SQLite
        config = BackendConfig(
            backend_type="sqlite",
            connection_string=url,
            options=options
        )
    else:
        raise ValueError(
            f"Could not determine backend type from URL: '{url}'. "
            "Use an explicit URL scheme (sqlite://, postgresql://) or "
            "use create_backend() with explicit configuration."
        )

    return create_backend(config)


def get_available_backends() -> list[dict]:
    """Get list of available backends with their information.

    Returns:
        List of dicts with 'name' and 'available' keys
    """
    all_backends = [
        {
            "name": "sqlite",
            "description": "SQLite with FTS5 full-text search",
            "available": True,  # Always available
        },
        {
            "name": "postgres",
            "description": "PostgreSQL with tsvector full-text search",
            "available": "postgres" in _BACKENDS,
            "install_hint": "pip install psycopg2-binary" if "postgres" not in _BACKENDS else None,
        },
    ]
    return all_backends


def register_backend(name: str, backend_class: Type[BaseBackend]) -> None:
    """Register a custom backend implementation.

    Args:
        name: The backend type identifier (e.g., "mysql", "mongodb")
        backend_class: The backend class (must inherit from BaseBackend)

    Example:
        >>> from backends import register_backend, BaseBackend
        >>> class MyBackend(BaseBackend):
        ...     # Implementation
        ...     pass
        >>> register_backend("mybackend", MyBackend)
    """
    if not issubclass(backend_class, BaseBackend):
        raise TypeError(f"{backend_class} must inherit from BaseBackend")
    _BACKENDS[name.lower()] = backend_class
