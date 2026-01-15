"""
Abstract base class defining the backend interface for the Knowledge Base.

All backend implementations (SQLite, PostgreSQL, etc.) must implement this interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BackendConfig:
    """Configuration for a backend connection.

    Attributes:
        backend_type: Type of backend ("sqlite", "postgres", etc.)
        connection_string: Database connection string or path
        options: Additional backend-specific options
    """
    backend_type: str
    connection_string: str
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class Document:
    """Represents a document in the knowledge base.

    Attributes:
        id: Unique document identifier
        content: The text content of the document
        source: Source identifier (e.g., filename, URL)
        created_at: Timestamp when the document was created
        embedding: Vector embedding as a list of floats (optional)
        embedding_model: Name of the model used to generate the embedding
        metadata: Flexible key-value metadata (JSON-serializable dict)
        expires_at: Optional expiration timestamp (ISO format)
    """
    id: int
    content: str
    source: str
    created_at: str | None = None
    embedding: list[float] | None = None
    embedding_model: str | None = None
    metadata: dict[str, Any] | None = None
    expires_at: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        result = {
            "id": self.id,
            "content": self.content,
            "source": self.source,
            "created_at": self.created_at,
            "embedding_model": self.embedding_model,
        }
        if self.metadata is not None:
            result["metadata"] = self.metadata
        if self.expires_at is not None:
            result["expires_at"] = self.expires_at
        return result


@dataclass
class SearchResult:
    """Represents a search result.

    Attributes:
        id: Document identifier
        content: Document content
        source: Document source
        score: Relevance score (higher is better)
        metadata: Document metadata (optional)
    """
    id: int
    content: str
    source: str
    score: float
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        result = {
            "id": self.id,
            "content": self.content,
            "source": self.source,
            "score": self.score,
        }
        if self.metadata is not None:
            result["metadata"] = self.metadata
        return result


class BaseBackend(ABC):
    """Abstract base class for knowledge base backends.

    All backends must implement these methods to provide:
    - Document CRUD operations (add, get, delete, count)
    - Full-text search (keyword search)
    - Storage/retrieval of embeddings for semantic search
    - Batch operations for efficiency

    Note: Semantic search computation (cosine similarity) is handled at a higher
    level since it's the same regardless of backend. Backends just store/retrieve
    the embeddings.
    """

    @abstractmethod
    def __init__(self, config: BackendConfig):
        """Initialize the backend with configuration.

        Args:
            config: Backend configuration including connection details
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the backend connection and release resources."""
        pass

    # -------------------------------------------------------------------------
    # Document CRUD Operations
    # -------------------------------------------------------------------------

    @abstractmethod
    def add(
        self,
        content: str,
        source: str,
        embedding: list[float] | None,
        embedding_model: str | None,
        metadata: dict[str, Any] | None = None,
        expires_at: str | None = None
    ) -> int:
        """Add a document to the knowledge base.

        Args:
            content: The text content of the document
            source: Source identifier (e.g., filename, URL)
            embedding: Optional vector embedding
            embedding_model: Name of the model used for embedding
            metadata: Optional key-value metadata (JSON-serializable dict)
            expires_at: Optional expiration timestamp (ISO format)

        Returns:
            The ID of the newly created document
        """
        pass

    @abstractmethod
    def add_batch(
        self,
        documents: list[dict],
    ) -> list[int]:
        """Add multiple documents in a batch.

        Args:
            documents: List of dicts with keys:
                - content: str (required)
                - source: str (optional, defaults to "unknown")
                - embedding: list[float] | None (optional)
                - embedding_model: str | None (optional)

        Returns:
            List of IDs for the newly created documents
        """
        pass

    @abstractmethod
    def get(self, doc_id: int) -> Document | None:
        """Retrieve a document by ID.

        Args:
            doc_id: The document ID to retrieve

        Returns:
            The Document object if found, None otherwise
        """
        pass

    @abstractmethod
    def update(
        self,
        doc_id: int,
        content: str | None = None,
        source: str | None = None,
        embedding: list[float] | None = None,
        embedding_model: str | None = None,
        metadata: dict[str, Any] | None = None,
        expires_at: str | None = None
    ) -> bool:
        """Update an existing document.

        Only provided fields are updated; None values are ignored.

        Args:
            doc_id: The document ID to update
            content: New content (optional)
            source: New source (optional)
            embedding: New embedding vector (optional)
            embedding_model: New embedding model name (optional)
            metadata: New metadata dict (optional, replaces existing)
            expires_at: New expiration timestamp (optional)

        Returns:
            True if the document was updated, False if not found
        """
        pass

    @abstractmethod
    def delete(self, doc_id: int) -> bool:
        """Delete a document by ID.

        Args:
            doc_id: The document ID to delete

        Returns:
            True if the document was deleted, False if not found
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """Get the total number of documents.

        Returns:
            Total document count
        """
        pass

    # -------------------------------------------------------------------------
    # Search Operations
    # -------------------------------------------------------------------------

    @abstractmethod
    def search_keyword(self, query: str, limit: int = 5) -> list[SearchResult]:
        """Perform full-text keyword search.

        Args:
            query: The search query string
            limit: Maximum number of results to return

        Returns:
            List of SearchResult objects ordered by relevance (highest first)
        """
        pass

    @abstractmethod
    def get_all_embeddings(self) -> list[tuple[int, str, str, list[float]]]:
        """Retrieve all documents with their embeddings for semantic search.

        Returns:
            List of tuples: (id, content, source, embedding)
            Only includes documents that have embeddings.
        """
        pass

    # -------------------------------------------------------------------------
    # Embedding Management
    # -------------------------------------------------------------------------

    @abstractmethod
    def update_embedding(
        self,
        doc_id: int,
        embedding: list[float],
        embedding_model: str
    ) -> bool:
        """Update the embedding for a document.

        Args:
            doc_id: The document ID to update
            embedding: The new embedding vector
            embedding_model: Name of the model used

        Returns:
            True if updated successfully, False if document not found
        """
        pass

    @abstractmethod
    def get_all_documents(self) -> list[tuple[int, str]]:
        """Get all document IDs and content for re-embedding.

        Returns:
            List of tuples: (id, content)
        """
        pass

    # -------------------------------------------------------------------------
    # Source Management
    # -------------------------------------------------------------------------

    @abstractmethod
    def list_sources(self) -> list[dict]:
        """List all sources with document counts.

        Returns:
            List of dicts with 'source' and 'count' keys
        """
        pass

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    @abstractmethod
    def get_embedding_stats(self) -> list[dict]:
        """Get statistics about embeddings by model.

        Returns:
            List of dicts with 'model' and 'count' keys
        """
        pass

    # -------------------------------------------------------------------------
    # Document Listing and Filtering
    # -------------------------------------------------------------------------

    @abstractmethod
    def list_documents(
        self,
        source: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
        limit: int = 100,
        offset: int = 0
    ) -> list[Document]:
        """List documents with optional filtering.

        Args:
            source: Filter by source (optional)
            metadata_filter: Filter by metadata key-value pairs (optional).
                             Only documents matching ALL key-value pairs are returned.
            limit: Maximum number of documents to return
            offset: Number of documents to skip

        Returns:
            List of Document objects matching the filters
        """
        pass

    # -------------------------------------------------------------------------
    # Expiration Management
    # -------------------------------------------------------------------------

    @abstractmethod
    def cleanup_expired(self) -> int:
        """Delete all documents that have expired (expires_at < now).

        Returns:
            Number of documents deleted
        """
        pass

    # -------------------------------------------------------------------------
    # Deduplication
    # -------------------------------------------------------------------------

    @abstractmethod
    def find_duplicate(self, content: str) -> int | None:
        """Check if a document with the exact same content already exists.

        Args:
            content: The content to check for

        Returns:
            The document ID if a duplicate exists, None otherwise
        """
        pass

    # -------------------------------------------------------------------------
    # Bulk Operations
    # -------------------------------------------------------------------------

    @abstractmethod
    def delete_by_filter(
        self,
        source: str | None = None,
        metadata_filter: dict[str, Any] | None = None
    ) -> int:
        """Delete documents matching the filter criteria.

        Args:
            source: Filter by source (optional)
            metadata_filter: Filter by metadata key-value pairs (optional)

        Returns:
            Number of documents deleted
        """
        pass

    @abstractmethod
    def update_by_filter(
        self,
        source: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
        new_source: str | None = None,
        new_metadata: dict[str, Any] | None = None,
        metadata_merge: bool = False
    ) -> int:
        """Update documents matching the filter criteria.

        Args:
            source: Filter by source (optional)
            metadata_filter: Filter by metadata key-value pairs (optional)
            new_source: New source value to set (optional)
            new_metadata: New metadata to set (optional)
            metadata_merge: If True, merge with existing metadata; if False, replace

        Returns:
            Number of documents updated
        """
        pass

    # -------------------------------------------------------------------------
    # Transaction Support (optional - default no-op implementations)
    # -------------------------------------------------------------------------

    def begin_transaction(self) -> None:
        """Begin a database transaction."""
        pass

    def commit(self) -> None:
        """Commit the current transaction."""
        pass

    def rollback(self) -> None:
        """Rollback the current transaction."""
        pass

    # -------------------------------------------------------------------------
    # Backend Info
    # -------------------------------------------------------------------------

    @property
    @abstractmethod
    def backend_type(self) -> str:
        """Return the backend type identifier (e.g., 'sqlite', 'postgres')."""
        pass

    @property
    @abstractmethod
    def connection_info(self) -> str:
        """Return a safe string describing the connection (no passwords)."""
        pass
