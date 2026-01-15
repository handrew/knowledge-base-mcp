"""
Local Knowledge Base with pluggable backend support.

Supports SQLite (default) and PostgreSQL backends with:
- Full-text keyword search
- Semantic search via embeddings
- Hybrid search combining both approaches
"""

import math
from sentence_transformers import SentenceTransformer

from backends import BaseBackend, BackendConfig, create_backend, create_backend_from_url

# Model registry: add new models here
MODELS = {
    "BAAI/bge-m3": {"dim": 1024, "prefix": None, "trust_remote_code": False},
    "BAAI/bge-base-en-v1.5": {"dim": 768, "prefix": None, "trust_remote_code": False},
    "nomic-ai/nomic-embed-text-v1.5": {"dim": 768, "prefix": ("search_query: ", "search_document: "), "trust_remote_code": True},
}

# Default model
DEFAULT_MODEL = "BAAI/bge-m3"


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class KnowledgeBase:
    """Knowledge base with pluggable backend support.

    Supports multiple backends (SQLite, PostgreSQL) with a unified interface
    for document storage, full-text search, and semantic search.

    Example usage:
        # SQLite (default)
        kb = KnowledgeBase("knowledge_base.db")

        # PostgreSQL
        kb = KnowledgeBase("postgresql://user:pass@localhost/kb")

        # Explicit backend configuration
        from backends import BackendConfig
        config = BackendConfig(
            backend_type="postgres",
            connection_string="postgresql://localhost/kb",
            options={"schema": "my_schema"}
        )
        kb = KnowledgeBase(backend_config=config)
    """

    def __init__(
        self,
        db_path: str = "knowledge_base.db",
        model_name: str = DEFAULT_MODEL,
        backend_config: BackendConfig | None = None
    ):
        """Initialize the knowledge base.

        Args:
            db_path: Database path or connection URL. Used if backend_config is not provided.
                     Supports: file paths (SQLite), postgresql:// URLs, sqlite:// URLs
            model_name: Name of the embedding model to use
            backend_config: Explicit backend configuration (overrides db_path if provided)
        """
        self.model_name = model_name
        self.model = None
        self._current_model_name: str = None

        # Initialize backend
        if backend_config:
            self.backend = create_backend(backend_config)
        else:
            self.backend = create_backend_from_url(db_path)

    def _load_model(self, model_name: str = None):
        """Lazy load an embedding model."""
        model_name = model_name or self.model_name
        if self.model is None or self._current_model_name != model_name:
            model_config = MODELS.get(model_name, {})
            trust_remote = model_config.get("trust_remote_code", False)
            self.model = SentenceTransformer(model_name, trust_remote_code=trust_remote)
            self._current_model_name = model_name
        return self.model

    def _embed(self, text: str, is_query: bool = False, model_name: str = None) -> list[float]:
        """Generate embedding for text."""
        model_name = model_name or self.model_name
        model = self._load_model(model_name)
        model_config = MODELS.get(model_name, {})

        # Some models use prefixes for query vs document
        prefix = model_config.get("prefix")
        if prefix:
            query_prefix, doc_prefix = prefix
            text = (query_prefix if is_query else doc_prefix) + text

        embedding = model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def _embed_batch(self, texts: list[str], is_query: bool = False, model_name: str = None) -> list[list[float]]:
        """Generate embeddings for multiple texts in a batch (more efficient)."""
        if not texts:
            return []

        model_name = model_name or self.model_name
        model = self._load_model(model_name)
        model_config = MODELS.get(model_name, {})

        # Some models use prefixes for query vs document
        prefix = model_config.get("prefix")
        if prefix:
            query_prefix, doc_prefix = prefix
            prefix_str = query_prefix if is_query else doc_prefix
            texts = [prefix_str + t for t in texts]

        embeddings = model.encode(texts, normalize_embeddings=True)
        return [emb.tolist() for emb in embeddings]

    def add(
        self,
        content: str,
        source: str = "unknown",
        metadata: dict | None = None,
        expires_at: str | None = None,
        check_duplicate: bool = False
    ) -> int | None:
        """Add a document to the knowledge base.

        Args:
            content: The text content to add
            source: Source identifier (e.g., filename, URL)
            metadata: Optional key-value metadata (JSON-serializable dict)
            expires_at: Optional expiration timestamp (ISO format, e.g., "2024-12-31T23:59:59")
            check_duplicate: If True, check for exact content duplicate and return existing ID

        Returns:
            The document ID (new or existing if duplicate found), or None if duplicate and check_duplicate=True
        """
        if check_duplicate:
            existing_id = self.backend.find_duplicate(content)
            if existing_id is not None:
                return existing_id

        embedding = self._embed(content, is_query=False)
        return self.backend.add(content, source, embedding, self.model_name, metadata, expires_at)

    def add_batch(self, documents: list[dict], check_duplicate: bool = False) -> list[int]:
        """Add multiple documents with batch embedding for efficiency.

        Each dict should have:
            - content: str (required)
            - source: str (optional, defaults to "unknown")
            - metadata: dict (optional)
            - expires_at: str (optional, ISO format timestamp)

        Uses batch embedding which is significantly faster than embedding
        documents one at a time, especially for large batches.
        """
        if not documents:
            return []

        # Filter to valid documents and check duplicates if needed
        valid_docs = []
        for doc in documents:
            content = doc.get("content", "")
            if not content:
                continue
            if check_duplicate:
                existing_id = self.backend.find_duplicate(content)
                if existing_id is not None:
                    valid_docs.append({"existing_id": existing_id, **doc})
                    continue
            valid_docs.append(doc)

        if not valid_docs:
            return []

        # Separate new docs (need embedding) from existing duplicates
        new_docs = [d for d in valid_docs if "existing_id" not in d]
        existing_ids = [d["existing_id"] for d in valid_docs if "existing_id" in d]

        if not new_docs:
            return existing_ids

        # Batch embed all new documents at once
        contents = [d.get("content", "") for d in new_docs]
        embeddings = self._embed_batch(contents, is_query=False)

        # Add documents with their embeddings
        ids = list(existing_ids)  # Start with existing duplicate IDs
        for doc, embedding in zip(new_docs, embeddings):
            content = doc.get("content", "")
            source = doc.get("source", "unknown")
            metadata = doc.get("metadata")
            expires_at = doc.get("expires_at")

            doc_id = self.backend.add(content, source, embedding, self.model_name, metadata, expires_at)
            ids.append(doc_id)

        return ids

    def search_keyword(
        self,
        query: str,
        limit: int = 5,
        metadata_filter: dict | None = None
    ) -> list[dict]:
        """Full-text keyword search.

        Args:
            query: The search query
            limit: Maximum results to return
            metadata_filter: Filter results by metadata key-value pairs (optional).
                             Only documents matching ALL key-value pairs are returned.
        """
        # If filtering, fetch extra results to account for filtering
        fetch_limit = limit * 3 if metadata_filter else limit
        results = self.backend.search_keyword(query, fetch_limit)

        if metadata_filter:
            results = self._filter_by_metadata(results, metadata_filter)

        return [r.to_dict() for r in results[:limit]]

    def _filter_by_metadata(self, results: list, metadata_filter: dict) -> list:
        """Filter search results by metadata."""
        filtered = []
        for r in results:
            # Get full document to check metadata
            doc = self.backend.get(r.id)
            if doc and doc.metadata:
                if all(doc.metadata.get(k) == v for k, v in metadata_filter.items()):
                    r.metadata = doc.metadata
                    filtered.append(r)
            elif not metadata_filter:
                # If no filter required, include all
                filtered.append(r)
        return filtered

    def search_semantic(
        self,
        query: str,
        limit: int = 5,
        metadata_filter: dict | None = None
    ) -> list[dict]:
        """Vector similarity search computed in memory.

        Args:
            query: The search query
            limit: Maximum results to return
            metadata_filter: Filter results by metadata key-value pairs (optional).
                             Only documents matching ALL key-value pairs are returned.
        """
        query_embedding = self._embed(query, is_query=True)

        # Fetch all documents with embeddings
        docs = self.backend.get_all_embeddings()

        if not docs:
            return []

        # Compute similarities
        results = []
        for doc_id, content, source, embedding in docs:
            # If filtering, check metadata before computing similarity
            if metadata_filter:
                doc = self.backend.get(doc_id)
                if not doc or not doc.metadata:
                    continue
                if not all(doc.metadata.get(k) == v for k, v in metadata_filter.items()):
                    continue

            similarity = cosine_similarity(query_embedding, embedding)
            results.append({
                "id": doc_id,
                "content": content,
                "source": source,
                "score": similarity
            })

        # Sort by score descending and limit
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def search_hybrid(
        self,
        query: str,
        limit: int = 5,
        semantic_weight: float = 0.7,
        metadata_filter: dict | None = None
    ) -> list[dict]:
        """Hybrid search combining semantic and keyword results.

        Args:
            query: The search query
            limit: Maximum results to return
            semantic_weight: Weight for semantic vs keyword results (default: 0.7)
            metadata_filter: Filter results by metadata key-value pairs (optional).
                             Only documents matching ALL key-value pairs are returned.
        """
        semantic_results = self.search_semantic(query, limit * 2, metadata_filter)
        keyword_results = self.search_keyword(query, limit * 2, metadata_filter)

        # Normalize and combine scores
        scores = {}

        # Add semantic scores
        if semantic_results:
            max_sem = max(r["score"] for r in semantic_results)
            for r in semantic_results:
                norm_score = r["score"] / max_sem if max_sem > 0 else 0
                scores[r["id"]] = {
                    "content": r["content"],
                    "source": r["source"],
                    "score": semantic_weight * norm_score
                }

        # Add keyword scores
        if keyword_results:
            max_kw = max(r["score"] for r in keyword_results)
            for r in keyword_results:
                norm_score = r["score"] / max_kw if max_kw > 0 else 0
                kw_weight = 1 - semantic_weight
                if r["id"] in scores:
                    scores[r["id"]]["score"] += kw_weight * norm_score
                else:
                    scores[r["id"]] = {
                        "content": r["content"],
                        "source": r["source"],
                        "score": kw_weight * norm_score
                    }

        # Sort and return
        ranked = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)
        return [
            {"id": id, "content": data["content"], "source": data["source"], "score": data["score"]}
            for id, data in ranked[:limit]
        ]

    def update(
        self,
        doc_id: int,
        content: str | None = None,
        source: str | None = None,
        metadata: dict | None = None,
        expires_at: str | None = None,
        metadata_merge: bool = False
    ) -> bool:
        """Update an existing document.

        If content is updated, the embedding is automatically regenerated.

        Args:
            doc_id: The document ID to update
            content: New content (optional)
            source: New source (optional)
            metadata: New metadata dict (optional)
            expires_at: New expiration timestamp (optional)
            metadata_merge: If True, merge with existing metadata; if False, replace

        Returns:
            True if the document was updated, False if not found
        """
        # If content is being updated, regenerate embedding
        embedding = None
        embedding_model = None
        if content is not None:
            embedding = self._embed(content, is_query=False)
            embedding_model = self.model_name

        # Handle metadata merge at this level for consistency across backends
        final_metadata = metadata
        if metadata_merge and metadata is not None:
            doc = self.get(doc_id)
            if doc:
                existing = doc.get("metadata") or {}
                final_metadata = {**existing, **metadata}

        return self.backend.update(
            doc_id,
            content=content,
            source=source,
            embedding=embedding,
            embedding_model=embedding_model,
            metadata=final_metadata,
            expires_at=expires_at
        )

    def append(self, doc_id: int, content: str, separator: str = "\n\n") -> bool:
        """Append content to an existing document.

        The new content is concatenated to the existing content with a separator,
        and the embedding is regenerated for the combined text.

        Args:
            doc_id: The document ID to append to
            content: Content to append
            separator: Separator between existing and new content (default: double newline)

        Returns:
            True if the document was updated, False if not found
        """
        doc = self.get(doc_id)
        if not doc:
            return False
        new_content = doc["content"] + separator + content
        return self.update(doc_id, content=new_content)

    def delete(self, doc_id: int) -> bool:
        """Delete a document by ID."""
        return self.backend.delete(doc_id)

    def count(self) -> int:
        """Get total document count."""
        return self.backend.count()

    def get(self, doc_id: int) -> dict | None:
        """Get a document by ID."""
        doc = self.backend.get(doc_id)
        if doc:
            return doc.to_dict()
        return None

    def list_sources(self) -> list[dict]:
        """List all sources with document counts."""
        return self.backend.list_sources()

    def list_documents(
        self,
        source: str | None = None,
        metadata_filter: dict | None = None,
        limit: int = 100,
        offset: int = 0
    ) -> list[dict]:
        """List documents with optional filtering.

        Args:
            source: Filter by source (optional)
            metadata_filter: Filter by metadata key-value pairs (optional).
                             Only documents matching ALL key-value pairs are returned.
            limit: Maximum number of documents to return
            offset: Number of documents to skip (for pagination)

        Returns:
            List of documents matching the filters
        """
        docs = self.backend.list_documents(source, metadata_filter, limit, offset)
        return [doc.to_dict() for doc in docs]

    def cleanup_expired(self) -> int:
        """Delete all documents that have expired (expires_at < now).

        Returns:
            Number of documents deleted
        """
        return self.backend.cleanup_expired()

    def find_duplicate(self, content: str) -> int | None:
        """Check if a document with the exact same content already exists.

        Args:
            content: The content to check for

        Returns:
            The document ID if a duplicate exists, None otherwise
        """
        return self.backend.find_duplicate(content)

    def delete_by_filter(
        self,
        source: str | None = None,
        metadata_filter: dict | None = None
    ) -> int:
        """Delete documents matching the filter criteria.

        Args:
            source: Filter by source (optional)
            metadata_filter: Filter by metadata key-value pairs (optional).
                             Only documents matching ALL key-value pairs are deleted.

        Returns:
            Number of documents deleted

        Raises:
            ValueError: If neither source nor metadata_filter is provided
        """
        return self.backend.delete_by_filter(source, metadata_filter)

    def update_by_filter(
        self,
        source: str | None = None,
        metadata_filter: dict | None = None,
        new_source: str | None = None,
        new_metadata: dict | None = None,
        metadata_merge: bool = False
    ) -> int:
        """Update documents matching the filter criteria.

        Args:
            source: Filter by source (optional)
            metadata_filter: Filter by metadata key-value pairs (optional).
                             Only documents matching ALL key-value pairs are updated.
            new_source: New source value to set (optional)
            new_metadata: New metadata to set (optional)
            metadata_merge: If True, merge with existing metadata; if False, replace

        Returns:
            Number of documents updated

        Raises:
            ValueError: If no filter or no update is provided
        """
        return self.backend.update_by_filter(
            source, metadata_filter, new_source, new_metadata, metadata_merge
        )

    def get_embedding_stats(self) -> dict:
        """Get statistics about embeddings."""
        models = self.backend.get_embedding_stats()
        return {
            "current_model": self.model_name,
            "models": models,
        }

    def list_available_models(self) -> list[dict]:
        """List available embedding models."""
        return [
            {"name": name, "dim": info["dim"]}
            for name, info in MODELS.items()
        ]

    def reembed(self, target_model: str = None, batch_size: int = 100) -> dict:
        """
        Re-embed all documents with a new model.

        Args:
            target_model: Model to use (defaults to current model)
            batch_size: Number of docs to process at a time

        Returns:
            Stats about the re-embedding process
        """
        target_model = target_model or self.model_name
        if target_model not in MODELS:
            return {"error": f"Unknown model: {target_model}. Known models: {list(MODELS.keys())}"}

        target_dim = MODELS[target_model]["dim"]

        # Get all docs
        docs = self.backend.get_all_documents()
        total = len(docs)

        if total == 0:
            return {"reembedded": 0, "message": "No documents to re-embed"}

        # Re-embed in batches
        reembedded = 0
        for doc_id, content in docs:
            embedding = self._embed(content, is_query=False, model_name=target_model)
            self.backend.update_embedding(doc_id, embedding, target_model)
            reembedded += 1

            if reembedded % batch_size == 0:
                self.backend.commit()
                print(f"Re-embedded {reembedded}/{total} documents...")

        self.backend.commit()

        # Update current model
        self.model_name = target_model

        return {
            "reembedded": reembedded,
            "target_model": target_model,
            "target_dim": target_dim,
        }

    def close(self) -> None:
        """Close the backend connection."""
        if self.backend:
            self.backend.close()

    @property
    def db_path(self) -> str:
        """Get the connection info (for backwards compatibility)."""
        return self.backend.connection_info

    @property
    def backend_type(self) -> str:
        """Get the backend type."""
        return self.backend.backend_type


# Quick test
if __name__ == "__main__":
    kb = KnowledgeBase("test_kb.db")

    # Add some test documents
    kb.add("Python is a programming language known for its simplicity.", "test")
    kb.add("SQLite is a lightweight database engine.", "test")
    kb.add("Vector search enables semantic similarity matching.", "test")

    print(f"Backend: {kb.backend_type}")
    print(f"Total docs: {kb.count()}")
    print("\nSemantic search for 'database':")
    for r in kb.search_semantic("database", limit=2):
        print(f"  [{r['score']:.3f}] {r['content'][:50]}...")

    print("\nKeyword search for 'programming':")
    for r in kb.search_keyword("programming", limit=2):
        print(f"  [{r['score']:.3f}] {r['content'][:50]}...")

    kb.close()
