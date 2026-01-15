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

    def add(self, content: str, source: str = "unknown") -> int:
        """Add a document to the knowledge base."""
        embedding = self._embed(content, is_query=False)
        return self.backend.add(content, source, embedding, self.model_name)

    def add_batch(self, documents: list[dict]) -> list[int]:
        """Add multiple documents. Each dict should have 'content' and optionally 'source'."""
        ids = []
        for doc in documents:
            content = doc.get("content", "")
            source = doc.get("source", "unknown")
            if content:
                doc_id = self.add(content, source)
                ids.append(doc_id)
        return ids

    def search_keyword(self, query: str, limit: int = 5) -> list[dict]:
        """Full-text keyword search."""
        results = self.backend.search_keyword(query, limit)
        return [r.to_dict() for r in results]

    def search_semantic(self, query: str, limit: int = 5) -> list[dict]:
        """Vector similarity search computed in memory."""
        query_embedding = self._embed(query, is_query=True)

        # Fetch all documents with embeddings
        docs = self.backend.get_all_embeddings()

        if not docs:
            return []

        # Compute similarities
        results = []
        for doc_id, content, source, embedding in docs:
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

    def search_hybrid(self, query: str, limit: int = 5, semantic_weight: float = 0.7) -> list[dict]:
        """Hybrid search combining semantic and keyword results."""
        semantic_results = self.search_semantic(query, limit * 2)
        keyword_results = self.search_keyword(query, limit * 2)

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
