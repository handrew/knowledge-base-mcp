"""
Local Knowledge Base with SQLite FTS5 + Vector Search (sqlite-vec)
"""

import sqlite3
import json
import struct
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Model registry: add new models here
MODELS = {
    "BAAI/bge-m3": {"dim": 1024, "prefix": None, "trust_remote_code": False},
    "BAAI/bge-base-en-v1.5": {"dim": 768, "prefix": None, "trust_remote_code": False},
    "nomic-ai/nomic-embed-text-v1.5": {"dim": 768, "prefix": ("search_query: ", "search_document: "), "trust_remote_code": True},
}

# Default model
DEFAULT_MODEL = "BAAI/bge-m3"


class KnowledgeBase:
    def __init__(self, db_path: str = "knowledge_base.db", model_name: str = DEFAULT_MODEL):
        self.db_path = db_path
        self.model_name = model_name
        self.model = None
        self._init_db()

    def _load_model(self, model_name: str = None):
        """Lazy load an embedding model."""
        model_name = model_name or self.model_name
        if self.model is None or self._current_model_name != model_name:
            model_config = MODELS.get(model_name, {})
            trust_remote = model_config.get("trust_remote_code", False)
            self.model = SentenceTransformer(model_name, trust_remote_code=trust_remote)
            self._current_model_name = model_name
        return self.model

    _current_model_name: str = None

    def _init_db(self):
        """Initialize database with FTS5 and VSS tables."""
        self.db = sqlite3.connect(self.db_path)
        self.db.row_factory = sqlite3.Row

        # Try to load sqlite-vec extension (replacement for deprecated sqlite-vss)
        try:
            self.db.enable_load_extension(True)
            import sqlite_vec
            sqlite_vec.load(self.db)
            self.db.enable_load_extension(False)
            self.vec_available = True
        except Exception as e:
            print(f"Warning: sqlite-vec not available ({e}). Vector search disabled.")
            self.vec_available = False

        # Create tables
        self.db.executescript("""
            -- Main documents table
            CREATE TABLE IF NOT EXISTS docs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                source TEXT DEFAULT 'unknown',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                embedding TEXT,
                embedding_model TEXT
            );

            -- Full-text search index
            CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(
                content,
                source,
                content='docs',
                content_rowid='id',
                tokenize='porter unicode61'
            );

            -- Triggers to keep FTS in sync
            CREATE TRIGGER IF NOT EXISTS docs_ai AFTER INSERT ON docs BEGIN
                INSERT INTO docs_fts(rowid, content, source)
                VALUES (new.id, new.content, new.source);
            END;

            CREATE TRIGGER IF NOT EXISTS docs_ad AFTER DELETE ON docs BEGIN
                INSERT INTO docs_fts(docs_fts, rowid, content, source)
                VALUES ('delete', old.id, old.content, old.source);
            END;

            CREATE TRIGGER IF NOT EXISTS docs_au AFTER UPDATE ON docs BEGIN
                INSERT INTO docs_fts(docs_fts, rowid, content, source)
                VALUES ('delete', old.id, old.content, old.source);
                INSERT INTO docs_fts(rowid, content, source)
                VALUES (new.id, new.content, new.source);
            END;
        """)

        # Migration: add embedding_model column if missing
        try:
            self.db.execute("ALTER TABLE docs ADD COLUMN embedding_model TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Handle vector index setup and migrations
        if self.vec_available:
            self._migrate_vec_index()

        self.db.commit()

    def _migrate_vec_index(self):
        """Handle vector index migrations and setup."""
        model_dim = MODELS.get(self.model_name, {}).get("dim", 1024)

        # Check for deprecated docs_vss table and remove it
        vss_exists = self.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='docs_vss'"
        ).fetchone()
        if vss_exists:
            print("Migrating from deprecated sqlite-vss to sqlite-vec...")
            self.db.execute("DROP TABLE docs_vss")

        # Check if docs_vec exists and get its dimension
        vec_exists = self.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='docs_vec'"
        ).fetchone()

        needs_rebuild = False
        if vec_exists:
            # Check dimension by looking at the table schema
            # vec0 tables store dimension in their schema
            try:
                # Try to get a sample to check dimension
                sample = self.db.execute(
                    "SELECT rowid FROM docs_vec LIMIT 1"
                ).fetchone()

                if sample:
                    # Test insert with current model dimension to see if it matches
                    # We do this by checking if any doc's embedding matches expected dim
                    doc_with_embedding = self.db.execute(
                        "SELECT id, embedding FROM docs WHERE embedding IS NOT NULL LIMIT 1"
                    ).fetchone()
                    if doc_with_embedding:
                        embedding = json.loads(doc_with_embedding["embedding"])
                        if len(embedding) != model_dim:
                            print(f"Vector index dimension mismatch: index has {len(embedding)}-dim embeddings, model needs {model_dim}")
                            needs_rebuild = True
            except Exception:
                # If we can't determine, rebuild to be safe
                needs_rebuild = True
        else:
            # No vec table exists, create it
            needs_rebuild = True

        if needs_rebuild:
            self._rebuild_vec_index(model_dim)

    def _rebuild_vec_index(self, model_dim: int):
        """Rebuild the vector index with the specified dimension."""
        print(f"Building vector index with {model_dim} dimensions...")

        # Drop existing table if any
        self.db.execute("DROP TABLE IF EXISTS docs_vec")

        # Create new table with correct dimensions
        self.db.execute(f"""
            CREATE VIRTUAL TABLE docs_vec USING vec0(
                embedding float[{model_dim}]
            )
        """)

        # Populate from stored embeddings
        docs = self.db.execute(
            "SELECT id, embedding FROM docs WHERE embedding IS NOT NULL"
        ).fetchall()

        populated = 0
        skipped = 0
        for doc in docs:
            try:
                embedding = json.loads(doc["embedding"])
                if len(embedding) == model_dim:
                    self.db.execute(
                        "INSERT INTO docs_vec (rowid, embedding) VALUES (?, ?)",
                        (doc["id"], self._serialize_embedding(embedding))
                    )
                    populated += 1
                else:
                    skipped += 1
            except (json.JSONDecodeError, TypeError):
                skipped += 1

        if populated > 0:
            print(f"Populated vector index with {populated} documents")
        if skipped > 0:
            print(f"Skipped {skipped} documents with incompatible embeddings (run reembed() to fix)")

    def _serialize_embedding(self, embedding: list[float]) -> bytes:
        """Serialize embedding to bytes for sqlite-vec."""
        return struct.pack(f"{len(embedding)}f", *embedding)

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

        cursor = self.db.execute(
            "INSERT INTO docs (content, source, embedding, embedding_model) VALUES (?, ?, ?, ?)",
            (content, source, json.dumps(embedding), self.model_name)
        )
        doc_id = cursor.lastrowid

        # Add to vector index
        if self.vec_available:
            self.db.execute(
                "INSERT INTO docs_vec (rowid, embedding) VALUES (?, ?)",
                (doc_id, self._serialize_embedding(embedding))
            )

        self.db.commit()
        return doc_id

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
        """Full-text keyword search using FTS5."""
        words = query.split()
        if not words:
            return []

        # Escape special FTS5 characters in each word
        safe_words = ['"' + word.replace('"', '""') + '"' for word in words]
        fts_query = " OR ".join(safe_words)

        results = self.db.execute("""
            SELECT d.id, d.content, d.source, f.rank
            FROM docs_fts f
            JOIN docs d ON d.id = f.rowid
            WHERE docs_fts MATCH ?
            ORDER BY f.rank
            LIMIT ?
        """, (fts_query, limit)).fetchall()

        return [
            {"id": r["id"], "content": r["content"], "source": r["source"], "score": -r["rank"]}
            for r in results
        ]

    def search_semantic(self, query: str, limit: int = 5) -> list[dict]:
        """Vector similarity search."""
        if not self.vec_available:
            return self.search_keyword(query, limit)

        query_embedding = self._embed(query, is_query=True)

        results = self.db.execute("""
            SELECT d.id, d.content, d.source, v.distance
            FROM docs_vec v
            JOIN docs d ON d.id = v.rowid
            WHERE v.embedding MATCH ?
            AND k = ?
        """, (self._serialize_embedding(query_embedding), limit)).fetchall()

        return [
            {"id": r["id"], "content": r["content"], "source": r["source"], "score": 1 / (1 + r["distance"])}
            for r in results
        ]

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
        cursor = self.db.execute("DELETE FROM docs WHERE id = ?", (doc_id,))
        if self.vec_available:
            self.db.execute("DELETE FROM docs_vec WHERE rowid = ?", (doc_id,))
        self.db.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Get total document count."""
        return self.db.execute("SELECT COUNT(*) FROM docs").fetchone()[0]

    def get(self, doc_id: int) -> dict | None:
        """Get a document by ID."""
        result = self.db.execute(
            "SELECT id, content, source, created_at, embedding_model FROM docs WHERE id = ?",
            (doc_id,)
        ).fetchone()
        if result:
            return dict(result)
        return None

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
        current_dim = MODELS.get(self.model_name, {}).get("dim")

        # Check if we need to rebuild VSS index due to dimension change
        needs_vss_rebuild = target_dim != current_dim

        # Get all docs
        docs = self.db.execute("SELECT id, content FROM docs").fetchall()
        total = len(docs)

        if total == 0:
            return {"reembedded": 0, "message": "No documents to re-embed"}

        # Re-embed in batches
        reembedded = 0
        for doc in docs:
            doc_id, content = doc["id"], doc["content"]
            embedding = self._embed(content, is_query=False, model_name=target_model)

            self.db.execute(
                "UPDATE docs SET embedding = ?, embedding_model = ? WHERE id = ?",
                (json.dumps(embedding), target_model, doc_id)
            )
            reembedded += 1

            if reembedded % batch_size == 0:
                self.db.commit()
                print(f"Re-embedded {reembedded}/{total} documents...")

        self.db.commit()

        # Rebuild vec index if dimensions changed
        vec_rebuilt = False
        if self.vec_available and needs_vss_rebuild:
            print(f"Rebuilding vec index for {target_dim} dimensions...")
            try:
                self.db.execute("DROP TABLE IF EXISTS docs_vec")
                self.db.execute(f"""
                    CREATE VIRTUAL TABLE docs_vec USING vec0(
                        embedding float[{target_dim}]
                    )
                """)
                # Re-populate vec index
                for doc in self.db.execute("SELECT id, embedding FROM docs").fetchall():
                    embedding = json.loads(doc["embedding"])
                    self.db.execute(
                        "INSERT INTO docs_vec (rowid, embedding) VALUES (?, ?)",
                        (doc["id"], self._serialize_embedding(embedding))
                    )
                self.db.commit()
                vec_rebuilt = True
            except Exception as e:
                print(f"Warning: Could not rebuild vec index ({e})")

        # Update current model
        self.model_name = target_model

        return {
            "reembedded": reembedded,
            "target_model": target_model,
            "target_dim": target_dim,
            "vec_rebuilt": vec_rebuilt,
        }

# Quick test
if __name__ == "__main__":
    kb = KnowledgeBase("test_kb.db")

    # Add some test documents
    kb.add("Python is a programming language known for its simplicity.", "test")
    kb.add("SQLite is a lightweight database engine.", "test")
    kb.add("Vector search enables semantic similarity matching.", "test")

    print(f"Total docs: {kb.count()}")
    print("\nSemantic search for 'database':")
    for r in kb.search_semantic("database", limit=2):
        print(f"  [{r['score']:.3f}] {r['content'][:50]}...")

    print("\nKeyword search for 'programming':")
    for r in kb.search_keyword("programming", limit=2):
        print(f"  [{r['score']:.3f}] {r['content'][:50]}...")
