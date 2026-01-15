"""
SQLite backend implementation for the Knowledge Base.

Uses SQLite with FTS5 for full-text search and JSON storage for embeddings.
"""

import json
import sqlite3
from typing import Any

from .base import BaseBackend, BackendConfig, Document, SearchResult


class SQLiteBackend(BaseBackend):
    """SQLite backend using FTS5 for full-text search.

    Features:
    - FTS5 with Porter stemmer for keyword search
    - JSON storage for embeddings (computed similarity in Python)
    - Trigger-based FTS synchronization
    - Automatic schema migration
    """

    def __init__(self, config: BackendConfig):
        """Initialize SQLite backend.

        Args:
            config: BackendConfig with connection_string as the database path
        """
        self.db_path = config.connection_string
        self.options = config.options
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema with FTS5."""
        self.db = sqlite3.connect(self.db_path)
        self.db.row_factory = sqlite3.Row

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

        # Clean up any leftover sqlite-vec tables from previous versions
        self._cleanup_legacy_vec_tables()

        self.db.commit()

    def _cleanup_legacy_vec_tables(self) -> None:
        """Remove legacy sqlite-vec/sqlite-vss tables if they exist."""
        legacy_tables = [
            "docs_vss",  # old sqlite-vss
            "docs_vec", "docs_vec_chunks", "docs_vec_rowids",
            "docs_vec_vector_chunks00", "docs_vec_info"  # sqlite-vec
        ]
        for table in legacy_tables:
            try:
                self.db.execute(f"DROP TABLE IF EXISTS {table}")
            except sqlite3.OperationalError:
                pass

    def close(self) -> None:
        """Close the database connection."""
        if hasattr(self, 'db') and self.db:
            self.db.close()
            self.db = None

    # -------------------------------------------------------------------------
    # Document CRUD Operations
    # -------------------------------------------------------------------------

    def add(
        self,
        content: str,
        source: str,
        embedding: list[float] | None,
        embedding_model: str | None
    ) -> int:
        """Add a document to the knowledge base."""
        embedding_json = json.dumps(embedding) if embedding else None

        cursor = self.db.execute(
            "INSERT INTO docs (content, source, embedding, embedding_model) VALUES (?, ?, ?, ?)",
            (content, source, embedding_json, embedding_model)
        )
        doc_id = cursor.lastrowid
        self.db.commit()
        return doc_id

    def add_batch(self, documents: list[dict]) -> list[int]:
        """Add multiple documents in a batch."""
        ids = []
        for doc in documents:
            content = doc.get("content", "")
            if not content:
                continue
            source = doc.get("source", "unknown")
            embedding = doc.get("embedding")
            embedding_model = doc.get("embedding_model")
            doc_id = self.add(content, source, embedding, embedding_model)
            ids.append(doc_id)
        return ids

    def get(self, doc_id: int) -> Document | None:
        """Retrieve a document by ID."""
        result = self.db.execute(
            "SELECT id, content, source, created_at, embedding_model FROM docs WHERE id = ?",
            (doc_id,)
        ).fetchone()

        if result:
            return Document(
                id=result["id"],
                content=result["content"],
                source=result["source"],
                created_at=result["created_at"],
                embedding_model=result["embedding_model"],
            )
        return None

    def delete(self, doc_id: int) -> bool:
        """Delete a document by ID."""
        cursor = self.db.execute("DELETE FROM docs WHERE id = ?", (doc_id,))
        self.db.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Get total document count."""
        return self.db.execute("SELECT COUNT(*) FROM docs").fetchone()[0]

    # -------------------------------------------------------------------------
    # Search Operations
    # -------------------------------------------------------------------------

    def search_keyword(self, query: str, limit: int = 5) -> list[SearchResult]:
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
            SearchResult(
                id=r["id"],
                content=r["content"],
                source=r["source"],
                score=-r["rank"]  # FTS5 rank is negative, negate for positive scores
            )
            for r in results
        ]

    def get_all_embeddings(self) -> list[tuple[int, str, str, list[float]]]:
        """Retrieve all documents with embeddings for semantic search."""
        docs = self.db.execute(
            "SELECT id, content, source, embedding FROM docs WHERE embedding IS NOT NULL"
        ).fetchall()

        results = []
        for doc in docs:
            try:
                embedding = json.loads(doc["embedding"])
                results.append((doc["id"], doc["content"], doc["source"], embedding))
            except (json.JSONDecodeError, TypeError):
                continue

        return results

    # -------------------------------------------------------------------------
    # Embedding Management
    # -------------------------------------------------------------------------

    def update_embedding(
        self,
        doc_id: int,
        embedding: list[float],
        embedding_model: str
    ) -> bool:
        """Update the embedding for a document."""
        cursor = self.db.execute(
            "UPDATE docs SET embedding = ?, embedding_model = ? WHERE id = ?",
            (json.dumps(embedding), embedding_model, doc_id)
        )
        self.db.commit()
        return cursor.rowcount > 0

    def get_all_documents(self) -> list[tuple[int, str]]:
        """Get all document IDs and content for re-embedding."""
        docs = self.db.execute("SELECT id, content FROM docs").fetchall()
        return [(doc["id"], doc["content"]) for doc in docs]

    # -------------------------------------------------------------------------
    # Source Management
    # -------------------------------------------------------------------------

    def list_sources(self) -> list[dict]:
        """List all sources with document counts."""
        results = self.db.execute(
            "SELECT source, COUNT(*) as count FROM docs GROUP BY source ORDER BY count DESC"
        ).fetchall()
        return [{"source": r["source"], "count": r["count"]} for r in results]

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def get_embedding_stats(self) -> list[dict]:
        """Get statistics about embeddings by model."""
        results = self.db.execute("""
            SELECT embedding_model as model, COUNT(*) as count
            FROM docs
            WHERE embedding_model IS NOT NULL
            GROUP BY embedding_model
        """).fetchall()
        return [{"model": r["model"], "count": r["count"]} for r in results]

    # -------------------------------------------------------------------------
    # Transaction Support
    # -------------------------------------------------------------------------

    def begin_transaction(self) -> None:
        """Begin a database transaction."""
        self.db.execute("BEGIN")

    def commit(self) -> None:
        """Commit the current transaction."""
        self.db.commit()

    def rollback(self) -> None:
        """Rollback the current transaction."""
        self.db.rollback()

    # -------------------------------------------------------------------------
    # Backend Info
    # -------------------------------------------------------------------------

    @property
    def backend_type(self) -> str:
        """Return the backend type identifier."""
        return "sqlite"

    @property
    def connection_info(self) -> str:
        """Return the database path."""
        return self.db_path
