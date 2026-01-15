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

        # Migrations: add columns if missing
        migrations = [
            "ALTER TABLE docs ADD COLUMN embedding_model TEXT",
            "ALTER TABLE docs ADD COLUMN metadata TEXT",
            "ALTER TABLE docs ADD COLUMN expires_at TIMESTAMP",
        ]
        for migration in migrations:
            try:
                self.db.execute(migration)
            except sqlite3.OperationalError:
                pass  # Column already exists

        # Clean up any leftover sqlite-vec tables from previous versions
        self._cleanup_legacy_vec_tables()

        # Clean up expired documents on startup
        self.cleanup_expired()

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
        embedding_model: str | None,
        metadata: dict[str, Any] | None = None,
        expires_at: str | None = None
    ) -> int:
        """Add a document to the knowledge base."""
        embedding_json = json.dumps(embedding) if embedding else None
        metadata_json = json.dumps(metadata) if metadata else None

        cursor = self.db.execute(
            """INSERT INTO docs (content, source, embedding, embedding_model, metadata, expires_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (content, source, embedding_json, embedding_model, metadata_json, expires_at)
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
            """SELECT id, content, source, created_at, embedding_model, metadata, expires_at
               FROM docs WHERE id = ?""",
            (doc_id,)
        ).fetchone()

        if result:
            metadata = None
            if result["metadata"]:
                try:
                    metadata = json.loads(result["metadata"])
                except json.JSONDecodeError:
                    pass
            return Document(
                id=result["id"],
                content=result["content"],
                source=result["source"],
                created_at=result["created_at"],
                embedding_model=result["embedding_model"],
                metadata=metadata,
                expires_at=result["expires_at"],
            )
        return None

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
        """Update an existing document."""
        # Build dynamic UPDATE query based on provided fields
        updates = []
        params = []

        if content is not None:
            updates.append("content = ?")
            params.append(content)
        if source is not None:
            updates.append("source = ?")
            params.append(source)
        if embedding is not None:
            updates.append("embedding = ?")
            params.append(json.dumps(embedding))
        if embedding_model is not None:
            updates.append("embedding_model = ?")
            params.append(embedding_model)
        if metadata is not None:
            updates.append("metadata = ?")
            params.append(json.dumps(metadata))
        if expires_at is not None:
            updates.append("expires_at = ?")
            params.append(expires_at)

        if not updates:
            # Nothing to update, check if doc exists
            return self.get(doc_id) is not None

        params.append(doc_id)
        query = f"UPDATE docs SET {', '.join(updates)} WHERE id = ?"

        cursor = self.db.execute(query, params)
        self.db.commit()
        return cursor.rowcount > 0

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
    # Document Listing and Filtering
    # -------------------------------------------------------------------------

    def list_documents(
        self,
        source: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
        limit: int = 100,
        offset: int = 0
    ) -> list[Document]:
        """List documents with optional filtering."""
        query = """SELECT id, content, source, created_at, embedding_model, metadata, expires_at
                   FROM docs WHERE 1=1"""
        params = []

        if source is not None:
            query += " AND source = ?"
            params.append(source)

        # For metadata filtering in SQLite, we use JSON functions
        if metadata_filter:
            for key, value in metadata_filter.items():
                query += " AND json_extract(metadata, ?) = ?"
                params.append(f"$.{key}")
                params.append(json.dumps(value) if isinstance(value, (dict, list)) else value)

        query += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        results = self.db.execute(query, params).fetchall()

        documents = []
        for r in results:
            metadata = None
            if r["metadata"]:
                try:
                    metadata = json.loads(r["metadata"])
                except json.JSONDecodeError:
                    pass
            documents.append(Document(
                id=r["id"],
                content=r["content"],
                source=r["source"],
                created_at=r["created_at"],
                embedding_model=r["embedding_model"],
                metadata=metadata,
                expires_at=r["expires_at"],
            ))
        return documents

    # -------------------------------------------------------------------------
    # Expiration Management
    # -------------------------------------------------------------------------

    def cleanup_expired(self) -> int:
        """Delete all documents that have expired (expires_at < now)."""
        cursor = self.db.execute(
            "DELETE FROM docs WHERE expires_at IS NOT NULL AND expires_at < datetime('now')"
        )
        deleted = cursor.rowcount
        self.db.commit()
        return deleted

    # -------------------------------------------------------------------------
    # Deduplication
    # -------------------------------------------------------------------------

    def find_duplicate(self, content: str) -> int | None:
        """Check if a document with the exact same content already exists."""
        result = self.db.execute(
            "SELECT id FROM docs WHERE content = ? LIMIT 1",
            (content,)
        ).fetchone()
        return result["id"] if result else None

    # -------------------------------------------------------------------------
    # Bulk Operations
    # -------------------------------------------------------------------------

    def delete_by_filter(
        self,
        source: str | None = None,
        metadata_filter: dict[str, Any] | None = None
    ) -> int:
        """Delete documents matching the filter criteria."""
        if source is None and metadata_filter is None:
            raise ValueError("At least one filter (source or metadata_filter) must be provided")

        query = "DELETE FROM docs WHERE 1=1"
        params = []

        if source is not None:
            query += " AND source = ?"
            params.append(source)

        if metadata_filter:
            for key, value in metadata_filter.items():
                query += " AND json_extract(metadata, ?) = ?"
                params.append(f"$.{key}")
                params.append(json.dumps(value) if isinstance(value, (dict, list)) else value)

        cursor = self.db.execute(query, params)
        deleted = cursor.rowcount
        self.db.commit()
        return deleted

    def update_by_filter(
        self,
        source: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
        new_source: str | None = None,
        new_metadata: dict[str, Any] | None = None,
        metadata_merge: bool = False
    ) -> int:
        """Update documents matching the filter criteria."""
        if source is None and metadata_filter is None:
            raise ValueError("At least one filter (source or metadata_filter) must be provided")

        if new_source is None and new_metadata is None:
            raise ValueError("At least one update (new_source or new_metadata) must be provided")

        # For metadata merge, we need to handle it row by row in SQLite
        if metadata_merge and new_metadata:
            # Get matching doc IDs first
            select_query = "SELECT id, metadata FROM docs WHERE 1=1"
            params = []

            if source is not None:
                select_query += " AND source = ?"
                params.append(source)

            if metadata_filter:
                for key, value in metadata_filter.items():
                    select_query += " AND json_extract(metadata, ?) = ?"
                    params.append(f"$.{key}")
                    params.append(json.dumps(value) if isinstance(value, (dict, list)) else value)

            rows = self.db.execute(select_query, params).fetchall()
            updated = 0

            for row in rows:
                existing_metadata = {}
                if row["metadata"]:
                    try:
                        existing_metadata = json.loads(row["metadata"])
                    except json.JSONDecodeError:
                        pass

                merged_metadata = {**existing_metadata, **new_metadata}
                update_params = [json.dumps(merged_metadata)]

                update_query = "UPDATE docs SET metadata = ?"
                if new_source is not None:
                    update_query += ", source = ?"
                    update_params.append(new_source)

                update_query += " WHERE id = ?"
                update_params.append(row["id"])

                self.db.execute(update_query, update_params)
                updated += 1

            self.db.commit()
            return updated

        # Simple update without merge
        updates = []
        update_params = []

        if new_source is not None:
            updates.append("source = ?")
            update_params.append(new_source)

        if new_metadata is not None:
            updates.append("metadata = ?")
            update_params.append(json.dumps(new_metadata))

        query = f"UPDATE docs SET {', '.join(updates)} WHERE 1=1"

        if source is not None:
            query += " AND source = ?"
            update_params.append(source)

        if metadata_filter:
            for key, value in metadata_filter.items():
                query += " AND json_extract(metadata, ?) = ?"
                update_params.append(f"$.{key}")
                update_params.append(json.dumps(value) if isinstance(value, (dict, list)) else value)

        cursor = self.db.execute(query, update_params)
        updated = cursor.rowcount
        self.db.commit()
        return updated

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
