"""
PostgreSQL backend implementation for the Knowledge Base.

Uses PostgreSQL with tsvector/tsquery for full-text search and array storage for embeddings.
Requires psycopg2 (or psycopg2-binary) to be installed.
"""

import json
import re
from urllib.parse import urlparse, parse_qs

from .base import BaseBackend, BackendConfig, Document, SearchResult

try:
    import psycopg2
    import psycopg2.extras
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False


class PostgresBackend(BaseBackend):
    """PostgreSQL backend using tsvector for full-text search.

    Features:
    - PostgreSQL tsvector/tsquery with GIN index for keyword search
    - Native array storage for embeddings
    - Full ACID transaction support
    - Connection pooling ready (use external pooler like pgbouncer)

    Connection string format:
        postgresql://user:password@host:port/database
        or
        host=localhost port=5432 dbname=kb user=postgres password=secret
    """

    def __init__(self, config: BackendConfig):
        """Initialize PostgreSQL backend.

        Args:
            config: BackendConfig with connection_string for PostgreSQL
                   Options can include:
                   - schema: Schema name (default: 'public')
                   - table_prefix: Prefix for table names (default: 'kb_')

        Raises:
            ImportError: If psycopg2 is not installed
            psycopg2.Error: If connection fails
        """
        if not PSYCOPG2_AVAILABLE:
            raise ImportError(
                "psycopg2 is required for PostgreSQL backend. "
                "Install with: pip install psycopg2-binary"
            )

        self._connection_string = config.connection_string
        self.schema = config.options.get("schema", "public")
        self.table_prefix = config.options.get("table_prefix", "kb_")
        self._safe_connection_info = self._make_safe_connection_info(config.connection_string)

        self._init_db()

    def _make_safe_connection_info(self, conn_str: str) -> str:
        """Create a safe connection info string without password."""
        if conn_str.startswith("postgresql://") or conn_str.startswith("postgres://"):
            parsed = urlparse(conn_str)
            safe = f"{parsed.scheme}://{parsed.username}@{parsed.hostname}"
            if parsed.port:
                safe += f":{parsed.port}"
            safe += parsed.path
            return safe
        else:
            # DSN format - remove password
            parts = []
            for part in conn_str.split():
                if not part.startswith("password="):
                    parts.append(part)
            return " ".join(parts)

    @property
    def _table_docs(self) -> str:
        """Get the fully qualified docs table name."""
        return f"{self.schema}.{self.table_prefix}docs"

    def _init_db(self) -> None:
        """Initialize database schema."""
        self.conn = psycopg2.connect(self._connection_string)
        self.conn.autocommit = False

        with self.conn.cursor() as cur:
            # Create schema if needed
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")

            # Create main documents table with tsvector column
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._table_docs} (
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    source TEXT DEFAULT 'unknown',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    embedding DOUBLE PRECISION[],
                    embedding_model TEXT,
                    content_tsv TSVECTOR GENERATED ALWAYS AS (
                        to_tsvector('english', content)
                    ) STORED
                )
            """)

            # Create GIN index for full-text search
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.table_prefix}docs_fts_idx
                ON {self._table_docs} USING GIN (content_tsv)
            """)

            # Create index on source for filtering
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.table_prefix}docs_source_idx
                ON {self._table_docs} (source)
            """)

            # Create index on embedding_model for stats
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.table_prefix}docs_model_idx
                ON {self._table_docs} (embedding_model)
            """)

        self.conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
            self.conn = None

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
        with self.conn.cursor() as cur:
            cur.execute(f"""
                INSERT INTO {self._table_docs} (content, source, embedding, embedding_model)
                VALUES (%s, %s, %s, %s)
                RETURNING id
            """, (content, source, embedding, embedding_model))
            doc_id = cur.fetchone()[0]

        self.conn.commit()
        return doc_id

    def add_batch(self, documents: list[dict]) -> list[int]:
        """Add multiple documents in a batch."""
        ids = []
        with self.conn.cursor() as cur:
            for doc in documents:
                content = doc.get("content", "")
                if not content:
                    continue
                source = doc.get("source", "unknown")
                embedding = doc.get("embedding")
                embedding_model = doc.get("embedding_model")

                cur.execute(f"""
                    INSERT INTO {self._table_docs} (content, source, embedding, embedding_model)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                """, (content, source, embedding, embedding_model))
                ids.append(cur.fetchone()[0])

        self.conn.commit()
        return ids

    def get(self, doc_id: int) -> Document | None:
        """Retrieve a document by ID."""
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(f"""
                SELECT id, content, source, created_at, embedding_model
                FROM {self._table_docs}
                WHERE id = %s
            """, (doc_id,))
            result = cur.fetchone()

        if result:
            return Document(
                id=result["id"],
                content=result["content"],
                source=result["source"],
                created_at=str(result["created_at"]) if result["created_at"] else None,
                embedding_model=result["embedding_model"],
            )
        return None

    def delete(self, doc_id: int) -> bool:
        """Delete a document by ID."""
        with self.conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self._table_docs} WHERE id = %s", (doc_id,))
            deleted = cur.rowcount > 0

        self.conn.commit()
        return deleted

    def count(self) -> int:
        """Get total document count."""
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self._table_docs}")
            return cur.fetchone()[0]

    # -------------------------------------------------------------------------
    # Search Operations
    # -------------------------------------------------------------------------

    def search_keyword(self, query: str, limit: int = 5) -> list[SearchResult]:
        """Full-text keyword search using tsvector/tsquery."""
        words = query.split()
        if not words:
            return []

        # Build tsquery with OR operator
        # Escape special characters and join with |
        safe_words = []
        for word in words:
            # Remove special characters that could break tsquery
            clean_word = re.sub(r'[^\w\s]', '', word)
            if clean_word:
                safe_words.append(clean_word)

        if not safe_words:
            return []

        tsquery = " | ".join(safe_words)

        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(f"""
                SELECT id, content, source,
                       ts_rank(content_tsv, to_tsquery('english', %s)) as rank
                FROM {self._table_docs}
                WHERE content_tsv @@ to_tsquery('english', %s)
                ORDER BY rank DESC
                LIMIT %s
            """, (tsquery, tsquery, limit))
            results = cur.fetchall()

        return [
            SearchResult(
                id=r["id"],
                content=r["content"],
                source=r["source"],
                score=float(r["rank"])
            )
            for r in results
        ]

    def get_all_embeddings(self) -> list[tuple[int, str, str, list[float]]]:
        """Retrieve all documents with embeddings for semantic search."""
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(f"""
                SELECT id, content, source, embedding
                FROM {self._table_docs}
                WHERE embedding IS NOT NULL
            """)
            docs = cur.fetchall()

        return [
            (doc["id"], doc["content"], doc["source"], list(doc["embedding"]))
            for doc in docs
            if doc["embedding"] is not None
        ]

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
        with self.conn.cursor() as cur:
            cur.execute(f"""
                UPDATE {self._table_docs}
                SET embedding = %s, embedding_model = %s
                WHERE id = %s
            """, (embedding, embedding_model, doc_id))
            updated = cur.rowcount > 0

        self.conn.commit()
        return updated

    def get_all_documents(self) -> list[tuple[int, str]]:
        """Get all document IDs and content for re-embedding."""
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT id, content FROM {self._table_docs}")
            return cur.fetchall()

    # -------------------------------------------------------------------------
    # Source Management
    # -------------------------------------------------------------------------

    def list_sources(self) -> list[dict]:
        """List all sources with document counts."""
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(f"""
                SELECT source, COUNT(*) as count
                FROM {self._table_docs}
                GROUP BY source
                ORDER BY count DESC
            """)
            results = cur.fetchall()

        return [{"source": r["source"], "count": r["count"]} for r in results]

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def get_embedding_stats(self) -> list[dict]:
        """Get statistics about embeddings by model."""
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(f"""
                SELECT embedding_model as model, COUNT(*) as count
                FROM {self._table_docs}
                WHERE embedding_model IS NOT NULL
                GROUP BY embedding_model
            """)
            results = cur.fetchall()

        return [{"model": r["model"], "count": r["count"]} for r in results]

    # -------------------------------------------------------------------------
    # Transaction Support
    # -------------------------------------------------------------------------

    def begin_transaction(self) -> None:
        """Begin a database transaction (PostgreSQL uses implicit transactions)."""
        # PostgreSQL with autocommit=False already uses transactions
        pass

    def commit(self) -> None:
        """Commit the current transaction."""
        self.conn.commit()

    def rollback(self) -> None:
        """Rollback the current transaction."""
        self.conn.rollback()

    # -------------------------------------------------------------------------
    # Backend Info
    # -------------------------------------------------------------------------

    @property
    def backend_type(self) -> str:
        """Return the backend type identifier."""
        return "postgres"

    @property
    def connection_info(self) -> str:
        """Return a safe connection info string without password."""
        return self._safe_connection_info
