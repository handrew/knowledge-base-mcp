"""
Tests for the PostgreSQL backend implementation.

These tests require a running PostgreSQL instance.
Configure via environment variables:
    TEST_POSTGRES_URL=postgresql://user:pass@localhost:5432/test_kb

Skip tests if PostgreSQL is not available with:
    pytest tests/test_postgres_backend.py -v --skip-postgres

Run with: pytest tests/test_postgres_backend.py -v
"""

import os
import uuid
import pytest

# Check if psycopg2 is available
try:
    import psycopg2
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

from backends import BackendConfig

# Import PostgresBackend only if available
if PSYCOPG2_AVAILABLE:
    from backends import PostgresBackend
    from backends.base import Document, SearchResult


# Get test database URL from environment
TEST_POSTGRES_URL = os.environ.get(
    "TEST_POSTGRES_URL",
    "postgresql://postgres:postgres@localhost:5432/test_kb"
)


def postgres_available() -> bool:
    """Check if PostgreSQL is available for testing."""
    if not PSYCOPG2_AVAILABLE:
        return False

    try:
        conn = psycopg2.connect(TEST_POSTGRES_URL)
        conn.close()
        return True
    except Exception:
        return False


# Skip all tests in this file if PostgreSQL is not available
pytestmark = pytest.mark.skipif(
    not postgres_available(),
    reason="PostgreSQL not available (set TEST_POSTGRES_URL or start PostgreSQL)"
)


@pytest.fixture(scope="function")
def backend():
    """Create a PostgreSQL backend with unique schema for test isolation."""
    if not PSYCOPG2_AVAILABLE:
        pytest.skip("psycopg2 not installed")

    # Use unique schema per test for isolation
    test_schema = f"test_{uuid.uuid4().hex[:8]}"

    config = BackendConfig(
        backend_type="postgres",
        connection_string=TEST_POSTGRES_URL,
        options={"schema": test_schema, "table_prefix": "kb_"}
    )

    backend = PostgresBackend(config)
    yield backend

    # Cleanup: drop test schema
    try:
        with backend.conn.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {test_schema} CASCADE")
        backend.conn.commit()
    except Exception:
        pass
    finally:
        backend.close()


class TestPostgresBackendInit:
    """Test PostgreSQL backend initialization."""

    def test_backend_type(self, backend):
        """Test backend type property."""
        assert backend.backend_type == "postgres"

    def test_connection_info_hides_password(self, backend):
        """Test that connection info hides password."""
        info = backend.connection_info
        assert "password" not in info.lower()
        # Should still contain host/db info
        assert "postgres" in info.lower() or "localhost" in info.lower()

    def test_init_creates_table(self, backend):
        """Test that initialization creates the docs table."""
        with backend.conn.cursor() as cur:
            cur.execute(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = %s AND table_name = %s
                )
            """, (backend.schema, f"{backend.table_prefix}docs"))
            exists = cur.fetchone()[0]
        assert exists is True

    def test_init_creates_indexes(self, backend):
        """Test that initialization creates necessary indexes."""
        with backend.conn.cursor() as cur:
            cur.execute("""
                SELECT indexname FROM pg_indexes
                WHERE schemaname = %s
            """, (backend.schema,))
            indexes = [row[0] for row in cur.fetchall()]

        # Check for FTS index
        assert any("fts" in idx for idx in indexes)


class TestPostgresBackendCRUD:
    """Test CRUD operations."""

    def test_add_document(self, backend):
        """Test adding a document."""
        embedding = [0.1, 0.2, 0.3]
        doc_id = backend.add(
            content="Test content for PostgreSQL",
            source="test_source",
            embedding=embedding,
            embedding_model="test-model"
        )
        assert doc_id >= 1
        assert backend.count() == 1

    def test_add_document_without_embedding(self, backend):
        """Test adding document without embedding."""
        doc_id = backend.add(
            content="Test content",
            source="test",
            embedding=None,
            embedding_model=None
        )
        assert doc_id >= 1

    def test_add_batch(self, backend):
        """Test batch adding documents."""
        docs = [
            {"content": "Doc 1", "source": "batch", "embedding": [0.1, 0.2], "embedding_model": "m1"},
            {"content": "Doc 2", "source": "batch"},
            {"content": "Doc 3"},
        ]
        ids = backend.add_batch(docs)
        assert len(ids) == 3
        assert backend.count() == 3

    def test_add_batch_skips_empty(self, backend):
        """Test batch add skips empty content."""
        docs = [
            {"content": "", "source": "empty"},
            {"content": "Valid", "source": "valid"},
        ]
        ids = backend.add_batch(docs)
        assert len(ids) == 1

    def test_get_document(self, backend):
        """Test retrieving a document."""
        doc_id = backend.add("Test content", "test_source", [0.1], "model")

        doc = backend.get(doc_id)
        assert doc is not None
        assert isinstance(doc, Document)
        assert doc.id == doc_id
        assert doc.content == "Test content"
        assert doc.source == "test_source"
        assert doc.embedding_model == "model"

    def test_get_nonexistent(self, backend):
        """Test getting non-existent document."""
        doc = backend.get(99999)
        assert doc is None

    def test_delete_document(self, backend):
        """Test deleting a document."""
        doc_id = backend.add("To delete", "test", None, None)
        assert backend.count() == 1

        success = backend.delete(doc_id)
        assert success is True
        assert backend.count() == 0
        assert backend.get(doc_id) is None

    def test_delete_nonexistent(self, backend):
        """Test deleting non-existent document."""
        success = backend.delete(99999)
        assert success is False

    def test_count(self, backend):
        """Test document count."""
        assert backend.count() == 0

        backend.add("Doc 1", "test", None, None)
        assert backend.count() == 1

        backend.add("Doc 2", "test", None, None)
        assert backend.count() == 2


class TestPostgresBackendSearch:
    """Test search operations."""

    @pytest.fixture
    def backend_with_docs(self, backend):
        """Backend preloaded with test documents."""
        backend.add("Python is a programming language known for readability", "python", None, None)
        backend.add("Java is a statically typed programming language", "java", None, None)
        backend.add("PostgreSQL is a powerful relational database system", "db", None, None)
        backend.add("SQLite is a lightweight embedded database engine", "db", None, None)
        backend.add("Machine learning enables pattern recognition", "ml", None, None)
        return backend

    def test_keyword_search_basic(self, backend_with_docs):
        """Test basic keyword search."""
        results = backend_with_docs.search_keyword("programming", limit=5)
        assert len(results) >= 1
        assert all(isinstance(r, SearchResult) for r in results)

    def test_keyword_search_database(self, backend_with_docs):
        """Test keyword search for database-related content."""
        results = backend_with_docs.search_keyword("database", limit=5)
        assert len(results) >= 1
        # Should find PostgreSQL and/or SQLite docs
        contents = [r.content.lower() for r in results]
        assert any("postgresql" in c or "sqlite" in c for c in contents)

    def test_keyword_search_no_results(self, backend_with_docs):
        """Test keyword search with no matches."""
        results = backend_with_docs.search_keyword("xyznonexistent", limit=5)
        assert len(results) == 0

    def test_keyword_search_empty_query(self, backend_with_docs):
        """Test keyword search with empty query."""
        results = backend_with_docs.search_keyword("", limit=5)
        assert len(results) == 0

    def test_keyword_search_special_chars(self, backend_with_docs):
        """Test keyword search handles special characters."""
        # Should not crash
        results = backend_with_docs.search_keyword("test's special! chars?", limit=5)
        assert isinstance(results, list)

    def test_keyword_search_respects_limit(self, backend_with_docs):
        """Test that limit is respected."""
        results = backend_with_docs.search_keyword("language", limit=1)
        assert len(results) <= 1

    def test_keyword_search_scores(self, backend_with_docs):
        """Test that scores are meaningful."""
        results = backend_with_docs.search_keyword("programming language", limit=5)
        if results:
            assert all(r.score >= 0 for r in results)
            # Results should be ordered by score (descending)
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)


class TestPostgresBackendEmbeddings:
    """Test embedding storage and retrieval."""

    def test_get_all_embeddings(self, backend):
        """Test retrieving all embeddings."""
        backend.add("Doc 1", "test", [0.1, 0.2, 0.3], "model")
        backend.add("Doc 2", "test", [0.4, 0.5, 0.6], "model")
        backend.add("Doc 3 no embedding", "test", None, None)

        embeddings = backend.get_all_embeddings()
        assert len(embeddings) == 2

        for doc_id, content, source, embedding in embeddings:
            assert isinstance(doc_id, int)
            assert isinstance(content, str)
            assert isinstance(embedding, list)
            assert len(embedding) == 3

    def test_get_all_embeddings_empty(self, backend):
        """Test get_all_embeddings on empty database."""
        embeddings = backend.get_all_embeddings()
        assert embeddings == []

    def test_update_embedding(self, backend):
        """Test updating embedding."""
        doc_id = backend.add("Test", "test", [0.1, 0.2], "model-1")

        success = backend.update_embedding(doc_id, [0.9, 0.8, 0.7], "model-2")
        assert success is True

        # Verify
        embeddings = backend.get_all_embeddings()
        assert len(embeddings) == 1
        _, _, _, embedding = embeddings[0]
        assert len(embedding) == 3
        assert embedding[0] == pytest.approx(0.9, abs=1e-6)

    def test_update_embedding_nonexistent(self, backend):
        """Test updating non-existent document."""
        success = backend.update_embedding(99999, [0.1], "model")
        assert success is False

    def test_get_all_documents(self, backend):
        """Test getting all documents for re-embedding."""
        backend.add("Doc 1", "test", [0.1], "m1")
        backend.add("Doc 2", "test", None, None)

        docs = backend.get_all_documents()
        assert len(docs) == 2
        for doc_id, content in docs:
            assert isinstance(doc_id, int)
            assert isinstance(content, str)

    def test_high_dimensional_embedding(self, backend):
        """Test 1024-dimensional embeddings."""
        embedding = [0.001 * i for i in range(1024)]
        doc_id = backend.add("Test", "test", embedding, "high-dim")

        embeddings = backend.get_all_embeddings()
        assert len(embeddings) == 1
        _, _, _, stored = embeddings[0]
        assert len(stored) == 1024
        assert stored[0] == pytest.approx(0.0, abs=1e-6)
        assert stored[1023] == pytest.approx(1.023, abs=1e-6)


class TestPostgresBackendSources:
    """Test source management."""

    def test_list_sources(self, backend):
        """Test listing sources."""
        backend.add("Doc 1", "source_a", None, None)
        backend.add("Doc 2", "source_a", None, None)
        backend.add("Doc 3", "source_b", None, None)

        sources = backend.list_sources()
        assert len(sources) == 2

        source_dict = {s["source"]: s["count"] for s in sources}
        assert source_dict["source_a"] == 2
        assert source_dict["source_b"] == 1

    def test_list_sources_empty(self, backend):
        """Test listing sources on empty database."""
        sources = backend.list_sources()
        assert sources == []


class TestPostgresBackendStats:
    """Test statistics."""

    def test_get_embedding_stats(self, backend):
        """Test embedding statistics."""
        backend.add("Doc 1", "test", [0.1], "model-a")
        backend.add("Doc 2", "test", [0.2], "model-a")
        backend.add("Doc 3", "test", [0.3], "model-b")
        backend.add("Doc 4", "test", None, None)

        stats = backend.get_embedding_stats()
        assert len(stats) == 2

        stats_dict = {s["model"]: s["count"] for s in stats}
        assert stats_dict["model-a"] == 2
        assert stats_dict["model-b"] == 1


class TestPostgresBackendTransactions:
    """Test transaction support."""

    def test_commit(self, backend):
        """Test explicit commit."""
        backend.add("Test", "test", None, None)
        backend.commit()
        assert backend.count() == 1

    def test_rollback(self, backend):
        """Test rollback."""
        backend.add("Initial", "test", None, None)
        backend.commit()

        # Start new work
        backend.add("To rollback", "test", None, None)
        backend.rollback()

        # Only initial doc should remain
        assert backend.count() == 1


class TestPostgresBackendEdgeCases:
    """Test edge cases."""

    def test_unicode_content(self, backend):
        """Test unicode content."""
        content = "Unicode: 你好世界 🚀 émojis ñ"
        doc_id = backend.add(content, "unicode", None, None)
        doc = backend.get(doc_id)
        assert doc.content == content

    def test_long_content(self, backend):
        """Test long content."""
        content = "A" * 100000
        doc_id = backend.add(content, "long", None, None)
        doc = backend.get(doc_id)
        assert doc.content == content

    def test_special_chars_in_source(self, backend):
        """Test special characters in source."""
        doc_id = backend.add("Test", "path/to/file.txt", None, None)
        doc = backend.get(doc_id)
        assert doc.source == "path/to/file.txt"

    def test_newlines_in_content(self, backend):
        """Test content with newlines."""
        content = "Line 1\nLine 2\n\nLine 3"
        doc_id = backend.add(content, "test", None, None)
        doc = backend.get(doc_id)
        assert doc.content == content

    def test_empty_embedding_list(self, backend):
        """Test with empty embedding list."""
        # This might be an edge case depending on usage
        doc_id = backend.add("Test", "test", [], "model")
        embeddings = backend.get_all_embeddings()
        assert len(embeddings) == 1
        _, _, _, emb = embeddings[0]
        assert emb == []


class TestPostgresBackendMissingDependency:
    """Test behavior when psycopg2 is not installed."""

    def test_import_error_message(self):
        """Test that helpful error is raised without psycopg2."""
        # This test always passes if psycopg2 is installed
        # It's mainly for documentation purposes
        if PSYCOPG2_AVAILABLE:
            pytest.skip("psycopg2 is installed")

        from backends.postgres_backend import PSYCOPG2_AVAILABLE as pg_avail
        assert pg_avail is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
