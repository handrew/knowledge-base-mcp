"""
Tests for the backend abstraction layer.

Tests the base classes, factory functions, and backend implementations.
Run with: pytest tests/test_backends.py -v
"""

import os
import tempfile
import pytest
from pathlib import Path

from backends import (
    BaseBackend,
    BackendConfig,
    SQLiteBackend,
    create_backend,
    create_backend_from_url,
    get_available_backends,
)
from backends.base import Document, SearchResult
from backends.factory import register_backend, _BACKENDS


class TestBackendConfig:
    """Test BackendConfig dataclass."""

    def test_basic_config(self):
        """Test creating a basic config."""
        config = BackendConfig(
            backend_type="sqlite",
            connection_string="/path/to/db.sqlite"
        )
        assert config.backend_type == "sqlite"
        assert config.connection_string == "/path/to/db.sqlite"
        assert config.options == {}

    def test_config_with_options(self):
        """Test config with options."""
        config = BackendConfig(
            backend_type="postgres",
            connection_string="postgresql://localhost/kb",
            options={"schema": "my_schema", "pool_size": 5}
        )
        assert config.options["schema"] == "my_schema"
        assert config.options["pool_size"] == 5


class TestDocument:
    """Test Document dataclass."""

    def test_document_creation(self):
        """Test creating a document."""
        doc = Document(
            id=1,
            content="Test content",
            source="test_source",
            created_at="2024-01-01 00:00:00",
            embedding_model="test-model"
        )
        assert doc.id == 1
        assert doc.content == "Test content"
        assert doc.source == "test_source"

    def test_document_to_dict(self):
        """Test document serialization."""
        doc = Document(
            id=1,
            content="Test content",
            source="test_source",
            created_at="2024-01-01 00:00:00",
            embedding_model="test-model"
        )
        d = doc.to_dict()
        assert d["id"] == 1
        assert d["content"] == "Test content"
        assert d["source"] == "test_source"
        assert d["embedding_model"] == "test-model"
        assert "embedding" not in d  # Embedding is not included in to_dict

    def test_document_optional_fields(self):
        """Test document with optional fields as None."""
        doc = Document(id=1, content="Test", source="src")
        assert doc.created_at is None
        assert doc.embedding is None
        assert doc.embedding_model is None


class TestSearchResult:
    """Test SearchResult dataclass."""

    def test_search_result_creation(self):
        """Test creating a search result."""
        result = SearchResult(
            id=1,
            content="Test content",
            source="test_source",
            score=0.95
        )
        assert result.id == 1
        assert result.score == 0.95

    def test_search_result_to_dict(self):
        """Test search result serialization."""
        result = SearchResult(
            id=1,
            content="Test content",
            source="test_source",
            score=0.95
        )
        d = result.to_dict()
        assert d["id"] == 1
        assert d["content"] == "Test content"
        assert d["score"] == 0.95


class TestBackendFactory:
    """Test backend factory functions."""

    def test_create_sqlite_backend(self):
        """Test creating SQLite backend via factory."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            config = BackendConfig(
                backend_type="sqlite",
                connection_string=db_path
            )
            backend = create_backend(config)
            assert isinstance(backend, SQLiteBackend)
            assert backend.backend_type == "sqlite"
            backend.close()
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_create_backend_unknown_type(self):
        """Test factory with unknown backend type."""
        config = BackendConfig(
            backend_type="unknown_db",
            connection_string="whatever"
        )
        with pytest.raises(ValueError) as exc_info:
            create_backend(config)
        assert "Unknown backend type" in str(exc_info.value)

    def test_create_backend_from_url_sqlite_path(self):
        """Test creating backend from SQLite file path."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            backend = create_backend_from_url(db_path)
            assert isinstance(backend, SQLiteBackend)
            backend.close()
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_create_backend_from_url_sqlite_scheme(self):
        """Test creating backend from sqlite:// URL."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            backend = create_backend_from_url(f"sqlite:///{db_path}")
            assert isinstance(backend, SQLiteBackend)
            backend.close()
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_create_backend_from_url_with_extensions(self):
        """Test URL parsing with various SQLite extensions."""
        for ext in [".db", ".sqlite", ".sqlite3"]:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                db_path = f.name

            try:
                backend = create_backend_from_url(db_path)
                assert backend.backend_type == "sqlite"
                backend.close()
            finally:
                if os.path.exists(db_path):
                    os.remove(db_path)

    def test_create_backend_from_url_unknown(self):
        """Test URL parsing with unrecognized format."""
        with pytest.raises(ValueError) as exc_info:
            create_backend_from_url("unknown_format")
        assert "Could not determine backend type" in str(exc_info.value)

    def test_get_available_backends(self):
        """Test listing available backends."""
        backends = get_available_backends()
        assert len(backends) >= 1

        # SQLite should always be available
        sqlite_backend = next((b for b in backends if b["name"] == "sqlite"), None)
        assert sqlite_backend is not None
        assert sqlite_backend["available"] is True

        # PostgreSQL entry should exist (available or not)
        postgres_backend = next((b for b in backends if b["name"] == "postgres"), None)
        assert postgres_backend is not None

    def test_register_custom_backend(self):
        """Test registering a custom backend."""
        class MockBackend(BaseBackend):
            def __init__(self, config):
                self._config = config

            def close(self):
                pass

            def add(self, content, source, embedding, embedding_model):
                return 1

            def add_batch(self, documents):
                return [1]

            def get(self, doc_id):
                return None

            def delete(self, doc_id):
                return False

            def count(self):
                return 0

            def search_keyword(self, query, limit=5):
                return []

            def get_all_embeddings(self):
                return []

            def update_embedding(self, doc_id, embedding, embedding_model):
                return False

            def get_all_documents(self):
                return []

            def list_sources(self):
                return []

            def get_embedding_stats(self):
                return []

            def update(self, doc_id, content=None, source=None, embedding=None, embedding_model=None, metadata=None, expires_at=None):
                return False

            def list_documents(self, source=None, metadata_filter=None, limit=100, offset=0):
                return []

            def cleanup_expired(self):
                return 0

            def find_duplicate(self, content):
                return None

            def delete_by_filter(self, source=None, metadata_filter=None):
                return 0

            def update_by_filter(self, source=None, metadata_filter=None, new_source=None, new_metadata=None, metadata_merge=False):
                return 0

            def commit(self):
                pass

            def rollback(self):
                pass

            @property
            def backend_type(self):
                return "mock"

            @property
            def connection_info(self):
                return "mock://test"

        # Register the mock backend
        register_backend("mock", MockBackend)

        # Verify it's registered
        assert "mock" in _BACKENDS

        # Create instance via factory
        config = BackendConfig(backend_type="mock", connection_string="test")
        backend = create_backend(config)
        assert backend.backend_type == "mock"

        # Clean up
        del _BACKENDS["mock"]

    def test_register_invalid_backend(self):
        """Test registering a non-BaseBackend class."""
        class NotABackend:
            pass

        with pytest.raises(TypeError):
            register_backend("invalid", NotABackend)


class TestSQLiteBackend:
    """Test SQLite backend implementation."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        yield db_path
        # Cleanup
        for suffix in ["", "-shm", "-wal"]:
            path = db_path + suffix
            if os.path.exists(path):
                os.remove(path)

    @pytest.fixture
    def backend(self, temp_db):
        """Create a SQLite backend instance."""
        config = BackendConfig(backend_type="sqlite", connection_string=temp_db)
        backend = SQLiteBackend(config)
        yield backend
        backend.close()

    # -------------------------------------------------------------------------
    # CRUD Tests
    # -------------------------------------------------------------------------

    def test_add_document(self, backend):
        """Test adding a document."""
        embedding = [0.1, 0.2, 0.3]
        doc_id = backend.add(
            content="Test content",
            source="test_source",
            embedding=embedding,
            embedding_model="test-model"
        )
        assert doc_id == 1
        assert backend.count() == 1

    def test_add_document_without_embedding(self, backend):
        """Test adding a document without embedding."""
        doc_id = backend.add(
            content="Test content",
            source="test_source",
            embedding=None,
            embedding_model=None
        )
        assert doc_id >= 1

    def test_add_batch(self, backend):
        """Test batch adding documents."""
        docs = [
            {"content": "Doc 1", "source": "batch", "embedding": [0.1], "embedding_model": "m1"},
            {"content": "Doc 2", "source": "batch", "embedding": [0.2], "embedding_model": "m1"},
            {"content": "Doc 3"},  # Minimal doc
        ]
        ids = backend.add_batch(docs)
        assert len(ids) == 3
        assert backend.count() == 3

    def test_add_batch_skips_empty_content(self, backend):
        """Test that batch add skips empty content."""
        docs = [
            {"content": "", "source": "empty"},
            {"content": "Valid", "source": "valid"},
        ]
        ids = backend.add_batch(docs)
        assert len(ids) == 1

    def test_get_document(self, backend):
        """Test retrieving a document."""
        doc_id = backend.add(
            content="Test content",
            source="test_source",
            embedding=[0.1, 0.2],
            embedding_model="test-model"
        )

        doc = backend.get(doc_id)
        assert doc is not None
        assert doc.id == doc_id
        assert doc.content == "Test content"
        assert doc.source == "test_source"
        assert doc.embedding_model == "test-model"

    def test_get_nonexistent_document(self, backend):
        """Test retrieving a non-existent document."""
        doc = backend.get(9999)
        assert doc is None

    def test_delete_document(self, backend):
        """Test deleting a document."""
        doc_id = backend.add("To delete", "test", None, None)
        assert backend.count() == 1

        success = backend.delete(doc_id)
        assert success is True
        assert backend.count() == 0

    def test_delete_nonexistent_document(self, backend):
        """Test deleting a non-existent document."""
        success = backend.delete(9999)
        assert success is False

    def test_count(self, backend):
        """Test document count."""
        assert backend.count() == 0

        backend.add("Doc 1", "test", None, None)
        assert backend.count() == 1

        backend.add("Doc 2", "test", None, None)
        assert backend.count() == 2

    # -------------------------------------------------------------------------
    # Update Tests
    # -------------------------------------------------------------------------

    def test_update_content(self, backend):
        """Test updating document content."""
        doc_id = backend.add("Original content", "test", [0.1, 0.2], "model1")

        success = backend.update(doc_id, content="Updated content")
        assert success is True

        doc = backend.get(doc_id)
        assert doc.content == "Updated content"
        assert doc.source == "test"  # Unchanged

    def test_update_source(self, backend):
        """Test updating document source."""
        doc_id = backend.add("Test content", "original_source", None, None)

        success = backend.update(doc_id, source="new_source")
        assert success is True

        doc = backend.get(doc_id)
        assert doc.content == "Test content"  # Unchanged
        assert doc.source == "new_source"

    def test_update_content_and_source(self, backend):
        """Test updating both content and source."""
        doc_id = backend.add("Original", "old_source", None, None)

        success = backend.update(doc_id, content="New content", source="new_source")
        assert success is True

        doc = backend.get(doc_id)
        assert doc.content == "New content"
        assert doc.source == "new_source"

    def test_update_embedding(self, backend):
        """Test updating document embedding."""
        doc_id = backend.add("Test", "test", [0.1, 0.2], "model1")

        success = backend.update(doc_id, embedding=[0.9, 0.8, 0.7], embedding_model="model2")
        assert success is True

        # Verify embedding was updated
        embeddings = backend.get_all_embeddings()
        assert len(embeddings) == 1
        _, _, _, emb = embeddings[0]
        assert len(emb) == 3
        assert emb[0] == pytest.approx(0.9, abs=1e-6)

    def test_update_nonexistent_document(self, backend):
        """Test updating a non-existent document."""
        success = backend.update(9999, content="New content")
        assert success is False

    def test_update_no_fields(self, backend):
        """Test update with no fields specified."""
        doc_id = backend.add("Test", "test", None, None)

        # Should return True (doc exists) but not change anything
        success = backend.update(doc_id)
        assert success is True

        doc = backend.get(doc_id)
        assert doc.content == "Test"

    def test_update_preserves_other_fields(self, backend):
        """Test that update preserves fields not being updated."""
        doc_id = backend.add("Content", "source", [0.1, 0.2], "model")

        # Update only content
        backend.update(doc_id, content="New content")

        doc = backend.get(doc_id)
        assert doc.content == "New content"
        assert doc.source == "source"
        assert doc.embedding_model == "model"

        # Verify embedding preserved
        embeddings = backend.get_all_embeddings()
        assert len(embeddings) == 1

    def test_update_fts_sync(self, backend):
        """Test that FTS index is updated when content changes."""
        doc_id = backend.add("Python programming language", "test", None, None)

        # Should find it with original content
        results = backend.search_keyword("Python", limit=5)
        assert len(results) == 1

        # Update content
        backend.update(doc_id, content="Java programming language")

        # Should no longer find "Python"
        results = backend.search_keyword("Python", limit=5)
        assert len(results) == 0

        # Should find "Java"
        results = backend.search_keyword("Java", limit=5)
        assert len(results) == 1

    # -------------------------------------------------------------------------
    # Keyword Search Tests
    # -------------------------------------------------------------------------

    def test_keyword_search_basic(self, backend):
        """Test basic keyword search."""
        backend.add("Python is a programming language", "python", None, None)
        backend.add("Java is another programming language", "java", None, None)
        backend.add("SQLite is a database", "db", None, None)

        results = backend.search_keyword("programming", limit=5)
        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)
        assert all("programming" in r.content.lower() for r in results)

    def test_keyword_search_no_results(self, backend):
        """Test keyword search with no matches."""
        backend.add("Python programming", "test", None, None)

        results = backend.search_keyword("nonexistent_word_xyz", limit=5)
        assert len(results) == 0

    def test_keyword_search_empty_query(self, backend):
        """Test keyword search with empty query."""
        backend.add("Some content", "test", None, None)

        results = backend.search_keyword("", limit=5)
        assert len(results) == 0

    def test_keyword_search_special_characters(self, backend):
        """Test keyword search with special characters."""
        backend.add("Test with 'quotes' and special chars", "test", None, None)

        # Should not crash
        results = backend.search_keyword("quotes", limit=5)
        assert isinstance(results, list)

    def test_keyword_search_respects_limit(self, backend):
        """Test that keyword search respects limit."""
        for i in range(10):
            backend.add(f"Document {i} about testing", "test", None, None)

        results = backend.search_keyword("testing", limit=3)
        assert len(results) <= 3

    def test_keyword_search_scores_positive(self, backend):
        """Test that keyword search scores are positive."""
        backend.add("Python programming language", "test", None, None)
        backend.add("Python is great", "test", None, None)

        results = backend.search_keyword("Python", limit=5)
        assert len(results) >= 1
        assert all(r.score > 0 for r in results)

    # -------------------------------------------------------------------------
    # Embedding Storage Tests
    # -------------------------------------------------------------------------

    def test_get_all_embeddings(self, backend):
        """Test retrieving all embeddings."""
        backend.add("Doc 1", "test", [0.1, 0.2, 0.3], "model-1")
        backend.add("Doc 2", "test", [0.4, 0.5, 0.6], "model-1")
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
        """Test updating document embedding."""
        doc_id = backend.add("Test", "test", [0.1, 0.2], "model-1")

        success = backend.update_embedding(doc_id, [0.9, 0.8, 0.7], "model-2")
        assert success is True

        # Verify update
        embeddings = backend.get_all_embeddings()
        assert len(embeddings) == 1
        _, _, _, embedding = embeddings[0]
        assert embedding == [0.9, 0.8, 0.7]

    def test_update_embedding_nonexistent(self, backend):
        """Test updating embedding for non-existent document."""
        success = backend.update_embedding(9999, [0.1], "model")
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

    # -------------------------------------------------------------------------
    # Source Management Tests
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Statistics Tests
    # -------------------------------------------------------------------------

    def test_get_embedding_stats(self, backend):
        """Test embedding statistics."""
        backend.add("Doc 1", "test", [0.1], "model-a")
        backend.add("Doc 2", "test", [0.2], "model-a")
        backend.add("Doc 3", "test", [0.3], "model-b")
        backend.add("Doc 4 no model", "test", None, None)

        stats = backend.get_embedding_stats()
        assert len(stats) == 2

        stats_dict = {s["model"]: s["count"] for s in stats}
        assert stats_dict["model-a"] == 2
        assert stats_dict["model-b"] == 1

    # -------------------------------------------------------------------------
    # Transaction Tests
    # -------------------------------------------------------------------------

    def test_commit(self, backend):
        """Test explicit commit."""
        backend.add("Test", "test", None, None)
        backend.commit()
        assert backend.count() == 1

    def test_rollback(self, backend):
        """Test rollback."""
        backend.begin_transaction()
        backend.add("Test", "test", None, None)
        backend.rollback()
        # Note: SQLite's autocommit behavior may vary
        # This mainly tests that the method doesn't crash

    # -------------------------------------------------------------------------
    # Backend Info Tests
    # -------------------------------------------------------------------------

    def test_backend_type(self, backend):
        """Test backend type property."""
        assert backend.backend_type == "sqlite"

    def test_connection_info(self, backend, temp_db):
        """Test connection info property."""
        assert backend.connection_info == temp_db

    # -------------------------------------------------------------------------
    # Edge Cases
    # -------------------------------------------------------------------------

    def test_unicode_content(self, backend):
        """Test handling unicode content."""
        content = "Unicode test: 你好世界 🚀 émojis"
        doc_id = backend.add(content, "unicode", None, None)
        doc = backend.get(doc_id)
        assert doc.content == content

    def test_long_content(self, backend):
        """Test handling long content."""
        content = "A" * 100000  # 100KB of text
        doc_id = backend.add(content, "long", None, None)
        doc = backend.get(doc_id)
        assert doc.content == content

    def test_special_characters_in_source(self, backend):
        """Test special characters in source field."""
        doc_id = backend.add("Test", "source/with/slashes", None, None)
        doc = backend.get(doc_id)
        assert doc.source == "source/with/slashes"

    def test_high_dimensional_embedding(self, backend):
        """Test high-dimensional embeddings (like 1024-dim models)."""
        embedding = [0.001 * i for i in range(1024)]
        doc_id = backend.add("Test", "test", embedding, "high-dim-model")

        embeddings = backend.get_all_embeddings()
        assert len(embeddings) == 1
        _, _, _, stored_embedding = embeddings[0]
        assert len(stored_embedding) == 1024
        assert stored_embedding[0] == pytest.approx(0.0, abs=1e-6)
        assert stored_embedding[1023] == pytest.approx(1.023, abs=1e-6)


class TestSQLiteBackendLegacyCleanup:
    """Test SQLite backend's legacy table cleanup."""

    def test_cleanup_legacy_tables(self, tmp_path):
        """Test that legacy vec tables are cleaned up."""
        import sqlite3

        db_path = str(tmp_path / "legacy_test.db")

        # Create legacy tables manually
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE docs_vss (id INTEGER)")
        conn.execute("CREATE TABLE docs_vec (id INTEGER)")
        conn.commit()
        conn.close()

        # Create backend (should clean up legacy tables)
        config = BackendConfig(backend_type="sqlite", connection_string=db_path)
        backend = SQLiteBackend(config)

        # Verify legacy tables are gone
        result = backend.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('docs_vss', 'docs_vec')"
        ).fetchall()
        assert len(result) == 0

        backend.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
