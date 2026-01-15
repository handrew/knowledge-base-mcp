"""
Tests for the Knowledge Base module.
Run with: pytest tests/test_kb.py -v

For faster tests, use the smaller model:
    pytest tests/test_kb.py -v --fast
"""

import os
import tempfile
import pytest
from pathlib import Path

import knowledge_base as kb_module
from knowledge_base import KnowledgeBase, MODELS
from backends import BackendConfig

# Use smaller/faster model for tests
TEST_MODEL = "BAAI/bge-base-en-v1.5"  # 768 dim, much smaller than bge-m3


@pytest.fixture(scope="session")
def use_fast_model(request):
    """Session-scoped fixture to configure fast model."""
    if request.config.getoption("--fast", default=True):
        # Patch the default model for all tests
        original = kb_module.DEFAULT_MODEL
        kb_module.DEFAULT_MODEL = TEST_MODEL
        yield TEST_MODEL
        kb_module.DEFAULT_MODEL = original
    else:
        yield kb_module.DEFAULT_MODEL


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    # Cleanup
    if os.path.exists(db_path):
        os.remove(db_path)
    # Also remove any -shm and -wal files
    for suffix in ["-shm", "-wal"]:
        path = db_path + suffix
        if os.path.exists(path):
            os.remove(path)


@pytest.fixture
def kb(temp_db, use_fast_model):
    """Create a KnowledgeBase instance with temp database."""
    kb = KnowledgeBase(temp_db, model_name=use_fast_model)
    yield kb
    kb.close()


class TestBasicOperations:
    """Test basic CRUD operations."""

    def test_add_document(self, kb):
        """Test adding a single document."""
        doc_id = kb.add("This is a test document about Python programming.", "test")
        assert doc_id == 1
        assert kb.count() == 1

    def test_add_multiple_documents(self, kb):
        """Test adding multiple documents."""
        kb.add("Document one about cats.", "animals")
        kb.add("Document two about dogs.", "animals")
        kb.add("Document three about programming.", "tech")
        assert kb.count() == 3

    def test_get_document(self, kb, use_fast_model):
        """Test retrieving a document by ID."""
        doc_id = kb.add("Test content here.", "test_source")
        doc = kb.get(doc_id)
        assert doc is not None
        assert doc["content"] == "Test content here."
        assert doc["source"] == "test_source"
        assert doc["embedding_model"] == use_fast_model

    def test_get_nonexistent_document(self, kb):
        """Test retrieving a document that doesn't exist."""
        doc = kb.get(999)
        assert doc is None

    def test_delete_document(self, kb):
        """Test deleting a document."""
        doc_id = kb.add("To be deleted.", "test")
        assert kb.count() == 1
        success = kb.delete(doc_id)
        assert success is True
        assert kb.count() == 0
        assert kb.get(doc_id) is None

    def test_delete_nonexistent_document(self, kb):
        """Test deleting a document that doesn't exist."""
        success = kb.delete(999)
        assert success is False

    def test_add_batch(self, kb):
        """Test batch adding documents."""
        docs = [
            {"content": "First document", "source": "batch"},
            {"content": "Second document", "source": "batch"},
            {"content": "Third document"},  # No source, should default
        ]
        ids = kb.add_batch(docs)
        assert len(ids) == 3
        assert kb.count() == 3


class TestUpdate:
    """Test update and append functionality."""

    def test_update_content(self, kb):
        """Test updating document content."""
        doc_id = kb.add("Original content", "test")

        success = kb.update(doc_id, content="Updated content")
        assert success is True

        doc = kb.get(doc_id)
        assert doc["content"] == "Updated content"
        assert doc["source"] == "test"  # Unchanged

    def test_update_source(self, kb):
        """Test updating document source only."""
        doc_id = kb.add("Test content", "original_source")

        success = kb.update(doc_id, source="new_source")
        assert success is True

        doc = kb.get(doc_id)
        assert doc["content"] == "Test content"  # Unchanged
        assert doc["source"] == "new_source"

    def test_update_content_and_source(self, kb):
        """Test updating both content and source."""
        doc_id = kb.add("Original", "old_source")

        success = kb.update(doc_id, content="New content", source="new_source")
        assert success is True

        doc = kb.get(doc_id)
        assert doc["content"] == "New content"
        assert doc["source"] == "new_source"

    def test_update_regenerates_embedding(self, kb, use_fast_model):
        """Test that updating content regenerates embedding."""
        doc_id = kb.add("Original content about Python", "test")

        # Update with very different content
        kb.update(doc_id, content="Completely different content about databases")

        doc = kb.get(doc_id)
        assert doc["embedding_model"] == use_fast_model

        # Search should find it with new content
        results = kb.search_keyword("databases", limit=5)
        assert any(r["id"] == doc_id for r in results)

    def test_update_nonexistent_document(self, kb):
        """Test updating a non-existent document."""
        success = kb.update(9999, content="New content")
        assert success is False

    def test_append_basic(self, kb):
        """Test basic append functionality."""
        doc_id = kb.add("First part.", "test")

        success = kb.append(doc_id, "Second part.")
        assert success is True

        doc = kb.get(doc_id)
        assert doc["content"] == "First part.\n\nSecond part."

    def test_append_custom_separator(self, kb):
        """Test append with custom separator."""
        doc_id = kb.add("Line 1", "test")

        success = kb.append(doc_id, "Line 2", separator=" | ")
        assert success is True

        doc = kb.get(doc_id)
        assert doc["content"] == "Line 1 | Line 2"

    def test_append_multiple_times(self, kb):
        """Test appending multiple times."""
        doc_id = kb.add("Part 1", "test")

        kb.append(doc_id, "Part 2")
        kb.append(doc_id, "Part 3")

        doc = kb.get(doc_id)
        assert doc["content"] == "Part 1\n\nPart 2\n\nPart 3"

    def test_append_regenerates_embedding(self, kb):
        """Test that append regenerates embedding for combined content."""
        doc_id = kb.add("Python programming", "test")

        # Append related content
        kb.append(doc_id, "Also covers machine learning")

        # Should now be searchable for both terms
        results = kb.search_hybrid("machine learning", limit=5)
        assert any(r["id"] == doc_id for r in results)

    def test_append_nonexistent_document(self, kb):
        """Test appending to a non-existent document."""
        success = kb.append(9999, "New content")
        assert success is False

    def test_update_and_search(self, kb):
        """Test that updated documents are properly searchable."""
        doc_id = kb.add("Information about cats", "animals")

        # Update to completely different topic
        kb.update(doc_id, content="Information about databases and SQL")

        # Should NOT find with old content
        results = kb.search_keyword("cats", limit=5)
        assert not any(r["id"] == doc_id for r in results)

        # Should find with new content
        results = kb.search_keyword("databases", limit=5)
        assert any(r["id"] == doc_id for r in results)


class TestSearch:
    """Test search functionality."""

    @pytest.fixture
    def kb_with_docs(self, kb):
        """KB preloaded with test documents."""
        kb.add("Python is a programming language known for readability and simplicity.", "python")
        kb.add("SQLite is a lightweight embedded database engine written in C.", "database")
        kb.add("Machine learning enables computers to learn from data patterns.", "ml")
        kb.add("JavaScript runs in web browsers and on servers with Node.js.", "javascript")
        kb.add("PostgreSQL is a powerful open source relational database system.", "database")
        return kb

    def test_keyword_search(self, kb_with_docs):
        """Test FTS5 keyword search."""
        results = kb_with_docs.search_keyword("database", limit=5)
        assert len(results) >= 1
        # Should find SQLite and PostgreSQL docs
        contents = [r["content"] for r in results]
        assert any("SQLite" in c or "PostgreSQL" in c for c in contents)

    def test_keyword_search_no_results(self, kb_with_docs):
        """Test keyword search with no matches."""
        results = kb_with_docs.search_keyword("xyznonexistent", limit=5)
        assert len(results) == 0

    def test_semantic_search(self, kb_with_docs):
        """Test vector similarity search."""
        # Search for something semantically related to databases
        results = kb_with_docs.search_semantic("storing and querying data", limit=3)
        assert isinstance(results, list)
        if len(results) > 0:
            # Results should have scores
            assert all("score" in r for r in results)

    def test_hybrid_search(self, kb_with_docs):
        """Test hybrid search combining keyword and semantic."""
        results = kb_with_docs.search_hybrid("programming language", limit=3)
        assert len(results) >= 1
        # Python and JavaScript are programming languages
        contents = [r["content"].lower() for r in results]
        assert any("python" in c or "javascript" in c for c in contents)

    def test_search_limit(self, kb_with_docs):
        """Test that search respects limit parameter."""
        results = kb_with_docs.search_hybrid("software", limit=2)
        assert len(results) <= 2


class TestSources:
    """Test source tracking."""

    def test_list_sources(self, kb):
        """Test listing sources with counts."""
        kb.add("Doc 1", "source_a")
        kb.add("Doc 2", "source_a")
        kb.add("Doc 3", "source_b")

        sources = kb.list_sources()
        assert len(sources) == 2

        source_dict = {s["source"]: s["count"] for s in sources}
        assert source_dict["source_a"] == 2
        assert source_dict["source_b"] == 1

    def test_list_sources_empty(self, kb):
        """Test listing sources on empty KB."""
        sources = kb.list_sources()
        assert sources == []


class TestEmbeddingModel:
    """Test embedding model tracking and switching."""

    def test_model_stored_with_document(self, kb, use_fast_model):
        """Test that embedding model is stored with each document."""
        doc_id = kb.add("Test document", "test")
        doc = kb.get(doc_id)
        assert doc["embedding_model"] == use_fast_model

    def test_get_embedding_stats(self, kb, use_fast_model):
        """Test embedding statistics."""
        kb.add("Doc 1", "test")
        kb.add("Doc 2", "test")

        stats = kb.get_embedding_stats()
        assert stats["current_model"] == use_fast_model
        assert len(stats["models"]) == 1
        assert stats["models"][0]["model"] == use_fast_model
        assert stats["models"][0]["count"] == 2

    def test_list_available_models(self, kb):
        """Test listing available models."""
        models = kb.list_available_models()
        assert len(models) == len(MODELS)
        model_names = [m["name"] for m in models]
        assert "BAAI/bge-m3" in model_names
        assert "BAAI/bge-base-en-v1.5" in model_names


class TestReembed:
    """Test re-embedding functionality."""

    def test_reembed_same_model(self, kb, use_fast_model):
        """Test re-embedding with the same model."""
        kb.add("Test document one", "test")
        kb.add("Test document two", "test")

        result = kb.reembed(use_fast_model)
        assert result["reembedded"] == 2
        assert result["target_model"] == use_fast_model

    def test_reembed_empty_db(self, kb):
        """Test re-embedding an empty database."""
        result = kb.reembed()
        assert result["reembedded"] == 0

    def test_reembed_unknown_model(self, kb):
        """Test re-embedding with unknown model."""
        kb.add("Test", "test")
        result = kb.reembed("unknown/model")
        assert "error" in result


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_content(self, kb):
        """Test that empty content is handled in batch add."""
        docs = [
            {"content": "", "source": "empty"},
            {"content": "Valid content", "source": "valid"},
        ]
        ids = kb.add_batch(docs)
        assert len(ids) == 1  # Only valid content added

    def test_special_characters_in_content(self, kb):
        """Test content with special characters."""
        content = "Test with 'quotes' and \"double quotes\" and special chars: <>&"
        doc_id = kb.add(content, "special")
        doc = kb.get(doc_id)
        assert doc["content"] == content

    def test_unicode_content(self, kb):
        """Test content with unicode characters."""
        content = "Unicode test: 你好世界 🚀 émojis and ñ"
        doc_id = kb.add(content, "unicode")
        doc = kb.get(doc_id)
        assert doc["content"] == content

    def test_long_content(self, kb):
        """Test with longer content (multiple paragraphs)."""
        content = """
        This is the first paragraph with some information about testing.
        It contains multiple sentences to simulate real content.

        This is the second paragraph. It adds more context and detail.
        The knowledge base should handle this without issues.

        A third paragraph for good measure. This tests that we can handle
        content that spans multiple paragraphs as expected.
        """.strip()
        doc_id = kb.add(content, "long")
        doc = kb.get(doc_id)
        assert doc["content"] == content

    def test_keyword_search_special_chars(self, kb):
        """Test keyword search with special FTS5 characters."""
        kb.add("Testing quotes and 'special' characters", "test")
        # Should not crash with special characters
        results = kb.search_keyword("quotes", limit=5)
        assert len(results) >= 0  # May or may not find results, but shouldn't crash


class TestBackendType:
    """Test backend type detection and properties."""

    def test_sqlite_backend_type(self, kb):
        """Test that SQLite backend is used by default."""
        assert kb.backend_type == "sqlite"

    def test_db_path_property(self, kb, temp_db):
        """Test db_path property returns connection info."""
        assert kb.db_path == temp_db

    def test_explicit_backend_config(self, temp_db, use_fast_model):
        """Test using explicit backend configuration."""
        config = BackendConfig(
            backend_type="sqlite",
            connection_string=temp_db
        )
        kb = KnowledgeBase(backend_config=config, model_name=use_fast_model)
        assert kb.backend_type == "sqlite"
        kb.add("Test", "test")
        assert kb.count() == 1
        kb.close()


class TestClose:
    """Test resource cleanup."""

    def test_close(self, temp_db, use_fast_model):
        """Test closing the knowledge base."""
        kb = KnowledgeBase(temp_db, model_name=use_fast_model)
        kb.add("Test", "test")
        kb.close()
        # After close, operations should fail or be undefined
        # We mainly want to ensure close() doesn't crash


class TestMetadata:
    """Test metadata functionality."""

    def test_add_with_metadata(self, kb):
        """Test adding a document with metadata."""
        doc_id = kb.add("Test content", "test", metadata={"project": "foo", "private": True})
        doc = kb.get(doc_id)
        assert doc is not None
        assert doc["metadata"] == {"project": "foo", "private": True}

    def test_update_metadata(self, kb):
        """Test updating document metadata."""
        doc_id = kb.add("Test content", "test", metadata={"version": 1})
        kb.update(doc_id, metadata={"version": 2, "status": "reviewed"})
        doc = kb.get(doc_id)
        assert doc["metadata"] == {"version": 2, "status": "reviewed"}

    def test_list_documents_with_metadata_filter(self, kb):
        """Test listing documents with metadata filter."""
        kb.add("Doc 1", "test", metadata={"project": "alpha", "status": "active"})
        kb.add("Doc 2", "test", metadata={"project": "beta", "status": "active"})
        kb.add("Doc 3", "test", metadata={"project": "alpha", "status": "archived"})

        # Filter by project
        docs = kb.list_documents(metadata_filter={"project": "alpha"})
        assert len(docs) == 2
        assert all(d["metadata"]["project"] == "alpha" for d in docs)

        # Filter by multiple fields
        docs = kb.list_documents(metadata_filter={"project": "alpha", "status": "active"})
        assert len(docs) == 1

    def test_list_documents_with_source_filter(self, kb):
        """Test listing documents with source filter."""
        kb.add("Doc 1", "source_a")
        kb.add("Doc 2", "source_a")
        kb.add("Doc 3", "source_b")

        docs = kb.list_documents(source="source_a")
        assert len(docs) == 2
        assert all(d["source"] == "source_a" for d in docs)

    def test_list_documents_pagination(self, kb):
        """Test pagination in list_documents."""
        for i in range(5):
            kb.add(f"Document {i}", "test")

        # Get first 2
        docs_page1 = kb.list_documents(limit=2, offset=0)
        assert len(docs_page1) == 2

        # Get next 2
        docs_page2 = kb.list_documents(limit=2, offset=2)
        assert len(docs_page2) == 2

        # Should be different docs (ordered by id DESC)
        page1_ids = {d["id"] for d in docs_page1}
        page2_ids = {d["id"] for d in docs_page2}
        assert page1_ids.isdisjoint(page2_ids)


class TestExpiration:
    """Test document expiration functionality."""

    def test_add_with_expiration(self, kb):
        """Test adding a document with expiration."""
        doc_id = kb.add("Test content", "test", expires_at="2099-12-31T23:59:59")
        doc = kb.get(doc_id)
        assert doc is not None
        assert doc["expires_at"] == "2099-12-31T23:59:59"

    def test_cleanup_expired_removes_old_docs(self, kb):
        """Test that cleanup_expired removes expired documents."""
        # Add a document that already expired
        doc_id = kb.add("Expired doc", "test", expires_at="2020-01-01T00:00:00")
        assert kb.get(doc_id) is not None

        # Run cleanup
        deleted = kb.cleanup_expired()
        assert deleted == 1
        assert kb.get(doc_id) is None

    def test_cleanup_expired_keeps_valid_docs(self, kb):
        """Test that cleanup_expired keeps non-expired documents."""
        # Add a document that expires in the future
        doc_id = kb.add("Future doc", "test", expires_at="2099-12-31T23:59:59")

        # Run cleanup
        deleted = kb.cleanup_expired()
        assert deleted == 0
        assert kb.get(doc_id) is not None

    def test_cleanup_expired_keeps_no_expiration_docs(self, kb):
        """Test that documents without expiration are kept."""
        doc_id = kb.add("No expiration", "test")

        deleted = kb.cleanup_expired()
        assert deleted == 0
        assert kb.get(doc_id) is not None


class TestDeduplication:
    """Test deduplication functionality."""

    def test_find_duplicate_exists(self, kb):
        """Test finding an existing duplicate."""
        doc_id = kb.add("Unique content here", "test")

        found_id = kb.find_duplicate("Unique content here")
        assert found_id == doc_id

    def test_find_duplicate_not_exists(self, kb):
        """Test finding no duplicate."""
        kb.add("Some content", "test")

        found_id = kb.find_duplicate("Different content")
        assert found_id is None

    def test_add_with_check_duplicate(self, kb):
        """Test add with check_duplicate flag."""
        doc_id1 = kb.add("Same content", "test1", check_duplicate=True)
        doc_id2 = kb.add("Same content", "test2", check_duplicate=True)

        # Should return the same ID since content is duplicate
        assert doc_id1 == doc_id2
        assert kb.count() == 1

    def test_add_without_check_duplicate(self, kb):
        """Test add without check_duplicate allows duplicates."""
        doc_id1 = kb.add("Same content", "test1", check_duplicate=False)
        doc_id2 = kb.add("Same content", "test2", check_duplicate=False)

        # Should create two separate documents
        assert doc_id1 != doc_id2
        assert kb.count() == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
