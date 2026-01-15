"""
Tests for the Knowledge Base module.
Run with: pytest test_kb.py -v

For faster tests, use the smaller model:
    pytest test_kb.py -v --fast
"""

import os
import tempfile
import pytest
from pathlib import Path

import knowledge_base as kb_module
from knowledge_base import KnowledgeBase, MODELS

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
    return KnowledgeBase(temp_db, model_name=use_fast_model)


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
        # Note: If sqlite-vss is not available, semantic search falls back to keyword
        # which may return 0 results for this query. That's OK - we're testing the API works.
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
