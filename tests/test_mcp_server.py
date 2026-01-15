"""
Tests for the MCP server.
Run with: pytest test_mcp_server.py -v

Tests the MCP tools directly without spinning up a full server.
"""

import os
import sys
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch

# We need to patch the DB path before importing the server
@pytest.fixture(scope="module")
def temp_db_path():
    """Create a temporary database path for the module."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    # Cleanup
    for suffix in ["", "-shm", "-wal"]:
        path = db_path + suffix
        if os.path.exists(path):
            os.remove(path)


@pytest.fixture(scope="module")
def mcp_tools(temp_db_path):
    """Import MCP server with patched DB path and model."""
    import knowledge_base as kb_module

    # Use smaller model for tests
    test_model = "BAAI/bge-base-en-v1.5"
    original_model = kb_module.DEFAULT_MODEL
    kb_module.DEFAULT_MODEL = test_model

    with patch.dict(os.environ, {"KB_DB_PATH": temp_db_path}):
        # Force reimport with new env
        if "knowledge_base_mcp" in sys.modules:
            del sys.modules["knowledge_base_mcp"]

        import knowledge_base_mcp as server

        # Create fresh KB with correct model dimensions
        server.kb = kb_module.KnowledgeBase(temp_db_path, model_name=test_model)

        # Ensure vec0 table has correct dimensions (drop and recreate if exists)
        if server.kb.vec_available:
            model_dim = kb_module.MODELS[test_model]["dim"]
            server.kb.db.execute("DROP TABLE IF EXISTS docs_vec")
            server.kb.db.execute(f"CREATE VIRTUAL TABLE docs_vec USING vec0(embedding float[{model_dim}])")
            server.kb.db.commit()

        yield server

    kb_module.DEFAULT_MODEL = original_model


class TestMCPTools:
    """Test MCP tool functions."""

    def test_add_document(self, mcp_tools):
        """Test add_document tool."""
        result = mcp_tools.add_document("Test document for MCP", "mcp_test")
        assert result["status"] == "added"
        assert result["id"] >= 1
        assert result["source"] == "mcp_test"

    def test_search(self, mcp_tools):
        """Test search tool with different modes."""
        # Add a document first
        mcp_tools.add_document("Python is great for data science", "test")

        # Test keyword search
        results = mcp_tools.search("Python", mode="keyword", limit=5)
        assert isinstance(results, list)

        # Test hybrid search
        results = mcp_tools.search("programming", mode="hybrid", limit=5)
        assert isinstance(results, list)

        # Test semantic search
        results = mcp_tools.search("coding", mode="semantic", limit=5)
        assert isinstance(results, list)

    def test_semantic_search_scores_are_meaningful(self, mcp_tools):
        """Test that semantic search returns meaningful scores that differentiate results."""
        # Add documents with varying relevance to a topic
        mcp_tools.add_document("Machine learning is a subset of artificial intelligence", "ml_test")
        mcp_tools.add_document("Neural networks are used in deep learning", "ml_test")
        mcp_tools.add_document("The weather today is sunny and warm", "weather_test")

        # Search for ML-related content
        results = mcp_tools.search("artificial intelligence and machine learning", mode="semantic", limit=3)

        assert len(results) >= 2, "Should return at least 2 results"

        # Scores should be meaningfully different (not all the same tiny value)
        scores = [r["score"] for r in results]
        assert all(s > 0.01 for s in scores), f"Scores should be meaningful, got {scores}"

        # Scores should vary (not all identical)
        if len(scores) >= 2:
            assert max(scores) != min(scores), f"Scores should vary, got {scores}"

        # ML-related docs should score higher than weather doc
        ml_scores = [r["score"] for r in results if "ml_test" in r.get("source", "")]
        weather_scores = [r["score"] for r in results if "weather_test" in r.get("source", "")]
        if ml_scores and weather_scores:
            assert max(ml_scores) > max(weather_scores), "ML docs should rank higher than weather doc"

    def test_keyword_search_scores_are_meaningful(self, mcp_tools):
        """Test that keyword search returns meaningful scores."""
        # Add more documents to get meaningful BM25 scores (IDF needs corpus variety)
        mcp_tools.add_document("PostgreSQL is a powerful relational database", "db_test")
        mcp_tools.add_document("MongoDB is a popular NoSQL database", "db_test")
        mcp_tools.add_document("Redis is an in-memory data store", "db_test")
        mcp_tools.add_document("Cooking recipes for Italian pasta dishes", "food_test")
        mcp_tools.add_document("Travel guide to European destinations", "travel_test")

        # Search for database-related content
        results = mcp_tools.search("database", mode="keyword", limit=5)

        assert len(results) >= 2, "Should return at least 2 results"

        # All results should contain the search term
        for r in results:
            assert "database" in r["content"].lower(), f"Result should contain 'database': {r['content']}"

        # Scores should be positive
        scores = [r["score"] for r in results]
        assert all(s > 0 for s in scores), f"Keyword scores should be positive, got {scores}"

    def test_get_document(self, mcp_tools):
        """Test get_document tool."""
        # Add and retrieve
        result = mcp_tools.add_document("Retrievable content", "test")
        doc_id = result["id"]

        doc = mcp_tools.get_document(doc_id)
        assert doc is not None
        assert doc["content"] == "Retrievable content"

    def test_delete_document(self, mcp_tools):
        """Test delete_document tool."""
        result = mcp_tools.add_document("To be deleted", "test")
        doc_id = result["id"]

        delete_result = mcp_tools.delete_document(doc_id)
        assert delete_result["deleted"] is True

        # Verify it's gone
        doc = mcp_tools.get_document(doc_id)
        assert doc is None

    def test_stats(self, mcp_tools):
        """Test stats tool."""
        stats = mcp_tools.stats()
        assert "total_documents" in stats
        assert "database_path" in stats
        assert "current_model" in stats
        assert "vector_search_enabled" in stats

    def test_add_documents_batch(self, mcp_tools):
        """Test add_documents tool for batch adding."""
        docs = [
            {"content": "Batch doc 1", "source": "batch"},
            {"content": "Batch doc 2", "source": "batch"},
        ]
        result = mcp_tools.add_documents(docs)
        assert result["added"] == 2
        assert len(result["ids"]) == 2


class TestMCPIngestFile:
    """Test file ingestion."""

    def test_ingest_file(self, mcp_tools, tmp_path):
        """Test ingest_file tool."""
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("First paragraph about testing.\n\nSecond paragraph with more content.")

        result = mcp_tools.ingest_file(str(test_file))
        assert "error" not in result
        assert result["chunks_created"] >= 1
        assert result["source"] == "test.txt"

    def test_ingest_nonexistent_file(self, mcp_tools):
        """Test ingest_file with missing file."""
        result = mcp_tools.ingest_file("/nonexistent/path/file.txt")
        assert "error" in result


class TestMigration:
    """Test database migration scenarios."""

    def test_dimension_mismatch_triggers_rebuild(self, tmp_path):
        """Test that changing models with different dimensions rebuilds the index."""
        import knowledge_base as kb_module

        db_path = str(tmp_path / "migration_test.db")

        # Create KB with 768-dim model and add a doc
        kb1 = kb_module.KnowledgeBase(db_path, model_name="BAAI/bge-base-en-v1.5")
        kb1.add("Test document for migration", "test")
        assert kb1.count() == 1
        kb1.db.close()

        # Reopen with 1024-dim model - should trigger rebuild
        kb2 = kb_module.KnowledgeBase(db_path, model_name="BAAI/bge-m3")

        # The doc should still exist but vec index should be rebuilt (and empty since dims don't match)
        assert kb2.count() == 1

        # Semantic search should still work (falls back or uses rebuilt index)
        results = kb2.search_semantic("migration", limit=5)
        # Results may be empty since old embedding is wrong dimension
        assert isinstance(results, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
