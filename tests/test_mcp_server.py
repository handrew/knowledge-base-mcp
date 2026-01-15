"""
Tests for the MCP server.
Run with: pytest tests/test_mcp_server.py -v

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

    with patch.dict(os.environ, {"KB_DB_PATH": temp_db_path, "KB_BACKEND": "sqlite"}):
        # Force reimport with new env
        if "knowledge_base_mcp" in sys.modules:
            del sys.modules["knowledge_base_mcp"]

        import knowledge_base_mcp as server

        # Create fresh KB with correct model
        from backends import BackendConfig
        config = BackendConfig(backend_type="sqlite", connection_string=temp_db_path)
        server.kb = kb_module.KnowledgeBase(backend_config=config, model_name=test_model)

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
        assert "backend_type" in stats
        assert "connection_info" in stats
        assert "current_model" in stats
        assert stats["backend_type"] == "sqlite"

    def test_add_documents_batch(self, mcp_tools):
        """Test add_documents tool for batch adding."""
        docs = [
            {"content": "Batch doc 1", "source": "batch"},
            {"content": "Batch doc 2", "source": "batch"},
        ]
        result = mcp_tools.add_documents(docs)
        assert result["added"] == 2
        assert len(result["ids"]) == 2

    def test_list_backends(self, mcp_tools):
        """Test list_backends tool."""
        result = mcp_tools.list_backends()
        assert "backends" in result
        assert "current_backend" in result
        assert result["current_backend"] == "sqlite"

        # Should have at least SQLite
        backend_names = [b["name"] for b in result["backends"]]
        assert "sqlite" in backend_names

    def test_update_document(self, mcp_tools):
        """Test update_document tool."""
        # Add a document first
        result = mcp_tools.add_document("Original content", "test")
        doc_id = result["id"]

        # Update it
        update_result = mcp_tools.update_document(doc_id, content="Updated content")
        assert update_result["updated"] is True
        assert update_result["document"]["content"] == "Updated content"

    def test_update_document_source_only(self, mcp_tools):
        """Test updating only the source."""
        result = mcp_tools.add_document("Test content", "old_source")
        doc_id = result["id"]

        update_result = mcp_tools.update_document(doc_id, source="new_source")
        assert update_result["updated"] is True
        assert update_result["document"]["source"] == "new_source"
        assert update_result["document"]["content"] == "Test content"

    def test_update_document_not_found(self, mcp_tools):
        """Test updating non-existent document."""
        result = mcp_tools.update_document(99999, content="New content")
        assert result["updated"] is False
        assert "error" in result

    def test_append_to_document(self, mcp_tools):
        """Test append_to_document tool."""
        # Add a document first
        result = mcp_tools.add_document("First part", "test")
        doc_id = result["id"]

        # Append to it
        append_result = mcp_tools.append_to_document(doc_id, "Second part")
        assert append_result["appended"] is True
        assert "First part" in append_result["document"]["content"]
        assert "Second part" in append_result["document"]["content"]

    def test_append_to_document_custom_separator(self, mcp_tools):
        """Test append with custom separator."""
        result = mcp_tools.add_document("Line 1", "test")
        doc_id = result["id"]

        append_result = mcp_tools.append_to_document(doc_id, "Line 2", separator=" -> ")
        assert append_result["appended"] is True
        assert append_result["document"]["content"] == "Line 1 -> Line 2"

    def test_append_to_document_not_found(self, mcp_tools):
        """Test appending to non-existent document."""
        result = mcp_tools.append_to_document(99999, "New content")
        assert result["appended"] is False
        assert "error" in result


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


class TestMCPMetadataTools:
    """Test metadata-related MCP tools."""

    def test_add_document_with_metadata(self, mcp_tools):
        """Test add_document with metadata."""
        result = mcp_tools.add_document(
            "Test content with metadata",
            "test",
            metadata={"project": "demo", "priority": "high"}
        )
        assert result["status"] == "added"
        assert result["metadata"] == {"project": "demo", "priority": "high"}

    def test_add_document_with_expiration(self, mcp_tools):
        """Test add_document with expiration."""
        result = mcp_tools.add_document(
            "Expiring content",
            "test",
            expires_at="2099-12-31T23:59:59"
        )
        assert result["status"] == "added"
        assert result["expires_at"] == "2099-12-31T23:59:59"

    def test_update_document_metadata(self, mcp_tools):
        """Test update_document with metadata."""
        # Add a document
        add_result = mcp_tools.add_document("Original content", "test")
        doc_id = add_result["id"]

        # Update metadata
        update_result = mcp_tools.update_document(
            doc_id,
            metadata={"status": "reviewed", "reviewer": "test"}
        )
        assert update_result["updated"] is True
        assert update_result["document"]["metadata"] == {"status": "reviewed", "reviewer": "test"}

    def test_list_documents_tool(self, mcp_tools):
        """Test list_documents tool."""
        # Add documents with metadata
        mcp_tools.add_document("Doc A", "test", metadata={"type": "alpha"})
        mcp_tools.add_document("Doc B", "test", metadata={"type": "beta"})

        # List all
        result = mcp_tools.list_documents()
        assert "documents" in result
        assert result["count"] >= 2

    def test_list_documents_with_filter(self, mcp_tools):
        """Test list_documents with metadata filter."""
        # Add documents with different metadata
        mcp_tools.add_document("Filtered doc 1", "filter_test", metadata={"category": "special"})
        mcp_tools.add_document("Filtered doc 2", "filter_test", metadata={"category": "normal"})

        # Filter by metadata
        result = mcp_tools.list_documents(metadata_filter={"category": "special"})
        assert result["count"] >= 1
        for doc in result["documents"]:
            if "metadata" in doc and doc["metadata"]:
                if "category" in doc["metadata"]:
                    assert doc["metadata"]["category"] == "special"

    def test_cleanup_expired_tool(self, mcp_tools):
        """Test cleanup_expired tool."""
        # Add expired document
        mcp_tools.add_document(
            "Old expired content",
            "expired_test",
            expires_at="2020-01-01T00:00:00"
        )

        # Run cleanup
        result = mcp_tools.cleanup_expired()
        assert "deleted" in result
        assert result["deleted"] >= 1

    def test_find_duplicate_tool(self, mcp_tools):
        """Test find_duplicate tool."""
        # Add a document
        add_result = mcp_tools.add_document("Unique text for dedup test", "test")
        doc_id = add_result["id"]

        # Find duplicate
        result = mcp_tools.find_duplicate("Unique text for dedup test")
        assert result["duplicate_found"] is True
        assert result["doc_id"] == doc_id

    def test_find_duplicate_not_found(self, mcp_tools):
        """Test find_duplicate when no duplicate exists."""
        result = mcp_tools.find_duplicate("This content does not exist in the KB")
        assert result["duplicate_found"] is False
        assert result["doc_id"] is None

    def test_add_document_check_duplicate(self, mcp_tools):
        """Test add_document with check_duplicate flag."""
        # Add first document
        result1 = mcp_tools.add_document("Duplicate test content", "test1", check_duplicate=True)
        doc_id1 = result1["id"]

        # Try to add same content with check_duplicate
        result2 = mcp_tools.add_document("Duplicate test content", "test2", check_duplicate=True)
        doc_id2 = result2["id"]

        # Should return same ID
        assert doc_id1 == doc_id2


class TestMigration:
    """Test database migration scenarios."""

    def test_model_switch_preserves_data(self, tmp_path):
        """Test that switching models preserves document data."""
        import knowledge_base as kb_module
        from backends import BackendConfig

        db_path = str(tmp_path / "migration_test.db")

        # Create KB with 768-dim model and add a doc
        config = BackendConfig(backend_type="sqlite", connection_string=db_path)
        kb1 = kb_module.KnowledgeBase(backend_config=config, model_name="BAAI/bge-base-en-v1.5")
        kb1.add("Test document for migration", "test")
        assert kb1.count() == 1
        kb1.close()

        # Reopen with 1024-dim model
        kb2 = kb_module.KnowledgeBase(backend_config=config, model_name="BAAI/bge-m3")

        # The doc should still exist
        assert kb2.count() == 1

        # Semantic search works (uses stored embeddings, dimension mismatch is handled)
        results = kb2.search_semantic("migration", limit=5)
        assert isinstance(results, list)
        # Results may be empty or have low scores since embeddings are different dimensions
        # but it shouldn't crash

        kb2.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
