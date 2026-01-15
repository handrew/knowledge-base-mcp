# MCP Tools Reference

The knowledge base exposes 13 tools via MCP (Model Context Protocol).

## Search

| Tool | Description |
|------|-------------|
| `search(query, mode, limit, metadata_filter, min_score, content_preview_length)` | Search with mode: "semantic", "keyword", or "hybrid". Filter by metadata, score threshold, and truncate content. |

**New parameters:**
- `min_score`: Filter out results below this relevance threshold (0.0-1.0)
- `content_preview_length`: Truncate content to N characters (useful for large docs)

## Document CRUD

| Tool | Description |
|------|-------------|
| `add_document(content, source, metadata, expires_at, expires_in, check_duplicate)` | Add a document with optional metadata and expiration |
| `add_documents(documents, check_duplicate)` | Batch add with efficient batch embedding |
| `get_document(doc_id)` | Retrieve a document by ID |
| `update_document(doc_id, content, source, metadata, expires_at, metadata_merge)` | Update document, optionally merge metadata |
| `append_to_document(doc_id, content, separator)` | Append content to an existing document |
| `delete_document(doc_id)` | Remove a document by ID |

**New parameter:**
- `expires_in`: TTL as seconds (int) or duration string ("1h", "30m", "7d") - easier than ISO timestamps

## Bulk Operations

| Tool | Description |
|------|-------------|
| `delete_by_filter(source, metadata_filter)` | Delete all documents matching filter |
| `update_by_filter(source, metadata_filter, new_source, new_metadata, metadata_merge)` | Update all documents matching filter |

## Document Management

| Tool | Description |
|------|-------------|
| `list_documents(source, metadata_filter, limit, offset)` | List/filter documents with pagination |
| `find_duplicate(content)` | Check if content already exists |
| `cleanup_expired()` | Manually trigger expiration cleanup |

## System

| Tool | Description |
|------|-------------|
| `stats()` | Get KB statistics (doc count, backend type, current model) |

**Not exposed as MCP tools** (use Python API directly):
- `reembed()` - Rare admin operation for switching embedding models
- `ingest_file()` - File ingestion with chunking (agents can do this themselves)
- `list_backends()` - One-time setup operation

## Examples

```python
# Search with quality filter and preview
search("how to configure webpack", mode="hybrid", limit=5, min_score=0.5)
search("python", metadata_filter={"topic": "programming"}, content_preview_length=200)

# Add document with TTL (easier than ISO timestamps)
add_document(
    "Session context for current task...",
    source="session",
    metadata={"session_id": "abc123"},
    expires_in="1h"  # or expires_in=3600 for seconds
)

# Add documents with metadata and expiration
add_document(
    "Project Alpha meeting notes...",
    source="meetings",
    metadata={"project": "alpha", "type": "notes", "private": True},
    expires_at="2025-12-31T23:59:59"
)

# Batch add (uses efficient batch embedding)
add_documents([
    {"content": "Doc 1", "source": "batch", "metadata": {"batch": True}},
    {"content": "Doc 2", "source": "batch", "metadata": {"batch": True}},
])

# Add with deduplication check
add_document("Same content", source="notes", check_duplicate=True)

# Update a document with metadata merge
update_document(doc_id=1, metadata={"reviewed": True}, metadata_merge=True)  # Preserves existing metadata

# Append to existing document
append_to_document(doc_id=1, content="Additional notes...")

# Bulk delete by filter
delete_by_filter(source="old_source")  # Delete all docs from this source
delete_by_filter(metadata_filter={"deprecated": True})  # Delete by metadata

# Bulk update by filter
update_by_filter(
    source="old_source",
    new_source="new_source"
)
update_by_filter(
    metadata_filter={"project": "alpha"},
    new_metadata={"status": "archived"},
    metadata_merge=True  # Merge instead of replace
)

# List documents with filters
list_documents(source="meetings", metadata_filter={"project": "alpha"}, limit=10)

# Check for duplicates before adding
find_duplicate("Some content to check")

# Check what's in the KB
stats()

# Manually clean up expired documents
cleanup_expired()
```
