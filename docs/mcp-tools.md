# MCP Tools Reference

The knowledge base exposes 12 tools via MCP (Model Context Protocol).

## Search

| Tool | Description |
|------|-------------|
| `search(query, mode, limit, metadata_filter, min_score, content_preview_length)` | Search with mode: "semantic", "keyword", or "hybrid". Returns `{results, total_count}`. |

**Parameters:**
- `min_score`: Filter out results below this relevance threshold (0.0-1.0)
- `content_preview_length`: Truncate content to N characters (useful for large docs)

**Returns:** `{"results": [...], "total_count": N}` - total_count shows how many matches exist (useful for deciding whether to refine query or paginate)

## Document CRUD

| Tool | Description |
|------|-------------|
| `add_document(content, source, metadata, expires_at, expires_in, check_duplicate)` | Add a document with optional metadata and expiration |
| `add_documents(documents, check_duplicate)` | Batch add with efficient batch embedding |
| `get_documents(doc_ids)` | Retrieve one or more documents by ID. Returns `{documents, not_found}` |
| `update_document(doc_id, content, source, metadata, expires_at, metadata_merge)` | Update document, optionally merge metadata |
| `append_to_document(doc_id, content, separator)` | Append content to an existing document |
| `delete_document(doc_id)` | Remove a document by ID |

**Parameters:**
- `expires_in`: TTL as seconds (int) or duration string ("1h", "30m", "7d") - easier than ISO timestamps
- `doc_ids`: Single int or list of ints for batch retrieval

## Bulk Operations

| Tool | Description |
|------|-------------|
| `delete_by_filter(source, metadata_filter)` | Delete all documents matching filter |
| `update_by_filter(source, metadata_filter, new_source, new_metadata, metadata_merge)` | Update all documents matching filter |

## Document Management

| Tool | Description |
|------|-------------|
| `list_documents(source, metadata_filter, limit, offset)` | List/filter documents with pagination |
| `cleanup_expired()` | Manually trigger expiration cleanup |

## System

| Tool | Description |
|------|-------------|
| `stats()` | Get KB statistics (doc count, backend type, current model) |

**Not exposed as MCP tools** (use Python API directly):
- `reembed()` - Rare admin operation for switching embedding models
- `ingest_file()` - File ingestion with chunking (agents can do this themselves)
- `list_backends()` - One-time setup operation
- `find_duplicate()` - Use `check_duplicate=True` on add_document instead

## Examples

```python
# Search with quality filter and preview
result = search("how to configure webpack", mode="hybrid", limit=5, min_score=0.5)
# result = {"results": [...], "total_count": 15}

result = search("python", metadata_filter={"topic": "programming"}, content_preview_length=200)

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

# Get one or more documents by ID
result = get_documents(1)  # Single ID
result = get_documents([1, 2, 3])  # Multiple IDs
# result = {"documents": [...], "not_found": [3]}

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

# Check what's in the KB
stats()

# Manually clean up expired documents
cleanup_expired()
```
