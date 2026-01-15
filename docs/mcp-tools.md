# MCP Tools Reference

The knowledge base exposes 16 tools via MCP (Model Context Protocol).

## Search

| Tool | Description |
|------|-------------|
| `search(query, mode, limit, metadata_filter)` | Search with mode: "semantic", "keyword", or "hybrid". Filter by metadata. |

## Document CRUD

| Tool | Description |
|------|-------------|
| `add_document(content, source, metadata, expires_at, check_duplicate)` | Add a document with optional metadata and expiration |
| `add_documents(documents, check_duplicate)` | Batch add with efficient batch embedding |
| `get_document(doc_id)` | Retrieve a document by ID |
| `update_document(doc_id, content, source, metadata, expires_at, metadata_merge)` | Update document, optionally merge metadata |
| `append_to_document(doc_id, content, separator)` | Append content to an existing document |
| `delete_document(doc_id)` | Remove a document by ID |

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
| `ingest_file(file_path, chunk_size, overlap)` | Ingest a text file, splitting into chunks |

## System

| Tool | Description |
|------|-------------|
| `stats()` | Get KB statistics (doc count, backend type, current model) |
| `reembed(target_model)` | Re-embed all documents with a different model |
| `list_backends()` | List available backends and current backend |

## Examples

```python
# Search with metadata filter
search("how to configure webpack", mode="hybrid", limit=5)
search("python", metadata_filter={"topic": "programming"})  # Filter by metadata

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

# Ingest a file
ingest_file("/path/to/document.txt", chunk_size=1000, overlap=200)

# Check what's in the KB
stats()

# Manually clean up expired documents
cleanup_expired()

# Switch embedding models
reembed("BAAI/bge-base-en-v1.5")

# List available backends
list_backends()
```
