# Python API

The `KnowledgeBase` class provides a Python interface for direct use without MCP.

## Basic Usage

```python
from knowledge_base import KnowledgeBase
from backends import BackendConfig

# SQLite (default)
kb = KnowledgeBase("knowledge_base.db")

# PostgreSQL via URL
kb = KnowledgeBase("postgresql://user:pass@localhost/kb")

# Explicit backend configuration
config = BackendConfig(
    backend_type="postgres",
    connection_string="postgresql://localhost/kb",
    options={"schema": "my_schema"}
)
kb = KnowledgeBase(backend_config=config)
```

## Adding Documents

```python
# Add with metadata and expiration
kb.add(
    "Document content",
    source="manual",
    metadata={"project": "demo", "tags": ["important"]},
    expires_at="2025-12-31T23:59:59",
    check_duplicate=True  # Returns existing ID if content exists
)

# Batch add (efficient batch embedding)
kb.add_batch([
    {"content": "Doc 1", "source": "batch"},
    {"content": "Doc 2", "source": "batch"},
])
```

## Updating Documents

```python
# Update with metadata merge (preserves existing keys)
kb.update(doc_id=1, metadata={"reviewed": True}, metadata_merge=True)

# Append to document
kb.append(doc_id=1, content="More content", separator="\n\n")
```

## Searching

```python
# Basic searches
results = kb.search_hybrid("query", limit=5)
results = kb.search_semantic("query", limit=5)
results = kb.search_keyword("query", limit=5)

# Search with metadata filter
results = kb.search_semantic("query", metadata_filter={"project": "demo"})
```

## Bulk Operations

```python
# Delete by filter
kb.delete_by_filter(source="old_source")

# Bulk update with metadata merge
kb.update_by_filter(
    metadata_filter={"project": "alpha"},
    new_metadata={"status": "archived"},
    metadata_merge=True
)
```

## Document Management

```python
# List and filter documents
docs = kb.list_documents(
    source="manual",
    metadata_filter={"project": "demo"},
    limit=10
)

# Deduplication check
existing_id = kb.find_duplicate("Some content")

# Expiration cleanup (also runs on startup)
deleted_count = kb.cleanup_expired()

# Get document by ID
doc = kb.get(doc_id=1)

# Delete document
kb.delete(doc_id=1)
```

## Embedding Management

```python
# Re-embed all documents with a different model
kb.reembed("BAAI/bge-base-en-v1.5")

# Get embedding stats
stats = kb.get_embedding_stats()

# List available models
models = kb.list_available_models()
```

## Properties

```python
print(f"Backend: {kb.backend_type}")  # "sqlite" or "postgres"
print(f"Connection: {kb.db_path}")
print(f"Document count: {kb.count()}")
```

## Cleanup

```python
kb.close()
```
