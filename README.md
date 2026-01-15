# Local Knowledge Base MCP Server

A local-first knowledge base with semantic (vector) and keyword search, exposed via MCP. Supports multiple database backends (SQLite, PostgreSQL).

## Features

- **Pluggable backends**: SQLite (default) and PostgreSQL support
- **Semantic search**: In-memory vector similarity using sentence-transformers (default: `BAAI/bge-m3`)
- **Keyword search**: FTS5 (SQLite) or tsvector (PostgreSQL) with BM25/ranking
- **Hybrid search**: Combines both for best results
- **File ingestion**: Chunk and ingest text files
- **Model portability**: Track which model embedded each document, re-embed when switching models
- **Metadata support**: Flexible JSON metadata for tagging/categorization
- **Document expiration**: TTL support with automatic cleanup on startup
- **Deduplication**: Detect and prevent duplicate content
- **Update & append**: Modify existing documents with automatic re-embedding
- **Bulk operations**: Update or delete documents by source/metadata filter
- **Metadata merge**: Merge new metadata with existing instead of replacing
- **Search filtering**: Filter search results by metadata
- **Batch embedding**: Efficient batch processing for multiple documents

## Installation

```bash
cd knowledge-base-mcp
pip install -r requirements.txt

# Optional: For PostgreSQL support
pip install psycopg2-binary
```

## Configuration for Claude Code

### SQLite Backend (Default)

Add to `~/.claude/claude_code_config.json`:

```json
{
  "mcpServers": {
    "kb": {
      "command": "python",
      "args": ["/path/to/knowledge-base-mcp/knowledge_base_mcp.py"],
      "env": {
        "KB_DB_PATH": "~/.local/share/knowledge-base/kb.db"
      }
    }
  }
}
```

### PostgreSQL Backend

```json
{
  "mcpServers": {
    "kb": {
      "command": "python",
      "args": ["/path/to/knowledge-base-mcp/knowledge_base_mcp.py"],
      "env": {
        "KB_BACKEND": "postgres",
        "KB_DB_URL": "postgresql://user:password@localhost:5432/knowledge_base",
        "KB_SCHEMA": "public"
      }
    }
  }
}
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `KB_BACKEND` | `sqlite` | Backend type: `sqlite` or `postgres` |
| `KB_DB_PATH` | `~/.local/share/knowledge-base/kb.db` | SQLite database path |
| `KB_DB_URL` | (none) | PostgreSQL connection URL |
| `KB_SCHEMA` | `public` | PostgreSQL schema name |

## Available Tools

### Search
| Tool | Description |
|------|-------------|
| `search(query, mode, limit, metadata_filter)` | Search with mode: "semantic", "keyword", or "hybrid". Filter by metadata. |

### Document CRUD
| Tool | Description |
|------|-------------|
| `add_document(content, source, metadata, expires_at, check_duplicate)` | Add a document with optional metadata and expiration |
| `add_documents(documents, check_duplicate)` | Batch add with efficient batch embedding |
| `get_document(doc_id)` | Retrieve a document by ID |
| `update_document(doc_id, content, source, metadata, expires_at, metadata_merge)` | Update document, optionally merge metadata |
| `append_to_document(doc_id, content, separator)` | Append content to an existing document |
| `delete_document(doc_id)` | Remove a document by ID |

### Bulk Operations
| Tool | Description |
|------|-------------|
| `delete_by_filter(source, metadata_filter)` | Delete all documents matching filter |
| `update_by_filter(source, metadata_filter, new_source, new_metadata, metadata_merge)` | Update all documents matching filter |

### Document Management
| Tool | Description |
|------|-------------|
| `list_documents(source, metadata_filter, limit, offset)` | List/filter documents with pagination |
| `find_duplicate(content)` | Check if content already exists |
| `cleanup_expired()` | Manually trigger expiration cleanup |
| `ingest_file(file_path, chunk_size, overlap)` | Ingest a text file, splitting into chunks |

### System
| Tool | Description |
|------|-------------|
| `stats()` | Get KB statistics (doc count, backend type, current model) |
| `reembed(target_model)` | Re-embed all documents with a different model |
| `list_backends()` | List available backends and current backend |

## Usage Examples

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

## Python API

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

# Add documents with metadata and expiration
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

# Update with metadata merge
kb.update(doc_id=1, metadata={"reviewed": True}, metadata_merge=True)

# Append to document
kb.append(doc_id=1, content="More content", separator="\n\n")

# Search with metadata filter
results = kb.search_hybrid("query", limit=5)
results = kb.search_semantic("query", metadata_filter={"project": "demo"})

# Bulk operations
kb.delete_by_filter(source="old_source")
kb.update_by_filter(
    metadata_filter={"project": "alpha"},
    new_metadata={"status": "archived"},
    metadata_merge=True
)

# List and filter documents
docs = kb.list_documents(
    source="manual",
    metadata_filter={"project": "demo"},
    limit=10
)

# Deduplication
existing_id = kb.find_duplicate("Some content")

# Expiration cleanup (also runs on startup)
deleted_count = kb.cleanup_expired()

print(f"Backend: {kb.backend_type}")  # "sqlite" or "postgres"

kb.close()
```

## Backend Architecture

The knowledge base uses a pluggable backend architecture:

```
┌─────────────────────────────┐
│  KnowledgeBase Class        │  (knowledge_base.py)
│  - Embedding management     │
│  - Search orchestration     │
└─────────────┬───────────────┘
              │
┌─────────────▼───────────────┐
│  Backend Interface          │  (backends/base.py)
│  - BaseBackend ABC          │
│  - BackendConfig            │
└─────────────┬───────────────┘
              │
     ┌────────┴────────┐
     │                 │
┌────▼────┐      ┌─────▼─────┐
│ SQLite  │      │ PostgreSQL│
│ Backend │      │  Backend  │
└─────────┘      └───────────┘
```

### Adding a New Backend

1. Create a new file in `backends/` (e.g., `mysql_backend.py`)
2. Implement the `BaseBackend` interface
3. Register in `backends/factory.py`

```python
from backends.base import BaseBackend, BackendConfig

class MyBackend(BaseBackend):
    def __init__(self, config: BackendConfig):
        # Initialize connection
        pass

    # Implement all abstract methods...
```

## Embedding Models

Available models (configured in `knowledge_base.py`):

| Model | Dimensions | Notes |
|-------|------------|-------|
| `BAAI/bge-m3` | 1024 | Default, best quality |
| `BAAI/bge-base-en-v1.5` | 768 | Faster, smaller |
| `nomic-ai/nomic-embed-text-v1.5` | 768 | Requires `trust_remote_code` |

Each document stores which model was used for its embedding. Use `reembed()` to migrate all documents to a new model.

## Running Tests

```bash
# All tests
pytest tests/ -v

# Backend tests only
pytest tests/test_backends.py -v

# PostgreSQL tests (requires running PostgreSQL)
TEST_POSTGRES_URL=postgresql://user:pass@localhost/test_kb pytest tests/test_postgres_backend.py -v
```
