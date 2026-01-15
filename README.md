# Local Knowledge Base MCP Server

A local-first knowledge base with semantic (vector) and keyword search, exposed via MCP. Supports multiple database backends (SQLite, PostgreSQL).

## Features

- **Pluggable backends**: SQLite (default) and PostgreSQL support
- **Semantic search**: In-memory vector similarity using sentence-transformers (default: `BAAI/bge-m3`)
- **Keyword search**: FTS5 (SQLite) or tsvector (PostgreSQL) with BM25/ranking
- **Hybrid search**: Combines both for best results
- **File ingestion**: Chunk and ingest text files
- **Model portability**: Track which model embedded each document, re-embed when switching models

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

| Tool | Description |
|------|-------------|
| `search(query, mode, limit)` | Search with mode: "semantic", "keyword", or "hybrid" |
| `add_document(content, source)` | Add a single document |
| `add_documents(documents)` | Batch add multiple documents |
| `delete_document(doc_id)` | Remove a document by ID |
| `get_document(doc_id)` | Retrieve a document by ID |
| `stats()` | Get KB statistics (doc count, backend type, current model) |
| `reembed(target_model)` | Re-embed all documents with a different model |
| `ingest_file(file_path, chunk_size, overlap)` | Ingest a text file, splitting into chunks |
| `list_backends()` | List available backends and current backend |

## Usage Examples

```python
# Search
search("how to configure webpack", mode="hybrid", limit=5)

# Add documents
add_document("Your text here...", source="notes")

# Ingest a file
ingest_file("/path/to/document.txt", chunk_size=1000, overlap=200)

# Check what's in the KB
stats()

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

# Use the knowledge base
kb.add("Document content", source="manual")
results = kb.search_hybrid("query", limit=5)
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
