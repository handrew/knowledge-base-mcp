# Backend Architecture

The knowledge base uses a pluggable backend architecture supporting SQLite (default) and PostgreSQL.

## Architecture

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

## SQLite Backend

The default backend using SQLite with:
- **FTS5** for full-text keyword search (Porter stemmer, unicode61 tokenizer)
- **JSON storage** for embeddings (cosine similarity computed in Python)
- **Trigger-based FTS synchronization**
- **Automatic schema migration**

Configuration:
```json
{
  "KB_DB_PATH": "~/.local/share/knowledge-base/kb.db"
}
```

## PostgreSQL Backend

For larger deployments with:
- **tsvector/tsquery** with GIN index for keyword search
- **Native array storage** for embeddings
- **Full ACID transaction support**
- **JSONB** for metadata with GIN index

Configuration:
```json
{
  "KB_BACKEND": "postgres",
  "KB_DB_URL": "postgresql://user:password@localhost:5432/knowledge_base",
  "KB_SCHEMA": "public"
}
```

Connection string formats:
```
postgresql://user:password@host:port/database
postgres://user:password@host:port/database
host=localhost port=5432 dbname=kb user=postgres password=secret
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `KB_BACKEND` | `sqlite` | Backend type: `sqlite` or `postgres` |
| `KB_DB_PATH` | `~/.local/share/knowledge-base/kb.db` | SQLite database path |
| `KB_DB_URL` | (none) | PostgreSQL connection URL |
| `KB_SCHEMA` | `public` | PostgreSQL schema name |

## Adding a New Backend

1. Create a new file in `backends/` (e.g., `mysql_backend.py`)
2. Implement the `BaseBackend` interface
3. Register in `backends/factory.py`

```python
from backends.base import BaseBackend, BackendConfig

class MyBackend(BaseBackend):
    def __init__(self, config: BackendConfig):
        # Initialize connection
        pass

    # Implement all abstract methods:
    # - add, get, update, delete, count
    # - search_keyword, get_all_embeddings
    # - update_embedding, get_all_documents
    # - list_sources, get_embedding_stats
    # - list_documents, cleanup_expired
    # - find_duplicate, delete_by_filter, update_by_filter
    # - begin_transaction, commit, rollback
    # - backend_type, connection_info properties
```

## Running PostgreSQL Tests

```bash
TEST_POSTGRES_URL=postgresql://user:pass@localhost/test_kb pytest tests/test_postgres_backend.py -v
```
