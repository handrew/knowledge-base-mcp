# Local Knowledge Base MCP Server

A local-first knowledge base with semantic (vector) and keyword search, exposed via MCP. Supports multiple database backends (SQLite, PostgreSQL).

## Features

- **Semantic search**: Vector similarity using sentence-transformers
- **Keyword search**: FTS5 (SQLite) or tsvector (PostgreSQL)
- **Hybrid search**: Combines both for best results
- **Pluggable backends**: SQLite (default) and PostgreSQL
- **Metadata & filtering**: Tag documents, filter searches
- **Bulk operations**: Update/delete by filter
- **Document expiration**: TTL support with auto-cleanup
- **Deduplication**: Detect and prevent duplicate content

## Quick Start

```bash
pip install -r requirements.txt
```

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

## Basic Usage

```python
# Search with quality threshold and preview
search("webpack config", mode="hybrid", limit=5, min_score=0.5, content_preview_length=200)

# Add document with TTL
add_document("Session notes...", source="session", metadata={"task": "current"}, expires_in="1h")

# Update with metadata merge
update_document(doc_id=1, metadata={"reviewed": True}, metadata_merge=True)

# Bulk delete by filter
delete_by_filter(source="old_source")

# Search with metadata filter
search("python", metadata_filter={"topic": "programming"})
```

## Documentation

- **[MCP Tools Reference](docs/mcp-tools.md)** - All 13 available tools with examples
- **[Python API](docs/python-api.md)** - Direct Python usage without MCP
- **[Backends](docs/backends.md)** - SQLite/PostgreSQL configuration, adding new backends
- **[Embedding Models](docs/embedding-models.md)** - Available models, switching models

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `KB_BACKEND` | `sqlite` | Backend type: `sqlite` or `postgres` |
| `KB_DB_PATH` | `~/.local/share/knowledge-base/kb.db` | SQLite database path |
| `KB_DB_URL` | (none) | PostgreSQL connection URL |
| `KB_SCHEMA` | `public` | PostgreSQL schema name |

## Running Tests

```bash
pytest tests/ -v
```
