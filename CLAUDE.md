# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Local-first knowledge base MCP server with semantic (vector) and keyword search. Exposes 16 MCP tools for document storage and retrieval with pluggable database backends (SQLite default, PostgreSQL optional).

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_kb.py -v
pytest tests/test_mcp_server.py -v

# Run single test
pytest tests/test_kb.py::TestBasicOperations::test_add_document -v

# PostgreSQL backend tests (requires running PostgreSQL)
TEST_POSTGRES_URL=postgresql://user:pass@localhost/test_kb pytest tests/test_postgres_backend.py -v

# Run MCP server directly
python knowledge_base_mcp.py
```

## Architecture

```
┌─────────────────────────────────────────────┐
│  knowledge_base_mcp.py                      │  MCP server entry point
│  - 16 @mcp.tool() functions                 │  FastMCP wrapper
│  - Environment variable configuration       │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│  knowledge_base.py                          │  Core logic
│  - KnowledgeBase class                      │
│  - Embedding management (sentence-transformers)
│  - Search orchestration (semantic/keyword/hybrid)
│  - MODELS registry for embedding config     │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│  backends/                                  │
│  ├── base.py      BaseBackend ABC, Document, SearchResult dataclasses
│  ├── factory.py   create_backend(), create_backend_from_url()
│  ├── sqlite_backend.py   FTS5, JSON embeddings
│  └── postgres_backend.py tsvector, array embeddings
└─────────────────────────────────────────────┘
```

**Key design decisions:**
- Semantic search computes cosine similarity in Python (not DB) for backend portability
- Embeddings stored as JSON (SQLite) or native arrays (PostgreSQL)
- FTS5 triggers keep keyword index in sync with docs table
- Backend interface defined in `backends/base.py` - all backends implement `BaseBackend`

## Adding New Embedding Models

Edit `MODELS` dict in `knowledge_base.py`:
```python
MODELS = {
    "model-name": {"dim": 768, "prefix": None, "trust_remote_code": False},
    # prefix: tuple of (query_prefix, doc_prefix) if model uses different prefixes
}
```

## Adding New Backends

1. Create `backends/new_backend.py` implementing `BaseBackend`
2. Register in `backends/factory.py` `_BACKENDS` dict
3. Add to `backends/__init__.py` exports

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `KB_BACKEND` | `sqlite` | Backend type |
| `KB_DB_PATH` | `~/.local/share/knowledge-base/kb.db` | SQLite path |
| `KB_DB_URL` | (none) | PostgreSQL URL |
| `KB_SCHEMA` | `public` | PostgreSQL schema |

## Test Configuration

Tests use `BAAI/bge-base-en-v1.5` (768 dim) by default for speed. The `--fast` flag is enabled by default. Tests create temporary SQLite databases via `temp_db` fixture.
