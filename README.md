# Local Knowledge Base MCP Server

A local-first knowledge base with semantic (vector) and keyword (FTS5) search, exposed via MCP.

## Features

- **Semantic search**: Vector similarity using sentence-transformers (default: `BAAI/bge-m3`)
- **Keyword search**: SQLite FTS5 with BM25 ranking
- **Hybrid search**: Combines both for best results
- **File ingestion**: Chunk and ingest text files
- **Model portability**: Track which model embedded each document, re-embed when switching models
- **Portable**: Single SQLite database file

## Installation

```bash
cd knowledge-base-mcp
pip install -r requirements.txt
```

### Optional: sqlite-vss for vector search

If `sqlite-vss` fails to install, the server will still work with keyword search only (semantic search falls back to keyword).

```bash
# macOS
pip install sqlite-vss

# If that fails:
brew install sqlite
pip install sqlite-vss --no-binary :all:
```

## Configuration for Claude Code

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

Default database path: `~/.local/share/knowledge-base/kb.db`

## Available Tools

| Tool | Description |
|------|-------------|
| `search(query, mode, limit)` | Search with mode: "semantic", "keyword", or "hybrid" |
| `add_document(content, source)` | Add a single document |
| `add_documents(documents)` | Batch add multiple documents |
| `delete_document(doc_id)` | Remove a document by ID |
| `get_document(doc_id)` | Retrieve a document by ID |
| `list_sources()` | Show all sources with document counts |
| `stats()` | Get KB statistics (docs, sources, models used) |
| `ingest_file(file_path)` | Import and chunk a text file |
| `list_models()` | Show available embedding models |
| `reembed(target_model)` | Re-embed all documents with a different model |

## Usage Examples

```python
# Search
search("how to configure webpack", mode="hybrid", limit=5)

# Add documents
add_document("Your text here...", source="notes")

# Ingest a file
ingest_file("~/notes/project.md")

# Check what's in the KB
stats()

# Switch embedding models
list_models()
reembed("BAAI/bge-base-en-v1.5")
```

## Embedding Models

Available models (configured in `knowledge_base.py`):

| Model | Dimensions | Context | Notes |
|-------|------------|---------|-------|
| `BAAI/bge-m3` | 1024 | 8192 | Default, best quality |
| `BAAI/bge-base-en-v1.5` | 768 | 512 | Faster, smaller |
| `nomic-ai/nomic-embed-text-v1.5` | 768 | 8192 | Requires `trust_remote_code` |

Each document stores which model was used for its embedding. Use `reembed()` to migrate all documents to a new model.

## Running Tests

```bash
pytest test_kb.py test_mcp_server.py -v
```
