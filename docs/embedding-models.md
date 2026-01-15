# Embedding Models

The knowledge base uses sentence-transformers for generating vector embeddings.

## Available Models

| Model | Dimensions | Notes |
|-------|------------|-------|
| `BAAI/bge-m3` | 1024 | Default, best quality |
| `BAAI/bge-base-en-v1.5` | 768 | Faster, smaller |
| `nomic-ai/nomic-embed-text-v1.5` | 768 | Requires `trust_remote_code` |

## Model Tracking

Each document stores which model was used for its embedding:
- Documents embedded with different models may have different dimensions
- Semantic search works by comparing embeddings in memory
- Dimension mismatches are handled gracefully (results may have lower scores)

## Switching Models

Use `reembed()` to migrate all documents to a new model:

```python
# Via MCP tool
reembed("BAAI/bge-base-en-v1.5")

# Via Python API
kb.reembed("BAAI/bge-base-en-v1.5")
```

This process:
1. Loads the target model
2. Re-embeds all documents in batches
3. Updates the stored embeddings
4. Updates the model name on each document

## Adding New Models

Edit `MODELS` in `knowledge_base.py`:

```python
MODELS = {
    "BAAI/bge-m3": {"dim": 1024, "prefix": None, "trust_remote_code": False},
    "BAAI/bge-base-en-v1.5": {"dim": 768, "prefix": None, "trust_remote_code": False},
    "nomic-ai/nomic-embed-text-v1.5": {"dim": 768, "prefix": ("search_query: ", "search_document: "), "trust_remote_code": True},
    # Add new models here
    "your-model/name": {"dim": 512, "prefix": None, "trust_remote_code": False},
}
```

Configuration options:
- `dim`: Embedding dimension
- `prefix`: Tuple of (query_prefix, document_prefix) if the model uses different prefixes
- `trust_remote_code`: Whether to trust remote code (required by some models)

## Embedding Stats

Check what models are in use:

```python
# Via MCP tool
stats()

# Via Python API
kb.get_embedding_stats()
# Returns: {"current_model": "BAAI/bge-m3", "models": [{"model": "BAAI/bge-m3", "count": 100}]}
```
