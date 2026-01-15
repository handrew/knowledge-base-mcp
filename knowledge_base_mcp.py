#!/usr/bin/env python3
"""
MCP Server for Local Knowledge Base
Exposes semantic and keyword search via pluggable backends (SQLite, PostgreSQL)
"""

import os
from pathlib import Path
from mcp.server.fastmcp import FastMCP

from knowledge_base import KnowledgeBase
from backends import BackendConfig, get_available_backends

# Configuration via environment variables
# For SQLite (default):
#   KB_DB_PATH=/path/to/kb.db
# For PostgreSQL:
#   KB_BACKEND=postgres
#   KB_DB_URL=postgresql://user:pass@host:port/database
#   KB_SCHEMA=public (optional)

BACKEND_TYPE = os.environ.get("KB_BACKEND", "sqlite").lower()
DB_URL = os.environ.get("KB_DB_URL", "")
DB_PATH = os.environ.get("KB_DB_PATH", str(Path.home() / ".local" / "share" / "knowledge-base" / "kb.db"))
SCHEMA = os.environ.get("KB_SCHEMA", "public")

# Determine connection string based on backend
if BACKEND_TYPE == "postgres" or BACKEND_TYPE == "postgresql":
    if not DB_URL:
        raise ValueError("KB_DB_URL environment variable is required for PostgreSQL backend")
    connection_string = DB_URL
    backend_config = BackendConfig(
        backend_type="postgres",
        connection_string=connection_string,
        options={"schema": SCHEMA}
    )
else:
    # SQLite - ensure directory exists
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    connection_string = DB_PATH
    backend_config = BackendConfig(
        backend_type="sqlite",
        connection_string=connection_string,
    )

# Initialize
mcp = FastMCP("knowledge-base")
kb = KnowledgeBase(backend_config=backend_config)


@mcp.tool()
def search(query: str, mode: str = "hybrid", limit: int = 5) -> list[dict]:
    """
    Search the knowledge base.

    Args:
        query: The search query
        mode: Search mode - "semantic" (vector similarity), "keyword" (FTS), or "hybrid" (both)
        limit: Maximum results to return (default: 5)

    Returns:
        List of matching documents with id, content, source, and relevance score
    """
    if mode == "semantic":
        return kb.search_semantic(query, limit)
    elif mode == "keyword":
        return kb.search_keyword(query, limit)
    else:
        return kb.search_hybrid(query, limit)


@mcp.tool()
def add_document(content: str, source: str = "manual") -> dict:
    """
    Add a document to the knowledge base.

    Args:
        content: The text content to add (1-3 paragraphs recommended)
        source: Optional source identifier (e.g., filename, URL, topic)

    Returns:
        The created document with its ID
    """
    doc_id = kb.add(content, source)
    return {"id": doc_id, "content": content, "source": source, "status": "added"}


@mcp.tool()
def add_documents(documents: list[dict]) -> dict:
    """
    Add multiple documents to the knowledge base.

    Args:
        documents: List of dicts with 'content' and optional 'source' keys

    Returns:
        Summary of added documents
    """
    ids = kb.add_batch(documents)
    return {"added": len(ids), "ids": ids}


@mcp.tool()
def update_document(doc_id: int, content: str = None, source: str = None) -> dict:
    """
    Update an existing document in the knowledge base.

    If content is updated, the embedding is automatically regenerated.
    Only provided fields are updated.

    Args:
        doc_id: The document ID to update
        content: New content (optional)
        source: New source (optional)

    Returns:
        Status of the update with the updated document
    """
    success = kb.update(doc_id, content=content, source=source)
    if success:
        doc = kb.get(doc_id)
        return {"id": doc_id, "updated": True, "document": doc}
    return {"id": doc_id, "updated": False, "error": "Document not found"}


@mcp.tool()
def append_to_document(doc_id: int, content: str, separator: str = "\n\n") -> dict:
    """
    Append content to an existing document in the knowledge base.

    The new content is added to the end of the existing content,
    and the embedding is regenerated for the combined text.

    Args:
        doc_id: The document ID to append to
        content: Content to append
        separator: Separator between existing and new content (default: double newline)

    Returns:
        Status of the append with the updated document
    """
    success = kb.append(doc_id, content, separator=separator)
    if success:
        doc = kb.get(doc_id)
        return {"id": doc_id, "appended": True, "document": doc}
    return {"id": doc_id, "appended": False, "error": "Document not found"}


@mcp.tool()
def delete_document(doc_id: int) -> dict:
    """
    Delete a document from the knowledge base.

    Args:
        doc_id: The document ID to delete

    Returns:
        Status of the deletion
    """
    success = kb.delete(doc_id)
    return {"id": doc_id, "deleted": success}


@mcp.tool()
def get_document(doc_id: int) -> dict | None:
    """
    Get a specific document by ID.

    Args:
        doc_id: The document ID

    Returns:
        The document or None if not found
    """
    return kb.get(doc_id)


@mcp.tool()
def stats() -> dict:
    """
    Get knowledge base statistics.

    Returns:
        Stats including total documents, backend type, connection info, and current model
    """
    return {
        "total_documents": kb.count(),
        "backend_type": kb.backend_type,
        "connection_info": kb.db_path,
        "current_model": kb.model_name,
    }


@mcp.tool()
def reembed(target_model: str = None) -> dict:
    """
    Re-embed all documents with a different model.

    Use this when switching embedding models to ensure all documents
    use consistent embeddings for accurate similarity search.

    Args:
        target_model: Model to use (e.g., "BAAI/bge-m3", "BAAI/bge-base-en-v1.5").
                      If not specified, uses current model.

    Returns:
        Stats about re-embedding (count, model, whether VSS was rebuilt)
    """
    return kb.reembed(target_model)


@mcp.tool()
def ingest_file(file_path: str, chunk_size: int = 1000, overlap: int = 200) -> dict:
    """
    Ingest a text file into the knowledge base, splitting into chunks.

    Args:
        file_path: Path to the text file
        chunk_size: Target size of each chunk in characters (default: 1000)
        overlap: Overlap between chunks in characters (default: 200)

    Returns:
        Summary of ingested chunks
    """
    path = Path(file_path).expanduser()
    if not path.exists():
        return {"error": f"File not found: {file_path}"}

    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        return {"error": f"Could not read file: {e}"}

    # Simple paragraph-aware chunking
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += ("\n\n" if current_chunk else "") + para
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = para

    if current_chunk:
        chunks.append(current_chunk)

    # Add chunks to KB
    source = path.name
    ids = []
    for chunk in chunks:
        doc_id = kb.add(chunk, source)
        ids.append(doc_id)

    return {
        "file": str(path),
        "chunks_created": len(chunks),
        "ids": ids,
        "source": source
    }


@mcp.tool()
def list_backends() -> dict:
    """
    List available database backends.

    Returns:
        List of backends with name, description, and availability status
    """
    return {
        "backends": get_available_backends(),
        "current_backend": kb.backend_type,
    }


if __name__ == "__main__":
    mcp.run()
