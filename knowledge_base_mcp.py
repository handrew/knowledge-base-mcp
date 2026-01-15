#!/usr/bin/env python3
"""
MCP Server for Local Knowledge Base
Exposes semantic and keyword search via SQLite FTS5 + sqlite-vss
"""

import os
from pathlib import Path
from mcp.server.fastmcp import FastMCP

from knowledge_base import KnowledgeBase

# Configuration
DB_PATH = os.environ.get("KB_DB_PATH", str(Path.home() / ".local" / "share" / "knowledge-base" / "kb.db"))

# Ensure directory exists
Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

# Initialize
mcp = FastMCP("knowledge-base")
kb = KnowledgeBase(DB_PATH)


@mcp.tool()
def search(query: str, mode: str = "hybrid", limit: int = 5) -> list[dict]:
    """
    Search the knowledge base.

    Args:
        query: The search query
        mode: Search mode - "semantic" (vector similarity), "keyword" (FTS5), or "hybrid" (both)
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
        Stats including total documents, database path, and vector search status
    """
    return {
        "total_documents": kb.count(),
        "database_path": DB_PATH,
        "vector_search_enabled": kb.vec_available,
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


if __name__ == "__main__":
    mcp.run()
