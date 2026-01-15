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
def search(
    query: str,
    mode: str = "hybrid",
    limit: int = 5,
    metadata_filter: dict = None,
    min_score: float = None,
    content_preview_length: int = None
) -> list[dict]:
    """
    Search the knowledge base.

    Args:
        query: The search query
        mode: Search mode - "semantic" (vector similarity), "keyword" (FTS), or "hybrid" (both)
        limit: Maximum results to return (default: 5)
        metadata_filter: Filter results by metadata key-value pairs (optional).
                         Only documents matching ALL key-value pairs are returned.
                         Example: {"project": "foo", "private": false}
        min_score: Minimum relevance score threshold (0.0-1.0). Results below this are excluded.
        content_preview_length: If set, truncate content to this many characters (useful for large docs)

    Returns:
        List of matching documents with id, content, source, and relevance score
    """
    if mode == "semantic":
        results = kb.search_semantic(query, limit, metadata_filter)
    elif mode == "keyword":
        results = kb.search_keyword(query, limit, metadata_filter)
    else:
        results = kb.search_hybrid(query, limit, metadata_filter=metadata_filter)

    # Apply min_score filter
    if min_score is not None:
        results = [r for r in results if r["score"] >= min_score]

    # Truncate content if requested
    if content_preview_length is not None:
        for r in results:
            if len(r["content"]) > content_preview_length:
                r["content"] = r["content"][:content_preview_length] + "..."

    return results


def _parse_expires_in(expires_in: str | int) -> str:
    """Convert expires_in to ISO timestamp.

    Args:
        expires_in: Duration as seconds (int) or string like "1h", "30m", "7d"

    Returns:
        ISO format timestamp
    """
    from datetime import datetime, timedelta

    if isinstance(expires_in, int):
        seconds = expires_in
    else:
        # Parse duration string
        s = expires_in.strip().lower()
        if s.endswith("s"):
            seconds = int(s[:-1])
        elif s.endswith("m"):
            seconds = int(s[:-1]) * 60
        elif s.endswith("h"):
            seconds = int(s[:-1]) * 3600
        elif s.endswith("d"):
            seconds = int(s[:-1]) * 86400
        else:
            seconds = int(s)  # Assume seconds if no suffix

    expires = datetime.now() + timedelta(seconds=seconds)
    return expires.isoformat()


@mcp.tool()
def add_document(
    content: str,
    source: str = "manual",
    metadata: dict = None,
    expires_at: str = None,
    expires_in: str | int = None,
    check_duplicate: bool = False
) -> dict:
    """
    Add a document to the knowledge base.

    Args:
        content: The text content to add (1-3 paragraphs recommended)
        source: Optional source identifier (e.g., filename, URL, topic)
        metadata: Optional key-value metadata (e.g., {"project": "foo", "private": true})
        expires_at: Optional expiration timestamp (ISO format, e.g., "2024-12-31T23:59:59")
        expires_in: Optional TTL as seconds (int) or duration string ("1h", "30m", "7d").
                    Ignored if expires_at is set.
        check_duplicate: If True, return existing doc ID if content already exists

    Returns:
        The created document with its ID, or existing ID if duplicate found
    """
    # Convert expires_in to expires_at if needed
    effective_expires_at = expires_at
    if effective_expires_at is None and expires_in is not None:
        effective_expires_at = _parse_expires_in(expires_in)

    doc_id = kb.add(content, source, metadata, effective_expires_at, check_duplicate)
    result = {"id": doc_id, "content": content, "source": source, "status": "added"}
    if metadata:
        result["metadata"] = metadata
    if effective_expires_at:
        result["expires_at"] = effective_expires_at
    return result


@mcp.tool()
def add_documents(documents: list[dict], check_duplicate: bool = False) -> dict:
    """
    Add multiple documents to the knowledge base.

    Args:
        documents: List of dicts with keys:
            - content: str (required)
            - source: str (optional)
            - metadata: dict (optional)
            - expires_at: str (optional, ISO format)
        check_duplicate: If True, skip documents whose content already exists

    Returns:
        Summary of added documents
    """
    ids = kb.add_batch(documents, check_duplicate)
    return {"added": len(ids), "ids": ids}


@mcp.tool()
def update_document(
    doc_id: int,
    content: str = None,
    source: str = None,
    metadata: dict = None,
    expires_at: str = None,
    metadata_merge: bool = False
) -> dict:
    """
    Update an existing document in the knowledge base.

    If content is updated, the embedding is automatically regenerated.
    Only provided fields are updated.

    Args:
        doc_id: The document ID to update
        content: New content (optional)
        source: New source (optional)
        metadata: New metadata dict (optional)
        expires_at: New expiration timestamp (optional)
        metadata_merge: If True, merge with existing metadata; if False, replace (default: False)

    Returns:
        Status of the update with the updated document
    """
    success = kb.update(doc_id, content=content, source=source, metadata=metadata, expires_at=expires_at, metadata_merge=metadata_merge)
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
def delete_by_filter(
    source: str = None,
    metadata_filter: dict = None
) -> dict:
    """
    Delete multiple documents matching the filter criteria.

    At least one filter (source or metadata_filter) must be provided
    to prevent accidental deletion of all documents.

    Args:
        source: Filter by source (optional)
        metadata_filter: Filter by metadata key-value pairs (optional).
                         Only documents matching ALL key-value pairs are deleted.
                         Example: {"project": "foo", "deprecated": true}

    Returns:
        Number of documents deleted
    """
    if source is None and metadata_filter is None:
        return {"error": "At least one filter (source or metadata_filter) must be provided"}
    try:
        deleted = kb.delete_by_filter(source=source, metadata_filter=metadata_filter)
        return {"deleted": deleted}
    except ValueError as e:
        return {"error": str(e)}


@mcp.tool()
def update_by_filter(
    source: str = None,
    metadata_filter: dict = None,
    new_source: str = None,
    new_metadata: dict = None,
    metadata_merge: bool = False
) -> dict:
    """
    Update multiple documents matching the filter criteria.

    At least one filter (source or metadata_filter) must be provided,
    and at least one update (new_source or new_metadata) must be specified.

    Args:
        source: Filter by source (optional)
        metadata_filter: Filter by metadata key-value pairs (optional).
                         Only documents matching ALL key-value pairs are updated.
        new_source: New source value to set (optional)
        new_metadata: New metadata to set (optional)
        metadata_merge: If True, merge with existing metadata; if False, replace (default: False)

    Returns:
        Number of documents updated
    """
    if source is None and metadata_filter is None:
        return {"error": "At least one filter (source or metadata_filter) must be provided"}
    if new_source is None and new_metadata is None:
        return {"error": "At least one update (new_source or new_metadata) must be provided"}
    try:
        updated = kb.update_by_filter(
            source=source,
            metadata_filter=metadata_filter,
            new_source=new_source,
            new_metadata=new_metadata,
            metadata_merge=metadata_merge
        )
        return {"updated": updated}
    except ValueError as e:
        return {"error": str(e)}


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


# Note: reembed is available via Python API (kb.reembed) but not exposed as MCP tool
# since it's a rare admin operation that's better done directly.


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


# Note: list_backends is available via Python API (get_available_backends) but not
# exposed as MCP tool since it's a one-time setup operation.


@mcp.tool()
def list_documents(
    source: str = None,
    metadata_filter: dict = None,
    limit: int = 100,
    offset: int = 0
) -> dict:
    """
    List documents with optional filtering.

    Args:
        source: Filter by source (optional)
        metadata_filter: Filter by metadata key-value pairs (optional).
                         Only documents matching ALL key-value pairs are returned.
                         Example: {"project": "foo", "private": true}
        limit: Maximum number of documents to return (default: 100)
        offset: Number of documents to skip for pagination (default: 0)

    Returns:
        List of documents matching the filters
    """
    docs = kb.list_documents(source=source, metadata_filter=metadata_filter, limit=limit, offset=offset)
    return {"documents": docs, "count": len(docs), "offset": offset, "limit": limit}


@mcp.tool()
def cleanup_expired() -> dict:
    """
    Delete all documents that have expired (expires_at < now).

    This is called automatically on server startup, but can also be
    invoked manually to clean up expired documents.

    Returns:
        Number of documents deleted
    """
    deleted = kb.cleanup_expired()
    return {"deleted": deleted}


@mcp.tool()
def find_duplicate(content: str) -> dict:
    """
    Check if a document with the exact same content already exists.

    Useful for deduplication before adding new documents.

    Args:
        content: The content to check for

    Returns:
        The document ID if a duplicate exists, or null if no duplicate found
    """
    doc_id = kb.find_duplicate(content)
    return {"duplicate_found": doc_id is not None, "doc_id": doc_id}


if __name__ == "__main__":
    mcp.run()
