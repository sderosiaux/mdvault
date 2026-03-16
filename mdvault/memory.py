import hashlib
import json
import sqlite3
import uuid
from collections.abc import Callable

import numpy as np

from mdvault.db import serialize_f32
from mdvault.indexer import Chunk, chunk_file

_CONFIDENCE_BY_SOURCE = {"user": 0.7, "agent": 0.5, "promoted": 0.3, "cli": 0.7, "api": 0.5}


def store_memory(
    conn: sqlite3.Connection,
    content: str,
    embedder: Callable[[list[str]], np.ndarray],
    namespace: str = "",
    source: str = "api",
    metadata: dict | None = None,
) -> dict:
    """Store a memory. Auto-chunks if content > 400 words."""
    mem_id = uuid.uuid4().hex[:12]
    file_path = f"memory://{namespace}/{mem_id}"
    file_hash = hashlib.sha256(content.encode()).hexdigest()
    meta_json = json.dumps(metadata or {})

    # Chunk if long, otherwise single chunk
    if len(content.split()) > 400:
        chunks = chunk_file(content)
        chunks = [c for c in chunks if c.content.strip()]
    else:
        chunks = [Chunk(content=content, heading=None)]

    if not chunks:
        chunks = [Chunk(content=content, heading=None)]

    # Build context prefix
    ns_display = f"memory://{namespace}" if namespace else "memory://"

    ctx_texts = []
    raw_texts = []
    for chunk in chunks:
        prefix_parts = [ns_display]
        if chunk.heading:
            prefix_parts.append(chunk.heading)
        prefix = f"[{' > '.join(prefix_parts)}]\n"
        ctx_texts.append(prefix + chunk.content)
        raw_texts.append(chunk.content)

    # Embed
    all_embeddings = []
    for batch_start in range(0, len(raw_texts), 32):
        batch = raw_texts[batch_start : batch_start + 32]
        all_embeddings.append(embedder(batch))
    embeddings = np.concatenate(all_embeddings, axis=0)

    # Atomic write
    conn.execute("SAVEPOINT sp_store_memory")
    try:
        conn.execute("INSERT INTO files (file_path, file_hash) VALUES (?, ?)", (file_path, file_hash))
        file_id = conn.execute("SELECT id FROM files WHERE file_path = ?", (file_path,)).fetchone()["id"]

        confidence = _CONFIDENCE_BY_SOURCE.get(source, 0.5)
        conn.execute(
            "INSERT INTO memory_meta (file_id, namespace, source, metadata, confidence) VALUES (?, ?, ?, ?, ?)",
            (file_id, namespace, source, meta_json, confidence),
        )

        for idx, (chunk, ctx_text, embedding) in enumerate(zip(chunks, ctx_texts, embeddings, strict=True)):
            cursor = conn.execute(
                "INSERT INTO chunks (file_id, chunk_idx, content, raw_content) VALUES (?, ?, ?, ?)",
                (file_id, idx, ctx_text, chunk.content),
            )
            chunk_id = cursor.lastrowid
            conn.execute("INSERT INTO chunks_fts(rowid, content) VALUES (?, ?)", (chunk_id, ctx_text))
            conn.execute("INSERT INTO chunks_vec(rowid, embedding) VALUES (?, ?)", (chunk_id, serialize_f32(embedding)))

        conn.execute("RELEASE SAVEPOINT sp_store_memory")
    except Exception:
        conn.execute("ROLLBACK TO SAVEPOINT sp_store_memory")
        raise

    return {"id": mem_id, "file_id": file_id, "chunks": len(chunks)}


def update_memory(
    conn: sqlite3.Connection,
    id: str,
    embedder: Callable[[list[str]], np.ndarray],
    content: str | None = None,
    metadata: dict | None = None,
) -> None:
    """Update a memory's content and/or metadata."""
    pattern = f"memory://%/{id}"
    file_row = conn.execute("SELECT id, file_path FROM files WHERE file_path LIKE ?", (pattern,)).fetchone()
    if not file_row:
        msg = f"Memory not found: {id}"
        raise ValueError(msg)

    file_id = file_row["id"]

    conn.execute("SAVEPOINT sp_update_memory")
    try:
        if content is not None:
            file_hash = hashlib.sha256(content.encode()).hexdigest()
            conn.execute("UPDATE files SET file_hash = ? WHERE id = ?", (file_hash, file_id))

            # Remove old chunks + fts + vec
            conn.execute("DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE file_id = ?)", (file_id,))
            conn.execute("DELETE FROM chunks_vec WHERE rowid IN (SELECT id FROM chunks WHERE file_id = ?)", (file_id,))
            conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))

            # Re-chunk and re-embed
            if len(content.split()) > 400:
                chunks = chunk_file(content)
                chunks = [c for c in chunks if c.content.strip()]
            else:
                chunks = [Chunk(content=content, heading=None)]

            if not chunks:
                chunks = [Chunk(content=content, heading=None)]

            meta_row = conn.execute("SELECT namespace FROM memory_meta WHERE file_id = ?", (file_id,)).fetchone()
            ns = meta_row["namespace"] if meta_row else ""
            ns_display = f"memory://{ns}" if ns else "memory://"

            raw_texts = [c.content for c in chunks]
            all_embeddings = []
            for batch_start in range(0, len(raw_texts), 32):
                batch = raw_texts[batch_start : batch_start + 32]
                all_embeddings.append(embedder(batch))
            embeddings = np.concatenate(all_embeddings, axis=0)

            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True)):
                prefix_parts = [ns_display]
                if chunk.heading:
                    prefix_parts.append(chunk.heading)
                ctx_text = f"[{' > '.join(prefix_parts)}]\n{chunk.content}"

                cursor = conn.execute(
                    "INSERT INTO chunks (file_id, chunk_idx, content, raw_content) VALUES (?, ?, ?, ?)",
                    (file_id, idx, ctx_text, chunk.content),
                )
                chunk_id = cursor.lastrowid
                conn.execute("INSERT INTO chunks_fts(rowid, content) VALUES (?, ?)", (chunk_id, ctx_text))
                conn.execute(
                    "INSERT INTO chunks_vec(rowid, embedding) VALUES (?, ?)", (chunk_id, serialize_f32(embedding))
                )

        if metadata is not None:
            meta_json = json.dumps(metadata)
            conn.execute("UPDATE memory_meta SET metadata = ? WHERE file_id = ?", (meta_json, file_id))

        conn.execute("UPDATE memory_meta SET updated_at = CURRENT_TIMESTAMP WHERE file_id = ?", (file_id,))

        conn.execute("RELEASE SAVEPOINT sp_update_memory")
    except Exception:
        conn.execute("ROLLBACK TO SAVEPOINT sp_update_memory")
        raise


def delete_memory(
    conn: sqlite3.Connection,
    id: str | None = None,
    namespace: str | None = None,
) -> int:
    """Delete memories by id or namespace. Returns count of deleted memories."""
    if id is None and namespace is None:
        msg = "Must provide either id or namespace"
        raise ValueError(msg)

    if id is not None:
        pattern = f"memory://%/{id}"
    else:
        pattern = f"memory://{namespace}/%"

    count = conn.execute("SELECT COUNT(*) as c FROM files WHERE file_path LIKE ?", (pattern,)).fetchone()["c"]
    if count == 0:
        return 0

    conn.execute(
        "DELETE FROM chunks_fts WHERE rowid IN ("
        "  SELECT c.id FROM chunks c JOIN files f ON c.file_id = f.id WHERE f.file_path LIKE ?"
        ")",
        (pattern,),
    )
    conn.execute(
        "DELETE FROM chunks_vec WHERE rowid IN ("
        "  SELECT c.id FROM chunks c JOIN files f ON c.file_id = f.id WHERE f.file_path LIKE ?"
        ")",
        (pattern,),
    )
    conn.execute(
        "DELETE FROM memory_meta WHERE file_id IN (  SELECT id FROM files WHERE file_path LIKE ?)",
        (pattern,),
    )
    conn.execute("DELETE FROM files WHERE file_path LIKE ?", (pattern,))

    return count
