import hashlib
import json
import sqlite3
import uuid
from collections.abc import Callable

import numpy as np

from mdvault.db import serialize_f32
from mdvault.indexer import Chunk, chunk_file


def store_memory(
    conn: sqlite3.Connection,
    content: str,
    embedder: Callable[[list[str]], np.ndarray],
    namespace: str = "",
    source: str = "api",
    metadata: dict | None = None,
) -> dict:
    """Store a memory. Auto-chunks if content > 400 words."""
    mem_id = uuid.uuid4().hex[:8]
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

        conn.execute(
            "INSERT INTO memory_meta (file_id, namespace, source, metadata) VALUES (?, ?, ?, ?)",
            (file_id, namespace, source, meta_json),
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
    conn.execute("DELETE FROM files WHERE file_path LIKE ?", (pattern,))

    return count
