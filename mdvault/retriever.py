import json
import posixpath
import sqlite3
import urllib.error
import urllib.request
from collections.abc import Callable

import numpy as np

from mdvault.db import serialize_f32


def bm25_search(
    conn: sqlite3.Connection,
    query: str,
    top_k: int = 50,
) -> list[dict]:
    """FTS5 BM25 search. Returns ranked results (best first)."""
    # Convert multi-word query to FTS5 OR semantics so partial matches score
    tokens = [token.replace('"', '""') for token in query.split() if token]
    fts_query = " OR ".join(f'"{t}"' for t in tokens) or query

    try:
        rows = conn.execute(
            """
            SELECT
                c.id AS chunk_id,
                f.file_path,
                c.chunk_idx,
                c.content,
                c.raw_content,
                fts.rank AS bm25_rank
            FROM chunks_fts fts
            JOIN chunks c ON c.id = fts.rowid
            JOIN files f ON f.id = c.file_id
            WHERE chunks_fts MATCH ?
            ORDER BY fts.rank ASC
            LIMIT ?
            """,
            (fts_query, top_k),
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    return [dict(row) for row in rows]


def vector_search(
    conn: sqlite3.Connection,
    query_vec: np.ndarray,
    top_k: int = 50,
) -> list[dict]:
    """sqlite-vec exact nearest neighbor search. Returns ranked by distance (closest first)."""
    blob = serialize_f32(query_vec)
    rows = conn.execute(
        """
        SELECT
            v.rowid AS chunk_id,
            v.distance
        FROM chunks_vec v
        WHERE embedding MATCH ?
        ORDER BY distance
        LIMIT ?
        """,
        (blob, top_k),
    ).fetchall()

    if not rows:
        return []

    chunk_ids = [row["chunk_id"] for row in rows]
    placeholders = ",".join("?" * len(chunk_ids))
    details = conn.execute(
        f"""
        SELECT c.id, c.chunk_idx, c.content, c.raw_content, f.file_path
        FROM chunks c
        JOIN files f ON f.id = c.file_id
        WHERE c.id IN ({placeholders})
        """,
        chunk_ids,
    ).fetchall()
    detail_map = {d["id"]: d for d in details}

    results = []
    for row in rows:
        d = detail_map.get(row["chunk_id"])
        if d:
            results.append(
                {
                    "chunk_id": row["chunk_id"],
                    "file_path": d["file_path"],
                    "chunk_idx": d["chunk_idx"],
                    "content": d["content"],
                    "raw_content": d["raw_content"],
                    "distance": row["distance"],
                }
            )
    return results


def rrf_fusion(
    bm25_results: list[dict],
    vec_results: list[dict],
    top_k: int = 5,
    k: int = 60,
) -> list[dict]:
    """Reciprocal Rank Fusion. Ranks are 1-indexed. score = 1/(k+rank_bm25) + 1/(k+rank_vec)."""
    scores: dict[int, float] = {}
    metadata: dict[int, dict] = {}

    for rank, r in enumerate(bm25_results, start=1):
        cid = r["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        metadata[cid] = r

    for rank, r in enumerate(vec_results, start=1):
        cid = r["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        if cid not in metadata:
            metadata[cid] = r

    sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
    results = []
    for cid in sorted_ids[:top_k]:
        m = metadata[cid]
        entry = {
            "chunk_id": m["chunk_id"],
            "file_path": m["file_path"],
            "chunk_idx": m["chunk_idx"],
            "content": m["content"],
            "raw_content": m["raw_content"],
            "score": scores[cid],
        }
        results.append(entry)
    return results


def expand_query_llm(query: str, model: str = "qwen3:0.6b") -> str | None:
    """Expand query via local Ollama LLM. Returns expanded text or None on failure."""
    prompt = (
        f"Expand this search query into a short paragraph (2-3 sentences) that a relevant "
        f"document might contain. Do not explain, just write the paragraph.\n\n"
        f"Query: {query}\n\nParagraph:"
    )
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 100},
        }
    ).encode()

    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            return data.get("response", "").strip() or None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def _dedup_results(results: list[dict], conn: sqlite3.Connection, top_k: int) -> list[dict]:
    """Deduplicate: one result per unique file content (by file_hash), then one per file_path."""
    # Build file_hash lookup for all file_paths in results
    file_paths = list({r["file_path"] for r in results})
    hash_map: dict[str, str] = {}
    if file_paths:
        placeholders = ",".join("?" * len(file_paths))
        rows = conn.execute(
            f"SELECT file_path, file_hash FROM files WHERE file_path IN ({placeholders})",
            file_paths,
        ).fetchall()
        hash_map = {row["file_path"]: row["file_hash"] for row in rows}

    seen_hashes: set[str] = set()
    seen_paths: set[str] = set()
    deduped: list[dict] = []
    for r in results:
        fp = r["file_path"]
        fh = hash_map.get(fp, fp)  # fallback to path if hash not found
        if fh in seen_hashes or fp in seen_paths:
            continue
        seen_hashes.add(fh)
        seen_paths.add(fp)
        deduped.append(r)
        if len(deduped) >= top_k:
            break
    return deduped


def hybrid_search(
    conn: sqlite3.Connection,
    query: str,
    embedder: Callable[[list[str]], np.ndarray],
    top_k: int = 5,
    expand: bool = False,
    expand_model: str = "qwen3:0.6b",
) -> list[dict]:
    """Full hybrid search: BM25 + vector + RRF fusion. Optional LLM query expansion."""
    # BM25 always uses original query (lexical match)
    bm25_results = bm25_search(conn, query, top_k=50)

    # Vector search: optionally use expanded query for richer embedding
    vec_query = query
    if expand:
        expanded = expand_query_llm(query, model=expand_model)
        if expanded:
            vec_query = f"{query} {expanded}"

    query_vec = embedder([vec_query])[0]
    vec_results = vector_search(conn, query_vec, top_k=50)
    # Fuse with extra headroom, then dedup by content hash to avoid duplicated files
    fused = rrf_fusion(bm25_results, vec_results, top_k=top_k * 3)
    return _dedup_results(fused, conn, top_k)


def get_total_chunks(conn: sqlite3.Connection) -> int:
    """Return total number of indexed chunks."""
    return conn.execute("SELECT COUNT(*) as c FROM chunks").fetchone()["c"]


def related_notes(
    conn: sqlite3.Connection,
    file_path: str,
    embedder: Callable[[list[str]], np.ndarray],
    top_k: int = 5,
) -> dict:
    """Find related notes: direct links, backlinks, and semantically similar files."""
    filename = posixpath.basename(file_path)

    # Direct links (outgoing)
    links = [
        row["target_path"]
        for row in conn.execute(
            """
            SELECT DISTINCT l.target_path
            FROM links l
            JOIN files f ON f.id = l.source_file_id
            WHERE f.file_path = ?
            """,
            (file_path,),
        ).fetchall()
    ]

    # Backlinks (incoming) — match by full path OR filename (for wikilinks)
    backlinks = [
        row["file_path"]
        for row in conn.execute(
            """
            SELECT DISTINCT f.file_path
            FROM links l
            JOIN files f ON f.id = l.source_file_id
            WHERE l.target_path = ? OR l.target_path = ?
            """,
            (file_path, filename),
        ).fetchall()
    ]

    # Semantically similar files via vector search on first chunk
    chunk = conn.execute(
        """
        SELECT c.content FROM chunks c
        JOIN files f ON f.id = c.file_id
        WHERE f.file_path = ?
        ORDER BY c.chunk_idx
        LIMIT 1
        """,
        (file_path,),
    ).fetchone()

    similar: list[str] = []
    if chunk:
        query_vec = embedder([chunk["content"]])[0]
        vec_results = vector_search(conn, query_vec, top_k=50)
        seen: set[str] = set()
        for r in vec_results:
            fp = r["file_path"]
            if fp != file_path and fp not in seen:
                seen.add(fp)
                similar.append(fp)
                if len(similar) >= top_k:
                    break

    return {
        "file_path": file_path,
        "links": links,
        "backlinks": backlinks,
        "similar": similar,
    }
