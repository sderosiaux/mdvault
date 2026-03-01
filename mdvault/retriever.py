import sqlite3
from typing import Callable

import numpy as np

from mdvault.db import serialize_f32


def bm25_search(
    conn: sqlite3.Connection,
    query: str,
    top_k: int = 50,
) -> list[dict]:
    """FTS5 BM25 search. Returns ranked results (best first)."""
    # Convert multi-word query to FTS5 OR semantics so partial matches score
    fts_query = " OR ".join(
        f'"{token}"' for token in query.split() if token
    ) or query

    rows = conn.execute(
        """
        SELECT
            c.id AS chunk_id,
            f.file_path,
            c.chunk_idx,
            c.content,
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

    results = []
    for row in rows:
        chunk_id = row["chunk_id"]
        detail = conn.execute(
            """
            SELECT c.chunk_idx, c.content, f.file_path
            FROM chunks c
            JOIN files f ON f.id = c.file_id
            WHERE c.id = ?
            """,
            (chunk_id,),
        ).fetchone()
        if detail:
            results.append({
                "chunk_id": chunk_id,
                "file_path": detail["file_path"],
                "chunk_idx": detail["chunk_idx"],
                "content": detail["content"],
                "distance": row["distance"],
            })
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
        entry = {
            "chunk_id": metadata[cid]["chunk_id"],
            "file_path": metadata[cid]["file_path"],
            "chunk_idx": metadata[cid]["chunk_idx"],
            "content": metadata[cid]["content"],
            "score": scores[cid],
        }
        results.append(entry)
    return results


def hybrid_search(
    conn: sqlite3.Connection,
    query: str,
    embedder: Callable[[list[str]], np.ndarray],
    top_k: int = 5,
) -> list[dict]:
    """Full hybrid search: BM25 + vector + RRF fusion."""
    bm25_results = bm25_search(conn, query, top_k=50)
    query_vec = embedder([query])[0]
    vec_results = vector_search(conn, query_vec, top_k=50)
    return rrf_fusion(bm25_results, vec_results, top_k=top_k)


def get_total_chunks(conn: sqlite3.Connection) -> int:
    """Return total number of indexed chunks."""
    return conn.execute("SELECT COUNT(*) as c FROM chunks").fetchone()["c"]
